"""
Simulates execution of a schedule to asses elapsed time and whether the execution
fits into given host constraints (memory in particular).

Assumes:
 - task execution record is exact
 - task can utilize all cpus ideally
 - no task inference on the same host
 - completely fair and ideal cpu sharing within a host
 - host2host communication is instantaneous (TODO fix, not that hard)
"""

import logging
from dataclasses import dataclass
from math import isclose
from typing import Any

from cascade.controller.api import ExecutableTaskInstance
from cascade.low.core import Environment, Host, JobExecutionRecord, JobInstance, TaskExecutionRecord

logger = logging.getLogger(__name__)


@dataclass
class HostState:
    def __init__(self, host: Host, record: JobExecutionRecord) -> None:
        self.task_cpusec_remaining: dict[str, float] = {}
        self.datasets: set[tuple[str, str]] = set()
        self.host = host
        self.record = record
        self.task_inputs: dict[str, set[tuple[str, str]]] = {}
        self.mem_record = 0

    def runnable_tasks(self) -> set[str]:
        return {
            k
            for k, v in self.task_cpusec_remaining.items()
            if v > 0
            and not isclose(v, 0)
            and k in self.task_inputs
            and self.task_inputs[k] <= self.datasets
        }

    def remaining_memory_mb(self) -> int:
        consumed_ds = sum((self.record.datasets_mb[e] for e in self.datasets), 0)
        runnable = self.runnable_tasks()
        consumed_ts = sum((self.record.tasks[k].memory_mb for k in runnable), 0)
        if consumed_ds + consumed_ts > self.mem_record:
            self.mem_record = consumed_ds + consumed_ts
            logger.debug(
                f"peak reached for memory with {self.datasets=} and {runnable=} => {consumed_ds=} + {consumed_ts=}"
            )
        if (remaining := self.host.memory_mb - consumed_ds - consumed_ts) < 0:
            raise ValueError(f"host run out of memory by {-remaining}")
        return remaining

    def progress_seconds(self, seconds: float) -> set[str]:
        runnable = self.runnable_tasks()
        rv: set[str] = set()
        if not runnable:
            return rv
        prog_per_task = (seconds * self.host.cpu) / len(runnable)
        for t in runnable:
            if isclose(self.task_cpusec_remaining[t], prog_per_task):
                rv.add(t)
        for t, o in self.record.datasets_mb:
            if t in rv:
                self.datasets.add(
                    (
                        t,
                        o,
                    )
                )
        self.remaining_memory_mb()
        # NOTE it is important we decrease the cpusec remaining only *after*
        # we have created the output datasets and checked the remaining memory,
        # otherwise we would not account for corresponding memory consumption
        for t in runnable:
            self.task_cpusec_remaining[t] -= prog_per_task
        return rv

    def next_event_in_secs(self) -> float:
        runnable = self.runnable_tasks()
        if not runnable:
            return -1
        min_remaining = min(self.task_cpusec_remaining[k] for k in runnable)
        progress_coeff = len(runnable) / self.host.cpu
        return min_remaining * progress_coeff


class SimulatingExecutor:
    def __init__(self, env: Environment, record: JobExecutionRecord) -> None:
        self.hosts = {k: HostState(v, record) for k, v in env.hosts.items()}
        self.env = env
        self.comm_queue: list[str] = []
        self.total_time_secs = 0.0
        self.record = record

    def get_environment(self) -> Environment:
        return self.env

    def run_at(self, task: ExecutableTaskInstance, host: str) -> str:
        self.hosts[host].task_cpusec_remaining[task.name] = self.record.tasks[
            task.name
        ].cpuseconds
        self.hosts[host].remaining_memory_mb()
        self.hosts[host].task_inputs[task.name] = set(
            (w.sourceTask, w.sourceOutput) for w in task.wirings
        )
        return f"{host}-{task.name}"

    def scatter(self, taskName: str, outputName: str, hosts: set[str]) -> str:
        k = f"{taskName}:{outputName}"
        for host in hosts:
            self.hosts[host].datasets.add(
                (
                    taskName,
                    outputName,
                )
            )
            self.hosts[host].remaining_memory_mb()
        self.comm_queue.append(k)
        return k

    def purge(
        self, taskName: str, outputName: str, hosts: set[str] | None = None
    ) -> None:
        k = (
            taskName,
            outputName,
        )
        for host in hosts if hosts else self.hosts:
            if k in self.hosts[host].datasets:
                self.hosts[host].datasets.remove(k)

    def fetch_as_url(self, taskName: str, outputName: str) -> str:
        raise NotImplementedError

    def fetch_as_value(self, taskName: str, outputName: str) -> Any:
        raise NotImplementedError

    def wait_some(self, ids: set[str], timeout_sec: int | None = None) -> set[str]:
        if self.comm_queue:
            rv = {e for e in self.comm_queue}
            self.comm_queue = []
            logger.debug(f"awaited {rv}")
            return rv
        next_events = [host.next_event_in_secs() for host in self.hosts.values()]
        next_event_at = min(
            e for e in next_events if e > 0
        )  # raises in case of no progress possible
        self.total_time_secs += next_event_at
        rv = set()
        for host in self.hosts:
            rv.update(
                f"{host}-{e}" for e in self.hosts[host].progress_seconds(next_event_at)
            )
            self.hosts[host].remaining_memory_mb()
        logger.debug(f"awaited {rv}")
        return rv

    def is_done(self, id_: str) -> bool:
        logger.debug(f"checking for {id_}")
        if id_ in self.comm_queue:
            return True
        host, task_name = id_.split("-", 1)
        return isclose(self.hosts[host].task_cpusec_remaining[task_name], 0)

def placeholder_execution_record(job: JobInstance) -> JobExecutionRecord:
    """We can't just use default factories with simulator because we need the datasets to have the right keys"""
    return JobExecutionRecord(
        tasks={t: TaskExecutionRecord(cpuseconds=1, memory_mb=1) for t in job.tasks},
        datasets_mb={
            (t, o): 1
            for (t, i) in job.tasks.items()
            for o in i.definition.output_schema
        }
    )
