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
from cascade.low.core import Environment, Host, JobExecutionRecord

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
        logger.debug(
            f"checking for runnable with {self.task_cpusec_remaining=}, {self.datasets=}, {self.task_inputs=}"
        )
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
        return self.host.memory_mb - consumed_ds - consumed_ts

    def progress_seconds(self, seconds: int) -> set[str]:
        runnable = self.runnable_tasks()
        prog_per_task = (seconds * self.host.cpu) / len(runnable)
        for e in runnable:
            self.task_cpusec_remaining[e] -= prog_per_task
        rv = set()
        for t in runnable:
            if isclose(self.task_cpusec_remaining[t], 0):
                rv.add(t)
        for t, o in self.record.datasets_mb:
            if t in rv:
                self.datasets.add(
                    (
                        t,
                        o,
                    )
                )
        if (overflow := self.remaining_memory_mb()) < 0:
            raise ValueError(
                f"host run out of memory by {-overflow} after having finished {rv}"
            )
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
        self.comm_queue: list[tuple[str, str]] = []
        self.total_time_secs = 0.0
        self.record = record

    def get_environment(self) -> Environment:
        return self.env

    def run_at(self, task: ExecutableTaskInstance, host: str) -> str:
        self.hosts[host].task_cpusec_remaining[task.name] = self.record.tasks[
            task.name
        ].cpuseconds
        if (overflow := self.hosts[host].remaining_memory_mb()) < 0:
            raise ValueError(
                f"host {host} run out of memory by {-overflow} when enqueuing task {task.name}"
            )
        self.hosts[host].task_inputs[task.name] = set(
            (w.sourceTask, w.sourceOutput) for w in task.wirings
        )
        return task.name

    def scatter(self, taskName: str, outputName: str, hosts: set[str]) -> str:
        k = (
            taskName,
            outputName,
        )
        for host in hosts:
            self.hosts[host].datasets.add(k)
            if (overflow := self.hosts[host].remaining_memory_mb()) < 0:
                raise ValueError(
                    f"host {host} run out of memory by {-overflow} when scattering dataset {taskName}:{outputName}"
                )
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
            return rv
        next_events = [host.next_event_in_secs() for host in self.hosts.values()]
        next_event_at = min(
            e for e in next_events if e > 0
        )  # raises in case of no progress possible
        self.total_time_secs += next_event_at
        rv = set()
        for host in self.hosts:
            rv.update(self.hosts[host].progress_seconds(next_event_at))
        return rv

    def is_done(self, id_: str) -> bool:
        logger.debug(f"checking for {id_}")
        if id_ in self.comm_queue:
            return True
        for host in self.hosts:
            logger.debug(
                f"checking for {id_} in {host}: {self.hosts[host].task_cpusec_remaining.keys()}"
            )
            if id_ in self.hosts[host].task_cpusec_remaining:
                return isclose(self.hosts[host].task_cpusec_remaining[id_], 0)
        raise ValueError(f"value {id_} found neither in tasks nor datasets")
