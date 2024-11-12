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
from typing import Any, Iterable, Callable

from cascade.controller.core import DatasetStatus, TaskStatus, Event, ActionDatasetTransmit, ActionSubmit, ActionDatasetPurge
from cascade.low.core import Environment, Worker, JobExecutionRecord, JobInstance, TaskExecutionRecord, TaskId, DatasetId, WorkerId
from cascade.executors.instant import SimpleEventQueue

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    def __init__(self, worker: Worker, record: JobExecutionRecord) -> None:
        self.task_cpusec_remaining: dict[TaskId, float] = {}
        self.datasets: set[DatasetId] = set()
        self.outputs: set[DatasetId] = set()
        self.worker = worker
        self.record = record
        self.task_inputs: dict[TaskId, set[DatasetId]] = {}
        self.mem_record = 0

    def runnable_tasks(self) -> set[TaskId]:
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
        if (remaining := self.worker.memory_mb - consumed_ds - consumed_ts) < 0:
            raise ValueError(f"worker run out of memory by {-remaining}")
        return remaining

    def progress_seconds(self, seconds: float) -> tuple[list[TaskId], list[DatasetId]]:
        # TODO also emit `running` transitions?
        runnable = self.runnable_tasks()
        if not runnable:
            return [], []
        prog_per_task = (seconds * self.worker.cpu) / len(runnable)
        finished_tasks = []
        for t in runnable:
            if isclose(self.task_cpusec_remaining[t], prog_per_task):
                finished_tasks.append(t)
        finished_datasets: list[DatasetId] = []
        for dsid, size in self.record.datasets_mb.items():
            if dsid.task in finished_tasks and dsid in self.outputs:
                self.datasets.add(dsid)
                finished_datasets.append(dsid)
        self.remaining_memory_mb()
        # NOTE it is important we decrease the cpusec remaining only *after*
        # we have created the output datasets and checked the remaining memory,
        # otherwise we would not account for corresponding memory consumption
        for t in runnable:
            self.task_cpusec_remaining[t] -= prog_per_task
        return finished_tasks, finished_datasets

    def next_event_in_secs(self) -> float:
        runnable = self.runnable_tasks()
        if not runnable:
            return -1
        min_remaining = min(self.task_cpusec_remaining[k] for k in runnable)
        progress_coeff = len(runnable) / self.worker.cpu
        return min_remaining * progress_coeff


class SimulatingExecutor:
    def __init__(self, env: Environment, task_inputs: dict[TaskId, set[DatasetId]], record: JobExecutionRecord) -> None:
        self.workers = {k: WorkerState(v, record) for k, v in env.workers.items()}
        self.env = env
        self.comm_queue: list[str] = []
        self.total_time_secs = 0.0
        self.record = record
        self.eq = SimpleEventQueue()
        self.task_inputs = task_inputs

    def get_environment(self) -> Environment:
        return self.env

    def submit(self, action: ActionSubmit) -> None:
        for task in action.tasks:
            self.workers[action.at].task_cpusec_remaining[task] = self.record.tasks[task].cpuseconds
            self.workers[action.at].remaining_memory_mb()
            self.workers[action.at].task_inputs[task] = self.task_inputs.get(task, set())
            self.workers[action.at].outputs.update(action.outputs)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        available = {ds for worker in action.fr for ds in self.workers[worker].datasets}
        for dataset in action.ds:
            if dataset not in available:
                raise ValueError(f"{action=} not possible as we only have {available=}")
            for worker in action.to:
                self.workers[worker].datasets.add(dataset)
                self.workers[worker].remaining_memory_mb()
        self.eq.transmit_done(action)

    def purge(self, action: ActionDatasetPurge) -> None:
        for worker in action.at:
            self.workers[worker].datasets -= set(action.ds)

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        return ""

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        return b""

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        self.eq.store_done(worker, dataset_id)

    def shutdown(self) -> None:
        pass

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        if self.eq.any():
            return self.eq.drain()

        next_event_at: float|None = None
        for worker_state in self.workers.values(): # min() except handling for empty seq
            worker_event_at = worker_state.next_event_in_secs()
            if worker_event_at > 0 and (next_event_at is None or worker_event_at < next_event_at):
                next_event_at = worker_event_at
        if next_event_at is None:
            return []
        logger.debug(f"waited for {next_event_at}")
        self.total_time_secs += next_event_at
        rv = []
        for worker in self.workers:
            tasks, datasets = self.workers[worker].progress_seconds(next_event_at)
            if len(tasks) > 0 or len(datasets) > 0:
                rv.append(Event(
                    at=worker,
                    ts_trans=[(task, TaskStatus.succeeded) for task in tasks],
                    ds_trans=[(dataset, DatasetStatus.available) for dataset in datasets],
                ))
        return rv
 
    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        raise NotImplementedError

def placeholder_execution_record(job: JobInstance) -> JobExecutionRecord:
    """We can't just use default factories with simulator because we need the datasets to have the right keys"""
    return JobExecutionRecord(
        tasks={t: TaskExecutionRecord(cpuseconds=1, memory_mb=1) for t in job.tasks},
        datasets_mb={
            DatasetId(t, o): 1
            for (t, i) in job.tasks.items()
            for o in i.definition.output_schema
        }
    )
