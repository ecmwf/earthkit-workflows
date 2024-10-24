"""
Instant infinite scheduler: everything completes immediately, no OOMs

For simulation and tests
"""

import sys
import logging
from typing import Iterable, Any
from cascade.low.core import DatasetId, TaskId, JobInstance, Environment, Worker
from cascade.controller.core import DatasetStatus, TaskStatus, Event, ActionDatasetTransmit, ActionSubmit, ActionDatasetPurge

logger = logging.getLogger(__name__)

class InstantExecutor():
    def __init__(self, workers: int, job: JobInstance) -> None:
        self.env = Environment(workers={f"w{i}": Worker(cpu=1, gpu=0, memory_mb=sys.maxsize) for i in range(workers)})
        self.job = job
        self.event_queue: list[Event] = []

    def get_environment(self) -> Environment:
        return self.env

    def submit(self, action: ActionSubmit) -> None:
        self.event_queue += [
            Event(
                at=action.at,
                ts_trans=[(task, TaskStatus.succeeded)],
                ds_trans=[],
            )
            for task in action.tasks
        ]
        self.event_queue += [
            Event(
                at=action.at,
                ds_trans=[(dataset, DatasetStatus.available)],
                ts_trans=[],
            )
            for dataset in action.outputs
        ]

    def transmit(self, action: ActionDatasetTransmit) -> None:
        self.event_queue += [
            Event(
                at=worker,
                ds_trans=[(dataset, DatasetStatus.available)],
                ts_trans=[],
            )
            for dataset in action.ds
            for worker in action.to
        ]

    def purge(self, action: ActionDatasetPurge) -> None:
        pass

    def fetch_as_url(self, dataset_id: DatasetId) -> str:
        raise NotImplementedError

    def fetch_as_value(self, dataset_id: DatasetId) -> Any:
        raise NotImplementedError

    def wait_some(self, timeout_sec: int | None = None) -> Iterable[Event]:
        rv = self.event_queue
        self.event_queue = []
        return rv
