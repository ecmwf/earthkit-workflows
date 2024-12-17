"""
Instant infinite scheduler: everything completes immediately, no OOMs, does not check preconditions, ...

For simulation and tests
"""

import sys
import logging
from typing import Iterable, Any, Callable
from cascade.low.core import DatasetId, TaskId, JobInstance, Environment, Worker, WorkerId
from cascade.controller.core import DatasetStatus, TaskStatus, Event, ActionDatasetTransmit, ActionSubmit, ActionDatasetPurge

logger = logging.getLogger(__name__)

class SimpleEventQueue():
    # In single-host executors the event handling is so easy that we just utilize this class in all of them
    def __init__(self) -> None:
        self.event_queue: list[Event] = []
        self.event_callbacks: list[Callable[[Event], None]] = []

    def drain(self) -> list[Event]:
        rv = self.event_queue
        self.event_queue = []
        return rv

    def any(self) -> bool:
        return bool(self.event_queue)

    def add(self, events: list[Event]) -> None:
        self.event_queue += events
        for callback in self.event_callbacks:
            for event in events:
                callback(event)

    def store_done(self, worker: WorkerId, dataset: DatasetId) -> None:
        # convenience for finished stores
        self.add([Event(at=worker, ds_trans=[(dataset, DatasetStatus.available)], ts_trans=[])])

    def transmit_done(self, action: ActionDatasetTransmit) -> None:
        # convenience for no-op transmits
        self.add([
            Event(
                at=worker,
                ds_trans=[(dataset, DatasetStatus.available)],
                ts_trans=[],
            )
            for dataset in action.ds
            for worker in action.to
        ])

    def submit_done(self, action: ActionSubmit) -> None:
        event = Event(
            at=action.at,
            ts_trans=[(task, TaskStatus.succeeded) for task in action.tasks],
            ds_trans=[(dataset, DatasetStatus.available) for dataset in action.outputs],
        )
        self.add([event])

    def submit_failed(self, action: ActionSubmit) -> None:
        event = Event(
            at=action.at,
            ts_trans=[(task, TaskStatus.failed) for task in action.tasks],
            ds_trans=[],
        )
        self.add([event])

    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        self.event_callbacks.append(callback)

class InstantExecutor():
    def __init__(self, workers: int, job: JobInstance, host_id: str = "hInstant") -> None:
        self.env = Environment(
            workers={f"{host_id}:w{i}": Worker(cpu=1, gpu=0, memory_mb=sys.maxsize) for i in range(workers)},
            colocations=[[f"{host_id}:w{i}" for i in range(workers)]],
        )
        self.job = job
        self.eq = SimpleEventQueue()

    def get_environment(self) -> Environment:
        return self.env

    def submit(self, action: ActionSubmit) -> None:
        self.eq.submit_done(action)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        self.eq.transmit_done(action)

    def purge(self, action: ActionDatasetPurge) -> None:
        pass

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        return ""

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        return b""

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        self.eq.store_done(worker, dataset_id)

    def shutdown(self) -> None:
        pass

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        return self.eq.drain()

    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        self.eq.register_event_callback(callback)
