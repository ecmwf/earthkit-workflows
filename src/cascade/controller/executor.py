"""
Defines the Executor protocol
"""

from typing import runtime_checkable, Protocol, Iterable, Any, Callable
from cascade.low.core import Environment
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId

@runtime_checkable
class Executor(Protocol):
    def get_environment(self) -> Environment:
        """Used by the scheduler to build a schedule"""
        raise NotImplementedError

    def submit(self, action: ActionSubmit) -> None:
        """Run a subgraph at the host asap."""
        raise NotImplementedError

    def transmit(self, action: ActionDatasetTransmit) -> None:
        """A particular dataset (task's output) should be asap scattered to listed hosts."""
        raise NotImplementedError

    def purge(self, action: ActionDatasetPurge) -> None:
        """Listed hosts should asap free said dataset."""
        raise NotImplementedError

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        """URL for downloading a particular result. Does not block, result does
        not need to be ready."""
        raise NotImplementedError

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        """A particular result as a value. If not ready, blocks. Fetching itself
        is also blocking."""
        raise NotImplementedError

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        """Uploads a particular dataset to the worker, to be utilized by subsequent submits.
        Expected usage only in multi-executor setting, to facilitate cross-executor data transfer."""
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        """Blocks until timeout elapses or some events are emitted"""
        raise NotImplementedError

    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        raise NotImplementedError

    # TODO its unfortunate to have both register and wait methods... but for controller-facing, we want
    # wait, for remote, we want callback... ideate and reconcile!
