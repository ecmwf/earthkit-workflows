"""
Defines the Executor protocol
"""

from typing import runtime_checkable, Protocol, Iterable, Any
from cascade.low.core import Environment
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event

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

    def fetch_as_url(self, dataset_id: DatasetId) -> str:
        """URL for downloading a particular result. Does not block, result does
        not need to be ready."""
        raise NotImplementedError

    def fetch_as_value(self, dataset_id: DatasetId) -> Any:
        """A particular result as a value. If not ready, blocks. Fetching itself
        is also blocking."""
        raise NotImplementedError

    # TODO consider alternative implementation with `def reconcile(state: State) -> State
    # which replaces the `notify` module as well
    def wait_some(self, timeout_sec: int | None = None) -> Iterable[Event]:
        """Blocks until timeout elapses or some events are emitted"""
        raise NotImplementedError
