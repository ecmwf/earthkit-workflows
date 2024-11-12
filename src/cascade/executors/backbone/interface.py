"""
The interface used by the `Backbone{Local/Controller}Executor`s to relay messages.
"""

from cascade.executors.backbone.serde import Message, DataTransmitObject
from cascade.low.core import Environment
from typing import Protocol, runtime_checkable
from cascade.controller.core import Event
from typing import Callable

HostId = str

# TODO split in server and client interfaces, but mind the overlap
# TODO split the data/message recv interfaces?

@runtime_checkable
class Backbone(Protocol):
    def send_message(self, host: HostId, message: Message) -> None:
        raise NotImplementedError

    def send_event_callback(self) -> Callable[[Event], None]:
        raise NotImplementedError

    def send_data(self, url: str, data: DataTransmitObject) -> None:
        raise NotImplementedError

    def recv_messages(self) -> list[Message]:
        raise NotImplementedError

    def broadcast_shutdown(self) -> None:
        raise NotImplementedError

    def get_environment(self) -> Environment:
        raise NotImplementedError

    def url_of(self, host: HostId) -> str:
        raise NotImplementedError

def merge_environments(environments: list[Environment]) -> Environment:
    """Utility function to handle global environment and make it colocation-aware"""
    workers = {
        worker_id: worker
        for environment in environments
        for worker_id, worker in environment.workers.items()
    }
    colocations = [
        list(environment.workers.keys())
        for environment in environments
    ]
    return Environment(workers=workers, colocations=colocations)
    
