"""
Adapter between the controller and backbone
"""

from typing import Iterable, Any, Callable, cast
from cascade.low.core import Environment
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId, TransmitPayload
from cascade.executors.backbone.interface import Backbone

class BackboneExecutor():
    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def get_environment(self) -> Environment:
        return self.backbone.get_environment()

    def submit(self, action: ActionSubmit) -> None:
        host_id = action.at.split(':', 1)[0]
        self.backbone.send_message(host_id, action)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        # TODO optimize to send less payloads
        for fr in action.fr:
            for to in action.to:
                other_url = self.backbone.url_of(to.split(':', 1)[0])
                payload = TransmitPayload(other_url=other_url, other_worker=to, this_worker=fr, datasets=action.ds, tracing_ctx_host=fr.split(':', 1)[0])
                self.backbone.send_message(fr.split(':', 1)[0], payload)

    def purge(self, action: ActionDatasetPurge) -> None:
        for at in action.at:
            host_id = at.split(':', 1)[0]
            # TODO groupby host instead
            self.backbone.send_message(host_id, action.model_copy(update={'at': [at]}))

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        raise NotImplementedError

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        raise NotImplementedError

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.backbone.broadcast_shutdown()

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        messages = self.backbone.recv_messages()
        for m in messages:
            if not isinstance(m, Event):
                raise TypeError(m)
        return cast(list[Event], messages)

    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        raise NotImplementedError
