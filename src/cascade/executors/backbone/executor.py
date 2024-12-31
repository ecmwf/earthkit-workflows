"""
Adapter between the controller and backbone
"""

import logging
from typing import Iterable, Any, Callable, cast

from cascade.low.core import Environment
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId, TransmitPayload
from cascade.executors.backbone.serde import DatasetFetch
from cascade.executors.backbone.interface import Backbone

logger = logging.getLogger(__name__)

class BackboneExecutor():
    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def get_environment(self) -> Environment:
        return self.backbone.get_environment()

    def submit(self, action: ActionSubmit) -> None:
        self.backbone.send_message(action.at.host, action)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        # TODO optimize to send less payloads
        for fr in action.fr:
            for to in action.to:
                if fr == to.host:
                    # NOTE this will need to change once we have persistent workers, ie, no default local broadcast
                    logger.warning(f"skipping unnecessary local transfer")
                    continue
                other_url = self.backbone.url_of(to.host)
                payload = TransmitPayload(other_url=other_url, other_worker=to, this_host=fr, datasets=action.ds)
                self.backbone.send_message(fr.split(':', 1)[0], payload)

    def purge(self, action: ActionDatasetPurge) -> None:
        self.backbone.send_message(action.at, action)

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        raise NotImplementedError

    def lazyfetch_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        self.backbone.send_message(worker.host, DatasetFetch(worker=worker, dataset=dataset_id))

    def fetch_as_value(self, dataset_id: DatasetId) -> Any:
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
