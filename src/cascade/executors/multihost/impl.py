"""
Implements the Multihost Router Proxy executor
"""

from typing import runtime_checkable, Protocol, Iterable, Any, cast, TypeVar, Callable
from cascade.low.core import Environment, HostId, WorkerId
from cascade.low.func import maybe_head, pyd_replace
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event
from cascade.executors.multihost.client import Client
from cascade.executors.multihost.event_queue import build_queue
from pydantic import BaseModel
from itertools import groupby
import logging

logger = logging.getLogger(__name__)

class RouterExecutor():
    def __init__(self, urls: dict[str, str]):
        """The remote hosts *must* already be running -- this calls the `get_environment` inside in a blocking fash"""
        urls_lookup = {host: url for host, url in urls.items()}
        writer, self.eq = build_queue()
        self.client = Client(writer, urls_lookup)
        colocations=[
            [inner for inner, worker in localEnv.workers.items()]
            for outer, localEnv in self.client.get_envs().items()
        ] # TODO possibly respect inner executor's own colocations?
        workers={
            inner: worker
            for outer, localEnv in self.client.get_envs().items()
            for inner, worker in localEnv.workers.items()
        }
        self.env = Environment(workers=workers, colocations=colocations)
        
    def get_environment(self) -> Environment:
        return self.env

    def submit(self, action: ActionSubmit) -> None:
        self.client.submit(action.at.host, action)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        if len(action.fr) != 1 or len(action.to) != 1:
            raise NotImplementedError("too many from/to in {action=}")
        fromHost = cast(HostId, maybe_head(action.fr))
        toWorker = cast(WorkerId, maybe_head(action.to))
        if fromHost == toWorker.host:
            self.client.transmit_local(fromHost, action)
        else:
            self.client.transmit_remote(fromHost, toWorker, list(action.ds))

    def purge(self, action: ActionDatasetPurge) -> None:
        for host in action.at:
            self.client.purge(host, pyd_replace(action, at=[host]))

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        return self.client.fetch_as_url(worker.host, worker, dataset_id)

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        return self.client.fetch_as_url(worker.host, worker, dataset_id)

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        return self.client.store_value(worker, dataset_id, data)

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        logger.debug("about to issue wait some")
        self.client.wait_some(timeout_sec)
        logger.debug("about to read events from queue")
        return self.eq.get(timeout_sec)

    def shutdown(self):
        self.client.shutdown()

    def register_event_callback(self, callback: Callable[[Event], None]) -> None:
        raise NotImplementedError
