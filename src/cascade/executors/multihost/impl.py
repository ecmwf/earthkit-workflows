"""
Implements the Multihost Router Proxy executor
"""

from typing import runtime_checkable, Protocol, Iterable, Any, cast, TypeVar
from cascade.low.core import Environment
from cascade.low.func import maybe_head
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId
from cascade.executors.multihost.client import Client
from cascade.executors.multihost.event_queue import build_queue
from pydantic import BaseModel
from itertools import groupby

B = TypeVar("B", bound=BaseModel)
def _replace(model: B, **kwargs) -> B:
    return model.model_copy(update=kwargs)


class RouterExecutor():
    def __init__(self, urls: list[str]):
        """The remote hosts *must* already be running -- this calls the `get_environment` inside in a blocking fash"""
        urls_lookup = {f"w{i}": url for i, url in enumerate(urls)}
        writer, self.eq = build_queue()
        self.client = Client(writer, urls_lookup)
        self.env = Environment(workers={
            f"{outer}:{inner}": worker
            for outer, localEnv in self.client.get_envs().items()
            for inner, worker in localEnv.workers.items()
        })
        
    def _worker_expand(self, full_worker: str) -> tuple[str, str]:
        """GlobalWorkerId -> RemoteHostId, LocalWorkerId"""
        g, l = full_worker.split(":", 1)
        return g, l

    def get_environment(self) -> Environment:
        return self.env

    def submit(self, action: ActionSubmit) -> None:
        host, worker = self._worker_expand(action.at)
        lAction = _replace(action, to=worker)
        self.client.submit(host, lAction)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        frHost, frWorker = self._worker_expand(cast(str, maybe_head(action.fr)))
        toHost, toWorker = self._worker_expand(cast(str, maybe_head(action.fr)))
        self.client.transmit(frHost, frWorker, toHost, toWorker, list(action.ds))

    def purge(self, action: ActionDatasetPurge) -> None:
        subs = [self._worker_expand(e) for e in action.at]
        subs.sort(key=lambda e: e[0])
        for host, group in groupby(subs, key=lambda e: e[0]):
            self.client.purge(host, _replace(action, at=[e[1] for e in group]))

    def fetch_as_url(self, worker: WorkerId, dataset_id: DatasetId) -> str:
        host, localWorker = self._worker_expand(worker)
        return self.client.fetch_as_url(host, localWorker, dataset_id)

    def fetch_as_value(self, worker: WorkerId, dataset_id: DatasetId) -> Any:
        host, localWorker = self._worker_expand(worker)
        return self.client.fetch_as_url(host, localWorker, dataset_id)

    def store_value(self, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        host, localWorker = self._worker_expand(worker)
        return self.client.store_value(host, localWorker, dataset_id, data)

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        self.client.wait_some(timeout_sec)
        # TODO translation of ids back
        return self.eq.get(timeout_sec)

    def shutdown(self):
        self.client.shutdown()
