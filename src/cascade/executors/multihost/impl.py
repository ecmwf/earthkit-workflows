"""
Implements the Multihost Router Proxy executor
"""

from typing import runtime_checkable, Protocol, Iterable, Any
from cascade.low.core import Environment
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event
from cascade.executors.multihost.client import ClientBuilder
from cascade.executors.multihost.event_queue import build_queue


class Executor(Protocol):
    def __init__(self, urls: list[str]):
        """The remote hosts *must* already be running -- this calls the `get_environment` inside in a blocking fash"""
        urls = {f"w{i}": url for i, url in enumerate(urls)}
        self.eq, writer = build_queue()
        self.client = ClientBuilder(writer, urls)
        
    def _worker_expand(self, full_worker: str) -> tuple[str, str]:
        """GlobalWorkerId -> RemoteHostId, LocalWorkerId"""
        g, l = full_worker.split(":", 1)
        return g, l

    def get_environment(self) -> Environment:
        return self.client.get_envs()

    def submit(self, action: ActionSubmit) -> None:
        host, worker = self._worker_expand(action.at)
        lAction = replace(action, to=worker)
        self.client.submit(host, lAction)

    def transmit(self, action: ActionDatasetTransmit) -> None:
        frHost, frWorker = self._worker_expand(cast(str, maybe_head(action.fr)))
        toHost, toWorker = self._worker_expand(cast(str, maybe_head(action.fr)))
        self.client.transmit(frHost, frWorker, toHost, toWorker, action.ds)

    def purge(self, action: ActionDatasetPurge) -> None:
        subs = [self._worker_expand(e) for e in action.at]
        subs.sort(key=lambda e: e[0])
        for host, group in groupby(subs, key=lambda e: e[0]):
            self.client.purge(host, replace(action, at={e[1] for e in g}))

    def fetch_as_url(self, dataset_id: DatasetId) -> str:
        # TODO extend the interface with host
        raise NotImplementedError

    def fetch_as_value(self, dataset_id: DatasetId) -> Any:
        # TODO extend the interface with host
        raise NotImplementedError

    def wait_some(self, timeout_sec: int | None = None) -> list[Event]:
        self.client.wait_some(timeout_sec)
        return self.eq.get(timeout_sec)

    def shutdown(self):
        self.client.shutdown()
