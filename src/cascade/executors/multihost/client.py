"""
Implements the client for remote calling of `controller.executor` interface, exposed by
the `executor.multihost.server`. All methods are trivial except for the `wait_some` which
submits the requests into a thread pool and the results are gathered into a queue.
"""

import httpx
from cascade.executors.multihost.event_queue import Writer
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId
from cascade.executors.multihost.worker_server import TransmitPayload
from typing import Any
from cascade.low.core import Environment
from cascade.low.func import pyd_replace
import logging

logger = logging.getLogger(__name__)

class _Sync():
    """Synchronisation primitive between a worker and a controller: a condition & a bool"""
    # NOTE we could have gone with a semaphore with +1 by controller and set_zero by worker,
    # but the python's Semaphore doesnt expose the counter value... So instead we have the
    # condition and a boolean which represents the "non-zero" here

    def __init__(self):
        self.work_available = False
        self.transitioned = Condition()

    def finish(self) -> None:
        """One or more work tasks have been finished. Blocks until more work available."""
        self.transitioned.acquire()
        try:
            if not self.work_available:
                self.transitioned.wait()
            else:
                self.work_available = False
        finally:
            self.transitioned.release()

    def notify(self) -> None:
        """Notify that more work should be done"""
        self.transitioned.acquire()
        try:
            if not self.work_available:
                self.work_available = True
                self.transitioned.notify()
            # if work was already available, we do nothing
        finally:
            self.transitioned.release()


class Client():
    def __init__(self, writer: Writer, urls: dict[str, str]) -> None:
        self.clients = {
            k: httpx.Client(base_url=v)
            for k, v in urls.items()
        }
        self.sync = {k: _Sync() for k in urls}
        self.tp = ThreadPoolExecutor(max_workers=len(urls))
        # NOTE we populate the envs here since we don't want to mix up thread pool usage
        self.envs = {
            k: Environment(**r.json())
            for k, r in self.tp.map(
                lambda tup: (tup[0], tup[1].get('/get_environment')),
                self.clients.items()
            )
        }
        self.writer = writer
        self.running = True
        self.futs = {k: self.tp.submit(self._perpetual_wait, k) for k in urls}

    def shutdown(self) -> None:
        self.running = False
        for host in self.clients:
            self.sync[host].notify()
        # NOTE be careful here, not waiting on futures results in hanging processes,
        # and closing the client before the futures complete causes the future to freeze
        self.tp.shutdown(wait=True, cancel_futures=True)
        for v in self.clients.values():
            v.close()

    def get_envs(self) -> dict[str, Environment]:
        return self.envs

    def submit(self, hostId: str, action: ActionSubmit) -> None:
        # self.sync[hostId].notify() # TODO enable this once the wait in server is async non-blocking
        rv = self.clients[hostId].put('/submit', json=action.model_dump()) 
        if rv.status_code != 200:
            raise ValueError(rv)

    def transmit_local(self, hostId: str, action: ActionDatasetTransmit) -> None:
        rv = self.clients[hostId].post('transmit_local', json=action.model_dump())
        if rv.status_code != 200:
            raise ValueError(rv)

    def transmit_remote(self, frHostId: str, frWorkerId: WorkerId, toHostId: str, toWorkerId: WorkerId, ds: list[DatasetId]) -> None:
        other_url = str(self.clients[toHostId].base_url)
        payload = TransmitPayload(other_url=other_url, other_worker=toWorkerId, this_worker=frWorkerId, datasets=ds, tracing_ctx_host=toHostId)
        rv = self.clients[frHostId].post('/transmit_remote', json=payload.model_dump())
        if rv.status_code != 200:
            raise ValueError(rv)

    def purge(self, hostId: str, action: ActionDatasetPurge) -> None:
        rv = self.clients[hostId].post('/purge', json=action.model_dump())
        if rv.status_code != 200:
            raise ValueError(rv)

    def fetch_as_url(self, host: str, worker: WorkerId, dataset_id: DatasetId) -> str:
        return self.clients[host].get(f'/fetch_as_url/{worker}/{dataset_id.task}/{dataset_id.output}').text

    def fetch_as_value(self, host: str, worker: WorkerId, dataset_id: DatasetId) -> Any:
        return self.clients[host].get(f'/fetch_as_url/{worker}/{dataset_id.task}/{dataset_id.output}').content

    def store_value(self, host: str, worker: WorkerId, dataset_id: DatasetId, data: bytes) -> None:
        rv = self.clients[host].put(f'/store_value/{worker}/{dataset_id.task}/{dataset_id.output}', content=data)
        if rv.status_code != 200:
            raise ValueError(rv)

    def _perpetual_wait(self, host: str) -> None:
        # TODO we dont propagate the timeout_secs here -- but then currently its always None so no harm done
        try:
            while True:
                logger.debug(f"about to finish for {host}")
                self.sync[host].finish()
                if not self.running:
                    break
                logger.debug(f"about to wait-some for {host}")
                # TODO we need the timeout to be sufficiently large, we dont have a recovery scenario yet
                result = self.clients[host].get(f"/wait_some", timeout=60.0)
                j = result.json()
                for e in j: #result.json():
                    event = Event(**e)
                    event = pyd_replace(event, at=f"{host}:{event.at}")
                    logger.debug(f"putting to queue: {event=}")
                    self.writer.put(event)
        except Exception as e:
            logger.error(f"perpetual wait failed with: {repr(e)}")
            raise

    def wait_some(self, timeout_secs: int|None) -> None:
        # check for exception, restart if needed
        for host in self.clients:
            logger.debug(f"checking future for {host=}")
            if self.futs[host].done():
                logger.error(f"unexpectedly finished future for {host} with exc {self.futs[host].exception()}")
                self._perpetual_wait(host)
        # notify that `wait_some` should be called again
        for host in self.clients:
            logger.debug(f"notify for {host=}")
            self.sync[host].notify()
