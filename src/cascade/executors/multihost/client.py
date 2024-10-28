"""
Implements the client for remote calling of `controller.executor` interface, exposed by
the `executor.multihost.server`. All methods are trivial except for the `wait_some` which
submits the requests into a thread pool and the results are gathered into a queue.
"""

from cascade.executors.multihost.event_queue import Writer
from concurrent.futures import Future, ThreadPoolExecutor

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
        if not self.work_available:
            self.transitioned.wait()
        else:
            self.work_available = False
            self.work_available_lock.release()

    def notify(self) -> None:
        """Notify that more work should be done"""
        self.transitioned.acquire()
        try:
            if not self.work_available:
                self.work_available = True
                self.transitioned.notify()
            # if work was already available, we do nothing
        finally:
            self.work_available_lock.release()


class Client():
    def __init__(self, writer: Writer, urls: dict[str, str]) -> None:
        self.clients = {
            k: httpx.Client(base_url=v)
            for k, v in urls.items()
        }
        self.sync = {k: _Sync() for k in self.urls.items()}
        self.tp = ThreadPoolExecutor(max_workers=len(urls))
        # NOTE we populate the envs here since we don't want to mix up thread pool usage
        self.envs = {
            k: Environment(**r.json())
            for k, r in self.tp.map(
                lambda k, v: (k, v.get('/get_environment')),
                self.clients.items()
            )
        }
        self.writer = writer
        self.futs = {k: self.tp.submit(self._perpetual_wait, k) for k in self.urls}

    def shutdown(self) -> None:
        for v in self.clients.values():
            v.close()
        self.tp.shutdown(wait=False, cancel_futures=True)

    def get_envs(self) -> dict[str, Environment]:
        return self.envs

    def submit(self, hostId: str, action: ActionSubmit) -> None:
        if not self.should_run[hostId]:
            self.should_run[hostId] = True
        self.clients[hostId].put('/submit', json=action.model_dump()) 

    def transmit(self, frHostId: str, frWorkerId: str, toHostId: str, toWorkerId: str, ds: set[DatasetId]) -> None:
        payload = TransmitPayload(other_url=self.urls[toHostId], other_worker=toWorkerId, this_worker=frWorkerId, datasets=ds)
        self.clients[frHostId].post('/transmit', json=payload.model_dump())

    def purge(self, hostId: str, action: ActionDatasetPurge) -> None:
        self.clients[hostId].post('/purge', json=action.model_dump())

    def _perpetual_wait(self, host: str) -> None:
        # TODO we dont propagate the timeout_secs here -- but then currently its always None so no harm done
        while True:
            self.sync[host].finish()
            result = self.clients[host].get(f"/wait_some")
            for e in result.json():
                self.writer.put(Event(**e))

    def wait_some(timeout_secs: int|None) -> None:
        # check for exception, restart if needed
        for host in self.clients
            if self.futs[host].done():
                logger.error(f"unexpectedly finished future for {host} with exc {self.futs[host].exception()}")
                self._perpetual_wait(host)
        # notify that `wait_some` should be called again
        for host in self.clients:
            self.sync[host].notify()
