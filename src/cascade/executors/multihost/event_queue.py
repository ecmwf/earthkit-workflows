"""
A thread-safe queue for gathering `controller.core.Event`s from remote workers into the Router Proxy,
to facilitate async/less-blocking behaviour. Does not support multiprocessing.

Two ends to the queue: the writer gets passed in the multihost.client to the thread pool for the Futures
to write the results to, the reader is in the main process/thread of the Router Proxy.
"""

import time
from queue import Queue, Empty
from cascade.controller.core import Event
from typing import cast
import logging

logger = logging.getLogger(__name__)

class Writer():
    def __init__(self, q: Queue) -> None:
        self.q = q

    def put(self, e: Event) -> None:
        """This is non-blocking and the queue is unbounded -- we don't expect we'll be submitting huge batches,
        thus the queue size is limited by external constraints anyway."""
        self.q.put(e)

class Reader():
    def __init__(self, q: Queue) -> None:
        self.q = q

    def _get_one(self, block: bool, timeout_secs: int|None) -> Event:
        rv = self.q.get(block, timeout_secs)
        # NOTE the task_done is currently extraneous but it doesn't hurt to be safe
        self.q.task_done()
        return cast(Event, rv)

    def get(self, timeout_secs: int|None) -> list[Event]:
        """Blocks until the first item available, and then it gathers all other elements obtainable without blocking.
        This is so that if there are already multiple events queued, we want to get them all on one call but don't
        block any further. If timeout elapses before any item is obtained, this returns empty list."""
        results: list[Event] = []
        try:
            results.append(self._get_one(True, timeout_secs))
            while True:
                results.append(self._get_one(False, None))
        except Empty:
            pass
        return results

def build_queue() -> tuple[Writer, Reader]:
    q: Queue = Queue()
    return Writer(q), Reader(q)
