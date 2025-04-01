# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles disk operations"""

import logging
import multiprocessing.resource_tracker
import tempfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Callable

logger = logging.getLogger(__name__)


class Disk:
    def __init__(self) -> None:
        self.root = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.readers = ThreadPoolExecutor(max_workers=4)
        self.writers = ThreadPoolExecutor(max_workers=4)

    def _page_in(self, shmid: str, size: int, callback: Callable[[bool], None]) -> None:
        try:
            chunk_size = 4096
            shm = SharedMemory(shmid, create=True, size=size)
            with open(f"{self.root.name}/{shmid}", "rb") as f:
                i = 0
                while True:
                    b = f.read(chunk_size)
                    l = len(b)
                    if not l:
                        break
                    shm.buf[i : i + l] = b
                    i += l
            shm.close()
            # TODO eleminate in favour of track=False, once we are on python 3.13+
            multiprocessing.resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore # _name
        except Exception:
            logger.exception("failure on page in")
            callback(False)
        else:
            callback(True)

    def page_in(self, shmid: str, size: int, callback: Callable) -> None:
        self.readers.submit(self._page_in, shmid, size, callback)

    def _page_out(self, shmid: str, callback: Callable[[bool], None]) -> None:
        try:
            shm = SharedMemory(shmid, create=False)
            with open(f"{self.root.name}/{shmid}", "wb") as f:
                f.write(shm.buf[:])
            shm.unlink()
            shm.close()
        except Exception:
            logger.exception("failure on page out")
            callback(False)
        else:
            callback(True)

    def page_out(self, shmid: str, callback: Callable) -> None:
        self.writers.submit(self._page_out, shmid, callback)

    def atexit(self) -> None:
        self.root.cleanup()
        self.readers.shutdown(wait=False, cancel_futures=True)
        self.writers.shutdown(wait=False, cancel_futures=True)
