"""
Handles disk operations
"""

import multiprocessing.resource_tracker
import tempfile
from multiprocessing.shared_memory import SharedMemory


class Disk:
    def __init__(self) -> None:
        self.root = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def page_in(self, shmid: str, size: int) -> None:
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

    def page_out(self, shmid: str) -> None:
        shm = SharedMemory(shmid, create=False)
        with open(f"{self.root.name}/{shmid}", "wb") as f:
            f.write(shm.buf[:])
        shm.unlink()
        shm.close()

    def atexit(self) -> None:
        self.root.cleanup()
