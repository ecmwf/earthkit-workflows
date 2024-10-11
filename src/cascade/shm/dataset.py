"""
Abstraction of shared memory, keeps track of:
- shared memory id
- size
- state (in shm / on disk)
- lru metadata

Manages the to-disk-and-back persistence
"""

import hashlib
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory

import cascade.shm.algorithms as algorithms
import cascade.shm.disk as disk
from cascade.shm.func import assert_never

logger = logging.getLogger(__name__)


def get_capacity() -> int:
    r = subprocess.run(
        ["findmnt", "-b", "-o", "AVAIL", "/dev/shm"], check=True, capture_output=True
    )
    avail = r.stdout.decode("ascii").split("\n", 1)[1].strip()
    return int(avail)


class DatasetStatus(int, Enum):
    created = auto()
    in_memory = auto()
    paging_out = auto()
    on_disk = auto()
    paged_in = auto()


@dataclass
class Dataset:
    shmid: str
    size: int
    status: DatasetStatus

    created: int
    ongoing_reads: set[str]
    retrieved_first: int
    retrieved_last: int


class Manager:
    """Keeps track of what is registered, how large it is, and under what name.

    Notes about thread safety: this class is _generally_ single-threaded, except for
    the `callback` functions present in the pageout/pagein functions, which will be
    called from within shm.Disk's threadpool.

    1/ For pagein, the situation is simple -- _before_ launching the thread, we set
    the dataset status to `paged_in`, which prevents another thread from being launched.
    Only after the thread in its callback puts the status back to `in_memory` as its
    last action can another thread be launched -- so we are thread-safe.

    2/ For pageout, we employ two locks, one for the initiation of the global pageout,
    one for decrement of counter. The first lock is locked before any thread is launched,
    and unlocked as the last thread finishes. As before, before threads are launched,
    in the main thread we set dataset status to `paging_out`, preventing interference
    with other main thread operation that could come later. The second lock prevents
    the paging out threads interfering with each other.

    3/ All threads are called with try-catch, and no matter what the locks would attempt
    a release at the end, so barring deaths of thread pools we should not deadlock
    ourselves.
    """

    def __init__(self, capacity: int | None = None) -> None:
        # key is as understood by the external apps
        self.datasets: dict[str, Dataset] = {}
        if not capacity:
            capacity = get_capacity()
        self.capacity = capacity
        self.free_space = capacity
        self.pageout_all = threading.Lock()
        self.pageout_one = threading.Lock()
        self.pageout_count = 0
        self.disk = disk.Disk()

    def add(self, key: str, size: int) -> tuple[str, str]:
        # TODO round the size up to page multiple?
        if size > self.capacity:
            return "", "capacity exceeded"

        if size > self.free_space:
            self.page_out_at_least(size - self.free_space)
            return "", "wait"
        self.free_space -= size

        h = hashlib.new("md5", usedforsecurity=False)
        h.update((key).encode())
        shmid = h.hexdigest()[:24]

        self.datasets[key] = Dataset(
            shmid=shmid,
            size=size,
            status=DatasetStatus.created,
            created=time.time_ns(),
            retrieved_first=0,
            retrieved_last=0,
            ongoing_reads=set(),
        )
        return shmid, ""

    def close_callback(self, key: str, rdid: str) -> None:
        if not rdid:
            if self.datasets[key].status != DatasetStatus.created:
                raise ValueError(
                    f"invalid transition from {self.datasets[key].status} for {key} and {rdid}"
                )
            self.datasets[key].status = DatasetStatus.in_memory
        else:
            if self.datasets[key].status != DatasetStatus.in_memory:
                raise ValueError(
                    f"invalid transition from {self.datasets[key].status} for {key} and {rdid}"
                )
            if rdid not in self.datasets[key].ongoing_reads:
                logger.warning(
                    f"unexpected/redundant remove of ongoing reader: {key}, {rdid}"
                )
            else:
                self.datasets[key].ongoing_reads.remove(rdid)

    def page_out(self, key: str) -> None:
        ds = self.datasets[key]
        if ds.status != DatasetStatus.in_memory and not bool(ds.ongoing_reads):
            raise ValueError(f"invalid page out on {ds}")
        ds.status = DatasetStatus.paging_out

        def callback(ok: bool) -> None:
            if ok:
                ds.status = DatasetStatus.on_disk
                logger.debug(f"pageout of {ds} finished")
                with self.pageout_one:
                    self.free_space += ds.size
                    self.pageout_count -= 1
                    if self.pageout_count == 0:
                        self.pageout_all.release()
            else:
                logger.error(f"pageout of {ds} failed, marking bad")
                self.purge(key)
                with self.pageout_one:
                    self.pageout_count -= 1
                    if self.pageout_count == 0:
                        self.pageout_all.release()

        self.disk.page_out(ds.shmid, callback)

    def page_out_at_least(self, amount: int) -> None:
        if not self.pageout_all.acquire(blocking=False):
            return
        candidates = (
            algorithms.Entity(
                key, ds.created, ds.retrieved_first, ds.retrieved_last, ds.size
            )
            for key, ds in self.datasets.items()
            if ds.status == DatasetStatus.in_memory
        )
        winners = algorithms.lottery(candidates, amount)
        self.pageout_count = len(winners)
        for winner in winners:
            self.page_out(winner)

    def page_in(self, key: str) -> None:
        ds = self.datasets[key]
        if ds.status != DatasetStatus.on_disk:
            raise ValueError(f"invalid restore on {ds}")
        ds.status = DatasetStatus.paged_in
        if self.free_space < ds.size:
            raise ValueError("insufficient space")
        self.free_space -= ds.size

        def callback(ok: bool):
            if ok:
                ds.status = DatasetStatus.in_memory
            else:
                logger.error(f"pagein of {ds} failed, marking bad")
                self.purge(key)

        self.disk.page_in(ds.shmid, ds.size, callback)

    def get(self, key: str) -> tuple[str, int, str, str]:
        ds = self.datasets[key]
        if ds.status in (
            DatasetStatus.created,
            DatasetStatus.paged_in,
            DatasetStatus.paging_out,
        ):
            return "", 0, "", "wait"
        if ds.status == DatasetStatus.on_disk:
            if ds.size > self.free_space:
                self.page_out_at_least(ds.size - self.free_space)
                return "", 0, "", "wait"
            self.page_in(key)
            return "", 0, "", "wait"
        if ds.status != DatasetStatus.in_memory:
            assert_never(ds.status)
        while True:
            rdid = str(uuid.uuid4())[:8]
            if rdid not in ds.ongoing_reads:
                break
        ds.ongoing_reads.add(rdid)
        retrieved = time.time_ns()
        if ds.retrieved_first == 0:
            ds.retrieved_first = retrieved
        ds.retrieved_last = retrieved
        return ds.shmid, ds.size, rdid, ""

    def purge(self, key: str) -> None:
        try:
            ds = self.datasets.pop(key)
            if ds.status in (
                DatasetStatus.created,
                DatasetStatus.paging_out,
                DatasetStatus.paged_in,
            ):
                logger.warning(f"calling purge in unsafe status: {key}, {ds.status}")
            elif ds.status == DatasetStatus.on_disk:
                logger.warning(f"skipping purge because is on disk: {key}, {ds.status}")
                return
            elif ds.status != DatasetStatus.in_memory:
                assert_never(ds.status)
            shm = SharedMemory(ds.shmid, create=False)
            shm.unlink()
            shm.close()
            with self.pageout_one:
                self.free_space += ds.size
        except Exception:
            logger.exception(
                "failed to purge {key}, free space may be incorrect, /dev/shm may have leaked"
            )

    def atexit(self) -> None:
        keys = list(self.datasets.keys())
        for key in keys:
            self.purge(key)
        self.disk.atexit()
