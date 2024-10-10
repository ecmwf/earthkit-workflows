"""
Abstraction of shared memory, keeps track of:
- shared memory id
- size
- state (in shm / on disk)
- lru metadata

Manages the to-disk-and-back persistence
"""

# TODO:
# - add threadpool for the disk operations
#   - add locks for the dataset status changes?

import hashlib
import logging
import subprocess
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
    """Keeps track of what is registered, how large it is, and under what name"""

    def __init__(self, capacity: int | None = None) -> None:
        # key is as understood by the external apps
        self.datasets: dict[str, Dataset] = {}
        if not capacity:
            capacity = get_capacity()
        self.capacity = capacity
        self.free_space = capacity
        self.purging = False
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

    def page_out(self, ds: Dataset) -> None:
        if ds.status != DatasetStatus.in_memory and not bool(ds.ongoing_reads):
            raise ValueError(f"invalid page out on {ds}")
        ds.status = DatasetStatus.paging_out
        self.disk.page_out(ds.shmid)
        # TODO have the restore return early, the status update as callback
        ds.status = DatasetStatus.on_disk
        self.free_space += ds.size
        logger.debug(f"paged out {ds}, free space is now {self.free_space}")

    def page_out_at_least(self, amount: int) -> None:
        if self.purging:
            return
        self.purging = True
        candidates = (
            algorithms.Entity(
                key, ds.created, ds.retrieved_first, ds.retrieved_last, ds.size
            )
            for key, ds in self.datasets.items()
            if ds.status == DatasetStatus.in_memory
        )
        for winner in algorithms.lottery(candidates, amount):
            logger.debug(f"purging lottery {winner = }")
            self.page_out(self.datasets[winner])
        # TODO replace the purging with some barrier; check for it at the beginning and instead wait
        self.purging = False

    def page_in(self, ds: Dataset) -> None:
        if ds.status != DatasetStatus.on_disk:
            raise ValueError(f"invalid restore on {ds}")
        ds.status = DatasetStatus.paged_in
        if self.free_space < ds.size:
            raise ValueError("insufficient space")
        self.free_space -= ds.size
        self.disk.page_in(ds.shmid, ds.size)
        # TODO have the restore return early, the status update as callback, return the wait
        ds.status = DatasetStatus.in_memory
        # return "", 0, "", "wait"

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
            self.page_in(ds)
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
        self.free_space += ds.size

    def atexit(self) -> None:
        keys = list(self.datasets.keys())
        for key in keys:
            self.purge(key)
        self.disk.atexit()
