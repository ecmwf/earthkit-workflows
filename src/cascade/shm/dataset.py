# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstraction of shared memory, keeps track of:
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
    try:
        r = subprocess.run(
            ["findmnt", "-b", "-o", "AVAIL", "/dev/shm"],
            check=True,
            capture_output=True,
        )
        # be careful -- hpc clusters have typically more rich output of findmnt
        avail = r.stdout.decode("ascii").split("\n")[1].strip()
        return int(avail)
    except FileNotFoundError:
        return 128 * (
            1024**3
        )  # gets likely trimmed later... this is for macos which doest have findmnt


class DatasetStatus(int, Enum):
    created = auto()
    in_memory = auto()
    paging_out = auto()
    on_disk = auto()
    paged_in = auto()


FIFTEEN_MINUTES = int(15 * 60 * 1e9)
STALE_CREATE = FIFTEEN_MINUTES
STALE_READ = FIFTEEN_MINUTES


@dataclass
class Dataset:
    shmid: str
    size: int
    status: DatasetStatus

    created: int
    ongoing_reads: dict[str, int]
    retrieved_first: int
    retrieved_last: int
    deser_fun: str
    delayed_purge: bool = False

    def is_pageoutable(self, ref_time: int) -> bool:
        created_stale = (
            self.status == DatasetStatus.created
            and ref_time - self.created > STALE_CREATE
        )
        no_fresh_read = (
            not (self.ongoing_reads)
            or ref_time - max(self.ongoing_reads.values()) > STALE_READ
        )
        return created_stale or (
            self.status == DatasetStatus.in_memory and no_fresh_read
        )


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

    4/ the at-exit based invocation of `purge` is more robust, assuming that some of
    the datasets have been purged half-way, and doing its best to clean up.
    """

    def __init__(self, prefix: str, capacity: int | None = None) -> None:
        # key is as understood by the external apps
        self.datasets: dict[str, Dataset] = {}
        default_capacity = get_capacity()
        if not capacity:
            capacity = default_capacity
        elif capacity > default_capacity:
            # TODO introduce capacity setter api which includes this check
            logger.warning(
                f"configured with more capacity than available, trimming to {default_capacity}"
            )
            capacity = default_capacity
        self.capacity = capacity
        logger.info(f"dataset started with actual capacity {self.capacity}")
        self.free_space = capacity
        self.pageout_all = threading.Lock()
        self.pageout_one = threading.Lock()
        self.pageout_count = 0
        self.disk = disk.Disk()
        self.prefix = prefix

    def add(self, key: str, size: int, deser_fun: str) -> tuple[str, str]:
        if key in self.datasets:
            return "", "conflict"

        # TODO round the size up to page multiple?
        if size > self.capacity:
            return "", "capacity exceeded"

        if size > self.free_space:
            self.page_out_at_least(size - self.free_space)
            return "", "wait"
        self.free_space -= size

        h = hashlib.new("md5", usedforsecurity=False)
        h.update((key).encode())
        shmid = self.prefix + h.hexdigest()[: (24 - len(self.prefix))]

        self.datasets[key] = Dataset(
            shmid=shmid,
            size=size,
            status=DatasetStatus.created,
            created=time.time_ns(),
            retrieved_first=0,
            retrieved_last=0,
            ongoing_reads={},
            deser_fun=deser_fun,
        )
        return shmid, ""

    def close_callback(self, key: str, rdid: str) -> None:
        if not rdid:
            if self.datasets[key].status != DatasetStatus.created:
                raise ValueError(
                    f"invalid transition from {self.datasets[key].status} for {key} and {rdid}"
                )
            logger.debug(f"create callback finished -> {key} is in-memory")
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
                self.datasets[key].ongoing_reads.pop(rdid)
        if self.datasets[key].delayed_purge and not self.datasets[key].ongoing_reads:
            self.purge(key, False)

    def page_out(self, key: str) -> None:
        ds = self.datasets[key]
        if ds.status != DatasetStatus.in_memory and not bool(ds.ongoing_reads):
            logger.warning(f"risky page out on {ds}, assuming staledness")
        ds.status = DatasetStatus.paging_out

        def callback(ok: bool) -> None:
            if ok:
                ds.status = DatasetStatus.on_disk
                logger.debug(f"pageout of {key} -> {ds} finished")
                with self.pageout_one:
                    self.free_space += ds.size
                    self.pageout_count -= 1
                    if self.pageout_count == 0:
                        self.pageout_all.release()
            else:
                logger.error(f"pageout of {key} -> {ds} failed, marking bad")
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
            if ds.is_pageoutable(time.time_ns())
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
                logger.debug(f"dataset {key} now considered in-memory")
                ds.status = DatasetStatus.in_memory
            else:
                logger.error(f"pagein of {ds} failed, marking bad")
                self.purge(key)

        self.disk.page_in(ds.shmid, ds.size, callback)

    def get(self, key: str) -> tuple[str, int, str, str, str]:
        ds = self.datasets[key]
        if ds.status in (
            DatasetStatus.created,
            DatasetStatus.paged_in,
            DatasetStatus.paging_out,
        ):
            logger.debug(f"returing wait on {key} because of {ds.status=}")
            return "", 0, "", "", "wait"
        if ds.status == DatasetStatus.on_disk:
            if ds.size > self.free_space:
                self.page_out_at_least(ds.size - self.free_space)
                logger.debug(f"returing wait on {key} because of page out issued first")
                return "", 0, "", "", "wait"
            self.page_in(key)
            logger.debug(f"returing wait on {key} because of page in issued")
            return "", 0, "", "", "wait"
        if ds.status != DatasetStatus.in_memory:
            assert_never(ds.status)
        while True:
            rdid = str(uuid.uuid4())[:8]
            if rdid not in ds.ongoing_reads:
                break
        retrieved = time.time_ns()
        ds.ongoing_reads[rdid] = retrieved
        if ds.retrieved_first == 0:
            ds.retrieved_first = retrieved
        ds.retrieved_last = retrieved
        return ds.shmid, ds.size, rdid, ds.deser_fun, ""

    def purge(self, key: str, is_exit: bool = False) -> None:
        # TODO the is_exit invocation is causing mess in logs, consider removal
        try:
            logger.debug(f"attempting purge-inquire of {key}")
            try:
                ds = self.datasets[key]
                if ds.ongoing_reads and not is_exit:
                    # logging as debug because this is a common and legit scenario due to scheduler speed
                    logger.debug(
                        f"premature purge of {key} while there are still reads, delaying"
                    )
                    ds.delayed_purge = True
                    return
            except Exception:
                if not is_exit:  # if this happens during exit, its acceptable race
                    raise
                else:
                    return
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
            logger.debug(f"attempting purge-unlink of {key} with {ds.shmid}")
            shm.unlink()
            shm.close()
            if not is_exit:  # we dont want to lock at exit, we may hang out unhealthily
                with self.pageout_one:
                    self.free_space += ds.size
            self.datasets.pop(key)
        except Exception:
            logger.exception(
                f"failed to purge {key}, free space may be incorrect, /dev/shm may have leaked"
            )

    def atexit(self) -> None:
        keys = list(self.datasets.keys())
        for key in keys:
            self.purge(key, is_exit=True)
        self.disk.atexit()
