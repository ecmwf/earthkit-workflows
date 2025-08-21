"""Benchmark shm in isolation (without cascade) by just invoking a lot of reads and writes in a limited memory setting"""

import multiprocessing as mp
import random
import sys
from time import perf_counter_ns

import numpy as np

import cascade.shm.client as c
from cascade.executor.config import logging_config_filehandler
from cascade.shm.api import publish_socket_addr
from cascade.shm.server import entrypoint

# addr = 12345 + random.randint(0, 1000) # due to freeing of port taking time
addr = "/tmp/comm"
publish_socket_addr(addr)


class Context:
    shm_process: mp.process.BaseProcess  # cls var
    steps: list[tuple[(str, int)]] = []

    @classmethod
    def track(cls, label: str):
        cls.steps.append((label, perf_counter_ns()))

    @classmethod
    def report(cls):
        rpt = (
            lambda t1, t2: f"from {t1[0]} to {t2[0]} took {(t2[1]-t1[1]) / 1e6:.3f} ms"
        )
        print(rpt(cls.steps[0], cls.steps[-1]))
        for i in range(len(cls.steps) - 1):
            print(rpt(cls.steps[i], cls.steps[i + 1]))


def start_shm(capacity_mb: int):
    capacity = capacity_mb * 1024**2
    logging_config = logging_config_filehandler("/tmp/shmlog.txt")
    Context.shm_process = mp.get_context("fork").Process(
        target=entrypoint,
        kwargs={"capacity": capacity, "logging_config": logging_config},
    )
    Context.shm_process.start()


def shutdown():
    p = Context.shm_process
    if p is None:
        return
    if p.is_alive():
        p.terminate()
        p.join(1)
    if p.is_alive():
        p.kill()
        p.join(1)
    if p.is_alive():
        print("SHM process failed to terminate in time!")


def scenario1():
    """Start 128MB shm, allocate 8 * 32MB (so half should get paged out), then access them in sequence
    This should incurr 4 pageouts on write, and then 4+ pageins and 4 pageouts on read.
    """
    start_shm(128)
    c.ensure()

    L = 32 * 1024**2
    Z = np.zeros(L, dtype=np.int8).tobytes()
    Context.track("start")
    for i in range(8):
        buf = c.allocate(f"k{i}", L, "whatever")
        buf.view()[:L] = Z
        buf.close()
        Context.track(f"alloc{i}")
    for i in range(8):
        buf = c.get(f"k{i}")
        buf.view()
        buf.close()
        Context.track(f"get{i}")
    Context.track("end")
    Context.report()


def scenario2():
    """Start 1024MB shm, allocate 128 * 8MB (so that all fits), then 128 ** 2 gets to simulate high load"""
    start_shm(1024)
    c.ensure()

    L = 8 * 1024**2
    Z = np.zeros(L, dtype=np.int8).tobytes()
    Context.track("start")
    for i in range(128):
        buf = c.allocate(f"k{i}", L, "whatever")
        buf.view()[:L] = Z
        buf.close()
    Context.track("allocs")
    for _ in range(128 * 4):
        buf = c.get(f"k{random.randint(0, 127)}")
        buf.view()
        buf.close()
    Context.track("end")
    Context.report()


if __name__ == "__main__":
    func = globals().get(sys.argv[1], None)
    if not func:
        raise NotImplementedError(sys.argv[1])
    try:
        func()
    finally:
        shutdown()
