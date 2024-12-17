"""
Tests building a small backbone cluster and checking the environment validity
"""

import logging
import os
import signal
from functools import reduce
from threading import Thread
from cascade.low.core import Environment, Worker
from cascade.executors.backbone.zmq import ZmqBackbone

logger = logging.getLogger(__name__)
port_start = 5555
controller_url = f"tcp://localhost:{port_start}"

def worker_entrypoint(idx: int, results: dict) -> None:
    self_url = f"tcp://localhost:{port_start+idx}"
    environment = Environment(workers={
        f"h{idx}:w0": Worker(cpu=1, gpu=0, memory_mb=1024),
        f"h{idx}:w1": Worker(cpu=1, gpu=0, memory_mb=1024),
    })
    backbone = ZmqBackbone(self_url, controller_url=controller_url, environment=environment, host_id=f"h{idx}")
    results[idx] = environment
    
def controller_entrypoint(expected_workers: int, results: dict) -> None:
    backbone = ZmqBackbone(controller_url, expected_workers=expected_workers)
    results[0] = backbone.get_environment()
    
def test_zmq_build():
    try:
        results = {}
        tC = Thread(target=controller_entrypoint, args=(2, results))
        tC.start()
        t1 = Thread(target=worker_entrypoint, args=(1, results))
        t1.start()
        t2 = Thread(target=worker_entrypoint, args=(2, results))
        t2.start()
        t1.join(1.0)
        t2.join(1.0)
        tC.join(1.0)
        assert not t1.is_alive()
        assert not t2.is_alive()
        assert not tC.is_alive()

        workers = reduce(lambda acc, e: acc.union(e.workers.keys()), [results[1], results[2]], set())
        assert set(results[0].workers.keys()) == workers
        colocations = {frozenset(cluster) for cluster in results[0].colocations}
        assert colocations == {frozenset(["h1:w0", "h1:w1"]), frozenset(["h2:w0", "h2:w1"])}
    except AssertionError:
        raise
    except Exception:
        logger.exception("test failure, sigint threads")
        os.kill(os.getpid(), signal.SIGINT)
        raise
