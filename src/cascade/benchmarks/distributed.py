"""
Utility function for actual distributed runs on a cluster
"""

# TODO rework, simplify

from forecastbox.utils import logging_config as fiab_logging
import cascade.executors.simulator
from cascade.scheduler.impl import naive_bfs_layers
import time
from cascade.controller.impl import run
import cascade.low.core
from multiprocessing import Process
from cascade.executors.backbone.executor import BackboneExecutor
import cascade.shm.server as shm_server
import logging
from forecastbox.executor.executor import SingleHostExecutor, Config as ExecutorConfig
from cascade.executors.backbone.zmq import ZmqBackbone
from cascade.executors.backbone.local import BackboneLocalExecutor
import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
import socket

def launch_zmq_worker(workers: int, controller_url: str, host_id: int, job: cascade.low.core.JobInstance):
    self_url = f"tcp://{socket.gethostname()}:6001"
    shm_port = 6000
    try:
        logging.config.dictConfig(fiab_logging)
        gb4 = 4 * (1024**3)
        shm_api.publish_client_port(shm_port)
        shm_pref = f"sCasc{host_id}_"
        shm = Process(target=shm_server.entrypoint, args=(shm_port, gb4, fiab_logging, shm_pref))
        shm.start()
        shm_client.ensure()
        executor = SingleHostExecutor(ExecutorConfig(workers, 1024, f"h{host_id}"), job)
        backbone = ZmqBackbone(self_url, controller_url=controller_url, environment=executor.get_environment(), host_id=f"h{host_id}")
        adapter = BackboneLocalExecutor(executor, backbone)
        adapter.recv_loop()
    finally:
        shm_client.shutdown()
        shm.terminate()

def launch_zmq_controller(hosts: int, controller_url: str, job: cascade.low.core.JobInstance) -> None:
    logging.config.dictConfig(fiab_logging)
    backbone = ZmqBackbone(controller_url, expected_workers=hosts)
    executor = BackboneExecutor(backbone)
    exe_rec = cascade.executors.simulator.placeholder_execution_record(job)
    schedule = naive_bfs_layers(job, exe_rec, set()).get_or_raise()
    start_fine = time.perf_counter_ns()
    run(job, executor, schedule)
    end_fine = time.perf_counter_ns()
    print(f"in-cluster time: {(end_fine - start_fine) / 1e9: .3f}")
