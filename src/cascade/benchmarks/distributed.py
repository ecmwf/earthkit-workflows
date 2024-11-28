"""
Utility function for actual distributed runs on a cluster
"""

# TODO rework, simplify

from pydantic import BaseModel, Field
from forecastbox.utils import logging_config as fiab_logging
import cascade.executors.simulator
from cascade.scheduler.impl import naive_bfs_layers
import time
from cascade.controller.core import State
from cascade.controller.impl import run
from cascade.low.core import JobInstance, DatasetId
from cascade.low.views import sinks
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
from cascade.controller.tracing import trace, Microtrace
from typing_extensions import Self
from typing import Any
import os
from cascade.low.into import graph2job
from cascade.graph import Graph

def init():
    os.environ["CLOUDPICKLE"] = "yes" # for fiab desers
    logging.config.dictConfig(fiab_logging)

class ZmqWorkerHostSpec(BaseModel):
    workers: int
    zmq_port: int
    shm_port: int
    shm_vol_gb: int

class ZmqControllerSpec(BaseModel):
    job: JobInstance
    url: str
    outputs: set[DatasetId] = Field(default_factory=set)

class ZmqClusterSpec(BaseModel):
    controller: ZmqControllerSpec
    worker_hosts: list[ZmqWorkerHostSpec]

    @classmethod
    def local(cls, graph: Graph, worker_hosts: int = 1, worker_per_host: int = 4, shm_vol_gb: int = 4) -> Self:
        port_base = 12345
        job = graph2job(graph)
        outputs = sinks(job)
        controller = ZmqControllerSpec(job=job, url=f"tcp://localhost:{port_base}", outputs=outputs)
        worker_hosts_specs = [
            ZmqWorkerHostSpec(workers=worker_per_host, zmq_port=port_base + 1 + i*2, shm_port = port_base + 2 + i*2, shm_vol_gb = shm_vol_gb)
            for i in range(worker_hosts)
        ]
        return cls(controller = controller, worker_hosts = worker_hosts_specs)
    
def launch_zmq_worker(spec: ZmqWorkerHostSpec, host_id: int, controller: ZmqControllerSpec):
    self_url = f"tcp://{socket.gethostname()}:{spec.zmq_port}"
    shm_port = spec.shm_port
    try:
        init()
        shm_api.publish_client_port(shm_port)
        shm_pref = f"sCasc{host_id}_"
        shm = Process(target=shm_server.entrypoint, args=(shm_port, spec.shm_vol_gb * 1024**3, fiab_logging, shm_pref))
        shm.start()
        shm_client.ensure()
        host_memory = 1024 # NOTE used *only* for planning, and actually not used rn
        executor = SingleHostExecutor(ExecutorConfig(spec.workers, host_memory, f"h{host_id}"), controller.job)
        backbone = ZmqBackbone(self_url, controller_url=controller.url, environment=executor.get_environment(), host_id=f"h{host_id}")
        adapter = BackboneLocalExecutor(executor, backbone)
        adapter.recv_loop()
    finally:
        shm_client.shutdown()
        shm.terminate()

def launch_zmq_controller(hosts: int, spec: ZmqControllerSpec) -> State:
    init()
    backbone = ZmqBackbone(spec.url, expected_workers=hosts)
    executor = BackboneExecutor(backbone)
    exe_rec = cascade.executors.simulator.placeholder_execution_record(spec.job)
    schedule = naive_bfs_layers(spec.job, exe_rec, set()).get_or_raise()
    start_fine = time.perf_counter_ns()
    end_state = run(spec.job, executor, schedule, spec.outputs)
    end_fine = time.perf_counter_ns()
    trace(Microtrace.total_incluster, end_fine - start_fine)
    print(f"in-cluster time: {(end_fine - start_fine) / 1e9: .3f}")

    return end_state

def launch_from_specs(spec: ZmqClusterSpec, idx: int|None) -> Any:
    """
    - Idx is None => launch whole cluster locally,
    - Idx is >= 0 => launch ith worker,
    - Idx is -1 => launch controller
    """
    
    run = lambda : launch_zmq_controller(len(spec.worker_hosts), spec.controller)
    if idx is None:
        ps = []
        for i, worker in enumerate(spec.worker_hosts):
            p = Process(target=launch_zmq_worker, args=(worker, i, spec.controller))
            ps.append(p)
            p.start()
        rv = run()
        for p in ps:
            p.join()
        return rv
    else:
        if idx >= 0:
            return launch_zmq_worker(spec.worker_hosts[idx], idx, spec.controller)
        else:
            return run()
