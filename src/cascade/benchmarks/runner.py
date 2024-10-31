"""
Runs actual job: controller + scheduler + executor
"""

import os
import signal
import httpx
import logging
from cascade import Cascade
import cascade.low.into
from dask.distributed import LocalCluster
from cascade.executors.dask_delayed import job2delayed
from cascade.executors.dask_futures import DaskFuturisticExecutor
from cascade.controller.impl import run
import cascade.scheduler
from cascade.scheduler.impl import naive_bfs_layers
import cascade.low.core
import cascade.executors.simulator
import time
from dask.threaded import get
from forecastbox.executor.executor import SingleHostExecutor, Config as ExecutorConfig
from multiprocessing import Process
import cascade.shm.server as shm_server
import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from forecastbox.utils import logging_config as fiab_logging
import cascade.benchmarks.api as api
from cascade.low.func import assert_never
import uvicorn
from cascade.executors.multihost.impl import RouterExecutor
from cascade.executors.multihost.worker_server import build_app

def wait_for(client: httpx.Client, root_url: str) -> None:
    """Calls /status endpoint, retry on ConnectError"""
    i = 0
    while i < 3:
        try:
            rc = client.get(f"{root_url}/status")
            if not rc.status_code == 200:
                raise ValueError(f"failed to start {root_url}: {rc}")
            return
        except httpx.ConnectError:
            i += 1
            time.sleep(2)
    raise ValueError(f"failed to start {root_url}: no more retries")

def launch_fiab_host(workers: int, host_id: int, port: int, job: cascade.low.core.JobInstance) -> None:
    try:
        logging.config.dictConfig(fiab_logging)
        gb4 = 4 * (1024**3)
        shm_api.publish_client_port(port + 1000)
        shm_pref = f"s{host_id}_"
        shm = Process(target=shm_server.entrypoint, args=(port + 1000, gb4, fiab_logging, shm_pref))
        shm.start()
        shm_client.ensure()
        executor = SingleHostExecutor(ExecutorConfig(workers, 1024), job, {"host": f"h{host_id}"})
        app = build_app(executor)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level=None, log_config=None)
    finally:
        print(f"shutting down shm {os.getpid()}")
        shm_client.shutdown()
        shm.terminate()
        print(f"shutdown done {os.getpid()}")

def run_job_on(graph: cascade.graph.Graph, opts: api.Options):
    job = cascade.low.into.graph2job(graph)
    exe_rec = cascade.executors.simulator.placeholder_execution_record(job)
    schedule = naive_bfs_layers(job, exe_rec, set()).get_or_raise()
    shm: Process|None = None
    start_raw = time.perf_counter_ns()
    if isinstance(opts, api.DaskDelayed):
        Cascade(graph).execute()
    elif isinstance(opts, api.Fiab):
        gb4 = 4 * (1024**3)
        port = 12345
        shm_api.publish_client_port(port)
        shm = Process(target=shm_server.entrypoint, args=(port, gb4, fiab_logging))
        shm.start()
        fiab_executor = SingleHostExecutor(ExecutorConfig(opts.workers, 1024), job, {"host": "fiab"})
        shm_client.ensure()
        try:
            start_fine = time.perf_counter_ns()
            run(job, fiab_executor, schedule)
            end_fine = time.perf_counter_ns()
            logging.info("controller in fiab done")
            fiab_executor.procwatch.join()
        except Exception as e:
            shm_client.shutdown()
            raise
    elif isinstance(opts, api.DaskThreaded):
        dly = job2delayed(job)
        cnts = {k: sum(1 for v in dly.values() if k in v[2]) for k in dly}
        sinks = [k for k in cnts if cnts[k] == 0]
        get(dly, sinks)
    elif isinstance(opts, api.DaskFutures):
        env = cascade.low.core.Environment(workers={
            f'{i}': cascade.low.core.Worker(cpu=1, gpu=0, memory_mb=1024)
            for i in range(opts.workers)
        })
        with LocalCluster(n_workers=opts.workers, processes=True, dashboard_address=":0") as clu:
            dask_executor = DaskFuturisticExecutor(clu, job, env)
            start_fine = time.perf_counter_ns()
            run(job, dask_executor, schedule)
            end_fine = time.perf_counter_ns()
    elif isinstance(opts, api.MultiHost):
        start = 8000
        urls = {f"h{i}": f"http://localhost:{start+i}" for i in range(opts.hosts)}
        ps = [Process(target=launch_fiab_host, args=(opts.workers_per_host, i, start+i, job)) for i in range(opts.hosts)]
        for p in ps:
            p.start()
        try:
            client = httpx.Client()
            for u in urls.values():
                wait_for(client, u)
            router_executor = RouterExecutor(urls)
            start_fine = time.perf_counter_ns()
            run(job, router_executor, schedule)
            end_fine = time.perf_counter_ns()
        finally:
            router_executor.shutdown()
            for p in ps:
                print(f"interrupt {p.pid}", flush=True)
                os.kill(p.pid, signal.SIGINT)
            for p in ps:
                print(f"join {p.pid}", flush=True)
                p.join(1)
            for p in ps:
                print(f"inquire {p.pid}", flush=True)
                if p.is_alive():
                    print(f"terminate {p.pid}", flush=True)
                    p.terminate()
    else:
        assert_never(opts)
    end_raw = time.perf_counter_ns()
    if isinstance(opts, api.DaskFutures|api.Fiab|api.MultiHost):
        print(f"total elapsed time: {(end_raw - start_raw) / 1e9: .3f}")
        print(f"in-cluster time: {(end_fine - start_fine) / 1e9: .3f}")
    else:
        print(f"total elapsed time: {(end_raw - start_raw) / 1e9: .3f}")
    if isinstance(opts, api.Fiab):
        shm_client.shutdown()
