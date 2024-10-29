"""
Rather bulky test, spins up a bunch of processes and uvicorn servers

There is a bug somewhere that sometimes causes this to hang, not sure yet where it is

The dask/simulator executors are commented out so that this doesnt take infinite time.
Use when developing a major change
"""

from cascade.executors.multihost.worker_server import build_app
from cascade.executors.dask_futures import DaskFuturisticExecutor
from dask.distributed import LocalCluster
from cascade.scheduler.impl import naive_bfs_layers
from cascade.executors.instant import InstantExecutor
from cascade.executors.simulator import SimulatingExecutor, placeholder_execution_record
from cascade.executors.multihost.impl import RouterExecutor
from cascade.controller.impl import run
from cascade.low.builders import JobBuilder, TaskBuilder
import uvicorn
import time
from cascade.low.core import JobExecutionRecord, JobInstance, Environment, Worker
from cascade.low.views import param_source
import httpx
from multiprocessing import Process
import logging

logging.getLogger("httpcore").setLevel("ERROR")

def launch_executor(port: int, kind: str, job: JobInstance):
    executor: Executor
    if kind == "instant":
        executor = InstantExecutor(workers=2, job=job)
    elif kind == "simulator":
        task_inputs = {
            task_id: set(task_param_source.values())
            for task_id, task_param_source in param_source(job.edges).items()
        }
        env = Environment(workers={
            "w1": Worker(cpu=1, gpu=0, memory_mb=2048),
            "w2": Worker(cpu=1, gpu=0, memory_mb=2048),
        })
        executor = SimulatingExecutor(env, task_inputs, placeholder_execution_record(job))
    elif kind == "dask.futures":
        cluster = LocalCluster(n_workers=1, processes=True, dashboard_address=":0")
        executor = DaskFuturisticExecutor(cluster, job)
    else:
        raise NotImplementedError(kind)
    try:
        app = build_app(executor)
        uvicorn.run(app, host="0.0.0.0", port=port) # , log_level="debug")
    finally:
        if kind == "dask.futures":
            cluster.close()

def wait_for(client: httpx.Client, root_url: str) -> None:
    """Calls /status endpoint, retry on ConnectError"""
    i = 0
    while i < 10:
        try:
            rc = client.get(f"{root_url}/status")
            if not rc.status_code == 200:
                raise ValueError(f"failed to start {root_url}: {rc}")
            return
        except httpx.ConnectError:
            i += 1
            time.sleep(2)
    raise ValueError(f"failed to start {root_url}: no more retries")

def launch_cluster_and_run(start: int, kind: str, workers: int, job: JobInstance):
    client = httpx.Client()
    urls = [f"http://localhost:{start+i}" for i in range(workers)]
    ps = [
        Process(target=launch_executor, args=(int(url.rsplit(":",1)[1]),kind, job))
        for url in urls
    ]
    executor: RouterExecutor|None = None
    try:
        for p in ps:
            p.start()
        for url in urls:
            wait_for(client, url)
        executor = RouterExecutor(urls)
        schedule = naive_bfs_layers(job, JobExecutionRecord(), set()).get_or_raise()
        run(job, executor, schedule)
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()
        if executor is not None:
            executor.shutdown()
            print("executor exited")

def test_instant_simple():
    def test_func(x, y, z):
        return x + y + z

    # 1-node graph
    task = TaskBuilder.from_callable(test_func).with_values(x=1, y=2, z=3)
    job = JobInstance(tasks={"task": task}, edges=[])
    launch_cluster_and_run(5400, "instant", 1, job)
    # launch_cluster_and_run(5410, "simulator", 1, job)
    # launch_cluster_and_run(5420, "dask.futures", 1, job)

    # 2-node graph
    task1 = TaskBuilder.from_callable(test_func).with_values(x=1, y=2, z=3)
    task2 = TaskBuilder.from_callable(test_func).with_values(y=4, z=5)
    job = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .with_edge("task1", "task2", 0)
        .build()
        .get_or_raise()
    )
    launch_cluster_and_run(5500, "instant", 1, job)
    # launch_cluster_and_run(5510, "simulator", 1, job)
    # launch_cluster_and_run(5520, "dask.futures", 1, job)
