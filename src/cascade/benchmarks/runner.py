"""
Runs actual job: controller + scheduler + executor
"""

import logging
from cascade import Cascade
import cascade.low.into
from dask.distributed import LocalCluster
from cascade.executors.dask_delayed import job2delayed
from cascade.executors.dask_futures import DaskFuturisticExecutor
from cascade.controller.impl import CascadeController
import cascade.scheduler
from cascade.scheduler.api import Scheduler
from cascade.scheduler.dynamic import DynamicScheduler, Config as DsConfig
from cascade.scheduler.transformers import FusingTransformer
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

def run_job_on(graph: cascade.graph.Graph, opts: api.Options):
    job = cascade.low.into.graph2job(graph)
    exe_rec = cascade.executors.simulator.placeholder_execution_record(job)
    dyn: Scheduler|None = None
    shm: Process|None = None
    fusing = FusingTransformer()
    start_raw = time.perf_counter_ns()
    if isinstance(opts, api.DaskDelayed):
        Cascade(graph).execute()
    elif isinstance(opts, api.Fiab):
        gb4 = 4 * (1024**3)
        port = 12345
        shm_api.publish_client_port(port)
        shm = Process(target=shm_server.entrypoint, args=(port, gb4, fiab_logging))
        shm.start()
        controller = CascadeController()
        executor = SingleHostExecutor(ExecutorConfig(opts.workers, 1024))
        if opts.dyn_sched:
            schedule = cascade.scheduler.Schedule.empty()
            dyn = DynamicScheduler(DsConfig(worker_eligible_mem_threshold=1))
        else:
            schedule = cascade.scheduler.schedule(job, executor.get_environment()).get_or_raise()
        if opts.fusing:
            schedule = fusing.transform(schedule)
        start_fine = time.perf_counter_ns()
        try:
            controller.submit(job, schedule, executor, dynamic_scheduler=dyn, execution_record=exe_rec)
            logging.info("controller in fiab done")
            executor.procwatch.join()
        except Exception as e:
            shm_client.shutdown()
            raise
    elif isinstance(opts, api.DaskThreaded):
        dly = job2delayed(job)
        cnts = {k: sum(1 for v in dly.values() if k in v[2]) for k in dly}
        sinks = [k for k in cnts if cnts[k] == 0]
        get(dly, sinks)
    elif isinstance(opts, api.DaskFutures):
        env = cascade.low.core.Environment(hosts={
            f'{i}': cascade.low.core.Host(cpu=1, gpu=0, memory_mb=1024)
            for i in range(opts.workers)
        })
        with LocalCluster(n_workers=opts.workers, processes=True, dashboard_address=":0") as clu:
            start_fine = time.perf_counter_ns()
            exe = DaskFuturisticExecutor(clu, env)
            if opts.dyn_sched:
                sched = cascade.scheduler.Schedule.empty()
                dyn = DynamicScheduler(DsConfig(worker_eligible_mem_threshold=1))
            else:
                sched = cascade.scheduler.schedule(job, exe.get_environment()).get_or_raise()
            if opts.fusing:
                sched = fusing.transform(sched)
            CascadeController().submit(job, sched, exe, dynamic_scheduler=dyn, execution_record=exe_rec)
    else:
        assert_never(opts)
    end_raw = time.perf_counter_ns()
    if isinstance(opts, api.DaskFutures|api.Fiab):
        print(f"total elapsed time: {(end_raw - start_raw) / 1e9: .3f}")
        print(f"in-cluster time: {(end_raw - start_fine) / 1e9: .3f}")
        if isinstance(dyn, DynamicScheduler):
            print(f"total time in dynamic scheduler: {dyn.cum_time / 1e9: .3f}")
    else:
        print(f"total elapsed time: {(end_raw - start_raw) / 1e9: .3f}")
    if isinstance(opts, api.Fiab):
        shm_client.shutdown()
