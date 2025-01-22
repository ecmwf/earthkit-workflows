"""
Entrypoint for running the benchmark suite

Example:
```
python -m cascade.benchmarks --job j1.prob --executor fiab --dynamic True --workers 2 --fusing False
```

Make sure you correctly configure:
 - LD_LIBRARY_PATH (a few lines below, in this mod)
 - JOB1_{...} as noted in benchmarks.job1 (presumably run `download_*` from job1 first)
 - your venv (cascade, fiab, pproc-cascade, compatible version of earthkit-data, ...)
"""

# TODO rework, simplify

import logging
import logging.config
from multiprocessing import Process
import os
from time import perf_counter_ns
from concurrent.futures import ThreadPoolExecutor

import fire

from cascade.low.func import msum
import cascade.low.into
from cascade.graph import Graph
from cascade.low.core import JobInstance
from cascade.executor.config import logging_config
from cascade.graph import deduplicate_nodes
from cascade.controller.impl import run
from cascade.scheduler.graph import precompute
from cascade.executor.executor import Executor
from cascade.executor.bridge import Bridge
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.executor.comms import callback


logger = logging.getLogger("cascade.benchmarks")

def get_job(id_: str) -> JobInstance:
    # NOTE because of os.environ, we don't import all... ideally we'd have some file-based init/config mech instead
    if id_.startswith("j1"):
        import cascade.benchmarks.job1 as job1
        graphs = {
            "j1.prob": job1.get_prob(),
            "j1.ensms": job1.get_ensms(),
            "j1.efi": job1.get_efi(),
        }
        union = lambda prefix : deduplicate_nodes(msum((v for k, v in graphs.items() if k.startswith(prefix)), Graph))
        graphs["j1.all"] = union("j1.")
        return cascade.low.into.graph2job(graphs[id_])
    elif id_.startswith("generators"):
        import cascade.benchmarks.generators as generators
        return generators.get_job()
    else:
        raise NotImplementedError(id_)

def launch_executor(job_instance: JobInstance, controller_address: BackboneAddress, workers_per_host: int, portBase: int, i: int, shm_vol_gb: int|None):
    logging.config.dictConfig(logging_config)
    executor = Executor(job_instance, controller_address, workers_per_host, f"h{i}", portBase, shm_vol_gb)
    executor.register()
    executor.recv_loop()

def run_locally(job: JobInstance, hosts: int, workers: int, portBase: int = 12345):
    logging.config.dictConfig(logging_config)
    launch = perf_counter_ns()
    preschedule = precompute(job)
    c = f"tcp://localhost:{portBase}"
    m = f"tcp://localhost:{portBase+1}"
    ps = []
    spawn = perf_counter_ns()
    for i, executor in enumerate(range(hosts)):
        p = Process(target=launch_executor, args=(job, c, workers, portBase+1+i*10, i, None))
        p.start()
        ps.append(p)
    try:
        b = Bridge(c, hosts)
        start = perf_counter_ns()
        run(job, b, preschedule)
        end = perf_counter_ns()
        print(f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s")
    except Exception as e:
        for p in ps:
            if p.is_alive():
                callback(m, ExecutorShutdown())
                import time
                time.sleep(1)
                p.kill()
        raise

def main_local(job: str, workers_per_host: int, hosts: int = 1) -> None:
    jobInstance = get_job(job)
    run_locally(jobInstance, hosts, workers_per_host)

def main_dist(job: str, idx: int, controller_url: str, hosts: int = 3, workers_per_host: int = 10, shm_vol_gb: int = 64) -> None:
    """Entrypoint for *both* controller and worker -- they are on different hosts! Distinguished by idx: 0 for
    controller, 1+ for worker. Assumed to come from slurm procid."""
    launch = perf_counter_ns()

    jobInstance = get_job(job) 

    if idx == 0:
        tp = ThreadPoolExecutor(max_workers=1)
        preschedule_fut = tp.submit(precompute, jobInstance)
        logging.config.dictConfig(logging_config)
        b = Bridge(controller_url, hosts)
        preschedule = preschedule_fut.result()
        tp.shutdown()
        start = perf_counter_ns()
        run(jobInstance, b, preschedule)
        end = perf_counter_ns()
        print(f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s")
    else:
        launch_executor(jobInstance, controller_url, workers_per_host, 12345, idx, shm_vol_gb)

if __name__ == "__main__":
    fire.Fire({"local": main_local, "dist": main_dist})
