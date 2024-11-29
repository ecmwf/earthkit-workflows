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

import os
from pathlib import Path
import fire
import cascade.benchmarks.api as api
import cascade.benchmarks.job1 as job1
from cascade.benchmarks.distributed import launch_zmq_worker, launch_zmq_controller, ZmqWorkerHostSpec, ZmqControllerSpec, ZmqClusterSpec, launch_from_specs
from cascade.low.func import msum
from cascade.benchmarks.local import run_job_on as run_locally
from cascade.graph import Graph
from cascade.low.core import JobInstance
import logging
from cascade.graph import deduplicate_nodes
import cascade.low.into
from cascade.scheduler.impl import naive_bfs_layers, naive_dfs_layers
from cascade.executors.simulator import placeholder_execution_record
import logging

logger = logging.getLogger("cascade.benchmarks")

def get_job(id_: str) -> JobInstance:
    graphs = {
        "j1.prob": job1.get_prob(),
        "j1.ensms": job1.get_ensms(),
        "j1.efi": job1.get_efi(),
    }
    union = lambda prefix : deduplicate_nodes(msum((v for k, v in graphs.items() if k.startswith(prefix)), Graph))
    graphs["j1.all"] = union("j1.")
    return cascade.low.into.graph2job(graphs[id_])

def main_local(job: str, executor: str, workers: int, hosts: int|None = None) -> None:
    os.environ["CLOUDPICKLE"] = "yes" # for fiab desers
    logging.basicConfig(level="INFO", format="{asctime}:{levelname}:{name}:{process}:{message:1.10000}", style="{")
    logging.getLogger("cascade").setLevel(level="DEBUG")
    logging.getLogger("forecastbox").setLevel(level="DEBUG")
    opts: api.Options

    match executor:
        case "fiab":
            opts = api.Fiab(workers=workers)
        case "dask.delayed":
            opts = api.DaskDelayed()
        case "dask.futures":
            opts = api.DaskFutures(workers=workers)
        case "dask.threaded":
            opts = api.DaskThreaded()
        case "multihost":
            if not hosts:
                raise ValueError
            opts = api.MultiHost(hosts=hosts, workers_per_host=workers)
        case "zmq":
            if not hosts:
                raise ValueError
            opts = api.ZmqBackbone(hosts=hosts, workers_per_host=workers)
        case _:
            raise NotImplementedError(executor)

    jobInstance = get_job(job)
    run_locally(jobInstance, opts)

def main_dist(job: str, idx: int, controller_url: str, hosts: int = 3, workers_per_host: int = 10, shm_vol_gb: int = 64) -> None:
    jobInstance = get_job(job) 
    port_base = 12345
    cspec = ZmqControllerSpec(job=jobInstance, url=controller_url)
    wspec = [
        ZmqWorkerHostSpec(workers=workers_per_host, zmq_port=port_base, shm_port = port_base+1, shm_vol_gb = shm_vol_gb)
        for i in range(hosts)
    ]
    specs = ZmqClusterSpec(controller=cspec, worker_hosts=wspec)
    # we subtract -1 because we use slurm procid as idx (ie, there 0 -> controller, 1+ -> worker)
    launch_from_specs(specs, idx - 1)

if __name__ == "__main__":
    fire.Fire({"local": main_local, "dist": main_dist})
