"""
Entrypoint for running the benchmark suite

Example:
```
python -m cascade.benchmarks --job j1.prob --executor fiab --dynamic True --workers 2 --fusing False
```

Make sure you correctly configure:
 - LD_LIBRARY_PATH (a few lines below, in this mod)
 - data_root in benchmarks.job1 (presumably run `download_*` from job1 first)
 - your venv (cascade, fiab, pproc-cascade, compatible version of earthkit-data, ...)
"""

# TODO rework, simplify

import os
from pathlib import Path
ld_ext = [str(Path.home() / ".local" / "lib")]
if (ld_orig := os.environ.get("LD_LIBRARY_PATH")):
    ld_ext.append(ld_orig)
os.environ["LD_LIBRARY_PATH"] = ":".join(ld_ext)
import fire
import cascade.benchmarks.api as api
import cascade.benchmarks.job1 as job1
from cascade.benchmarks.distributed import launch_zmq_worker, launch_zmq_controller
from cascade.low.func import msum
from cascade.benchmarks.local import run_job_on as run_locally
from cascade.graph import Graph
import logging
from cascade.graph import deduplicate_nodes
import cascade.low.into

def main(job: str, executor: str, workers: int, hosts: int|None = None, dist: str = "local", controller_url: str|None = None, host_id: int|None = None) -> None:
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

    jobs = {
        "j1.prob": job1.get_prob(),
        "j1.ensms": job1.get_ensms(),
        "j1.efi": job1.get_efi(),
    }
    union = lambda prefix : deduplicate_nodes(msum((v for k, v in jobs.items() if k.startswith(prefix)), Graph))
    jobs["j1.all"] = union("j1.")
    graph = jobs[job]
    jobInstance = cascade.low.into.graph2job(graph)

    if dist == "local":
        run_locally(jobInstance, opts)
    elif dist == "worker":
        if executor != "zmq":
            raise NotImplementedError
        if controller_url is None or host_id is None:
            raise ValueError
        launch_zmq_worker(workers, controller_url, host_id, jobInstance)
    elif dist == "controller":
        if executor != "zmq":
            raise NotImplementedError
        if controller_url is None or hosts is None:
            raise ValueError
        launch_zmq_controller(hosts, controller_url, jobInstance)
    else:
        raise NotImplementedError(dist)


if __name__ == "__main__":
    fire.Fire(main)
