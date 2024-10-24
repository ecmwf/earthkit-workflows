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

import os
from pathlib import Path
os.environ["LD_LIBRARY_PATH"] = str(Path.home() / ".local" / "lib") # mir, eccodes, etc
import fire
import cascade.benchmarks.api as api
import cascade.benchmarks.job1 as job1
from cascade.low.func import msum
from cascade.benchmarks.runner import run_job_on
from cascade.graph import Graph
import logging
from cascade.graph import deduplicate_nodes

def main(job: str, executor: str, dynamic: bool, workers: int, fusing: bool) -> None:
    os.environ["CLOUDPICKLE"] = "yes" # for fiab desers
    logging.basicConfig(level="INFO", format="{asctime}:{levelname}:{name}:{process}:{message:1.10000}", style="{")
    logging.getLogger("cascade").setLevel(level="DEBUG")
    logging.getLogger("forecastbox").setLevel(level="DEBUG")
    opts: api.Options

    match executor:
        case "fiab":
            opts = api.Fiab(dyn_sched=dynamic, fusing=fusing, workers=workers)
        case "dask.delayed":
            opts = api.DaskDelayed()
        case "dask.futures":
            opts = api.DaskFutures(workers=workers, dyn_sched=dynamic, fusing=fusing)
        case "dask.threaded":
            opts = api.DaskThreaded()
        case _:
            raise NotImplementedError(executor)

    jobs = {
        "j1.prob": job1.get_prob(),
        "j1.ensms": job1.get_ensms(),
        "j1.efi": job1.get_efi(),
    }
    union = lambda prefix : deduplicate_nodes(msum((v for k, v in jobs.items() if k.startswith(prefix)), Graph))
    jobs["j1.all"] = union("j1.")

    run_job_on(jobs[job], opts)

if __name__ == "__main__":
    fire.Fire(main)
