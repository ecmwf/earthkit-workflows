# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Entrypoint for running the benchmark suite

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
import multiprocessing
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

import fire
import orjson

import cascade.low.into
from cascade.controller.impl import run
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.executor import Executor
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.core import JobInstance
from cascade.low.func import msum
from cascade.scheduler.graph import precompute
from earthkit.workflows.graph import Graph, deduplicate_nodes

logger = logging.getLogger("cascade.benchmarks")


def get_job(benchmark: str | None, instance_path: str | None) -> JobInstance:
    # NOTE because of os.environ, we don't import all... ideally we'd have some file-based init/config mech instead
    if benchmark is not None and instance_path is not None:
        raise TypeError("specified both benchmark name and job instance")
    elif instance_path is not None:
        with open(instance_path, "rb") as f:
            d = orjson.loads(f.read())
            return JobInstance(**d)
    elif benchmark is not None:
        if benchmark.startswith("j1"):
            import cascade.benchmarks.job1 as job1

            graphs = {
                "j1.prob": job1.get_prob(),
                "j1.ensms": job1.get_ensms(),
                "j1.efi": job1.get_efi(),
            }
            union = lambda prefix: deduplicate_nodes(
                msum((v for k, v in graphs.items() if k.startswith(prefix)), Graph)
            )
            graphs["j1.all"] = union("j1.")
            return cascade.low.into.graph2job(graphs[benchmark])
        elif benchmark.startswith("generators"):
            import cascade.benchmarks.generators as generators

            return generators.get_job()
        else:
            raise NotImplementedError(benchmark)
    else:
        raise TypeError("specified neither benchmark name nor job instance")


def get_gpu_count() -> int:
    try:
        gpus = sum(
            1
            for l in subprocess.run(
                ["nvidia-smi", "--list-gpus"], check=True, capture_output=True
            )
            .stdout.decode("ascii")
            .split("\n")
            if "GPU" in l
        )
    except:
        # TODO support macos
        logger.exception("unable to determine available gpus")
        gpus = 0
    return gpus


def launch_executor(
    job_instance: JobInstance,
    controller_address: BackboneAddress,
    workers_per_host: int,
    portBase: int,
    i: int,
    shm_vol_gb: int | None,
    gpu_count: int,
    log_base: str | None,
):
    if log_base is not None:
        log_base = f"{log_base}.host{i}"
        log_path = f"{log_base}.txt"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    logger.info(f"will set {gpu_count} gpus on host {i}")
    os.environ["CASCADE_GPU_COUNT"] = str(gpu_count)
    executor = Executor(
        job_instance,
        controller_address,
        workers_per_host,
        f"h{i}",
        portBase,
        shm_vol_gb,
        log_base,
    )
    executor.register()
    executor.recv_loop()


def run_locally(
    job: JobInstance,
    hosts: int,
    workers: int,
    portBase: int = 12345,
    log_base: str | None = None,
    report_address: str | None = None,
):
    if log_base is not None:
        log_path = f"{log_base}.controller.txt"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    launch = perf_counter_ns()
    preschedule = precompute(job)
    c = f"tcp://localhost:{portBase}"
    m = f"tcp://localhost:{portBase+1}"
    ps = []
    for i, executor in enumerate(range(hosts)):
        if i == 0:
            gpu_count = get_gpu_count()
        else:
            gpu_count = 0
        # NOTE forkserver/spawn seem to forget venv, we need fork
        p = multiprocessing.get_context("fork").Process(
            target=launch_executor,
            args=(job, c, workers, portBase + 1 + i * 10, i, None, gpu_count, log_base),
        )
        p.start()
        ps.append(p)
    try:
        b = Bridge(c, hosts)
        start = perf_counter_ns()
        run(job, b, preschedule, report_address=report_address)
        end = perf_counter_ns()
        print(
            f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s"
        )
    except:
        for p in ps:
            if p.is_alive():
                callback(m, ExecutorShutdown())
                import time

                time.sleep(1)
                p.kill()
        raise


def main_local(
    workers_per_host: int,
    hosts: int = 1,
    report_address: str | None = None,
    job: str | None = None,
    instance: str | None = None,
    port_base: int = 12345,
    log_base: str | None = None,
) -> None:
    jobInstance = get_job(job, instance)
    run_locally(
        jobInstance,
        hosts,
        workers_per_host,
        report_address=report_address,
        portBase=port_base,
        log_base=log_base,
    )


def main_dist(
    idx: int,
    controller_url: str,
    hosts: int = 3,
    workers_per_host: int = 10,
    shm_vol_gb: int = 64,
    job: str | None = None,
    instance: str | None = None,
    report_address: str | None = None,
) -> None:
    """Entrypoint for *both* controller and worker -- they are on different hosts! Distinguished by idx: 0 for
    controller, 1+ for worker. Assumed to come from slurm procid.
    """
    launch = perf_counter_ns()

    jobInstance = get_job(job, instance)

    if idx == 0:
        logging.config.dictConfig(logging_config)
        tp = ThreadPoolExecutor(max_workers=1)
        preschedule_fut = tp.submit(precompute, jobInstance)
        b = Bridge(controller_url, hosts)
        preschedule = preschedule_fut.result()
        tp.shutdown()
        start = perf_counter_ns()
        run(jobInstance, b, preschedule, report_address=report_address)
        end = perf_counter_ns()
        print(
            f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s"
        )
    else:
        gpu_count = get_gpu_count()
        launch_executor(
            jobInstance,
            controller_url,
            workers_per_host,
            12345,
            idx,
            shm_vol_gb,
            gpu_count,
        )


if __name__ == "__main__":
    fire.Fire({"local": main_local, "dist": main_dist})
