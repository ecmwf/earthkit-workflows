# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contains utility methods for benchmark definitions and cluster starting"""

# TODO rework, simplify, split into benchmark.util and cluster.setup or smth

import logging
import logging.config
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

import orjson

import cascade.executor.platform as platform
import cascade.low.into
from cascade.controller.impl import run
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.executor import Executor
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.core import JobInstance
from cascade.low.func import msum
from cascade.scheduler.precompute import precompute
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
        elif benchmark.startswith("matmul"):
            import cascade.benchmarks.matmul as matmul

            return matmul.get_job()
        elif benchmark.startswith("dist"):
            import cascade.benchmarks.dist as dist

            return dist.get_job()
        elif benchmark.startswith("dask"):
            import cascade.benchmarks.dask as dask

            return dask.get_job(benchmark[len("dask.") :])
        else:
            raise NotImplementedError(benchmark)
    else:
        raise TypeError("specified neither benchmark name nor job instance")


def get_cuda_count() -> int:
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            # TODO we dont want to just count, we want to actually use literally these ids
            # NOTE this is particularly useful for "" value -- careful when refactoring
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            visible_count = sum(1 for e in visible if e == ",") + (1 if visible else 0)
            return visible_count
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
        logger.exception("unable to determine available gpus")
        gpus = 0
    return gpus


def get_gpu_count(host_idx: int, worker_count: int) -> int:
    if sys.platform == "darwin":
        # we should inspect some gpu capabilities details to prevent overcommit
        return worker_count
    else:
        if host_idx == 0:
            return get_cuda_count()
        else:
            return 0


def launch_executor(
    job_instance: JobInstance,
    controller_address: BackboneAddress,
    workers_per_host: int,
    portBase: int,
    i: int,
    shm_vol_gb: int | None,
    gpu_count: int,
    log_base: str | None,
    url_base: str,
):
    if log_base is not None:
        log_base = f"{log_base}.host{i}"
        log_path = f"{log_base}.txt"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    try:
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
            url_base,
        )
        executor.register()
        executor.recv_loop()
    except Exception:
        # NOTE we log this to get the stacktrace into the logfile
        logger.exception("executor failure")
        raise


def run_locally(
    job: JobInstance,
    hosts: int,
    workers: int,
    portBase: int = 12345,
    log_base: str | None = None,
    report_address: str | None = None,
) -> None:
    if log_base is not None:
        log_path = f"{log_base}.controller.txt"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    logger.debug(f"local run starting with {hosts=} and {workers=} on {portBase=}")
    launch = perf_counter_ns()
    c = f"tcp://localhost:{portBase}"
    m = f"tcp://localhost:{portBase+1}"
    ps = []
    try:
        # executors forking
        for i, executor in enumerate(range(hosts)):
            gpu_count = get_gpu_count(i, workers)
            # NOTE forkserver/spawn seem to forget venv, we need fork
            logger.debug(f"forking into executor on host {i}")
            p = platform.get_mp_ctx("executor-loc").Process(
                target=launch_executor,
                args=(
                    job,
                    c,
                    workers,
                    portBase + 1 + i * 10,
                    i,
                    None,
                    gpu_count,
                    log_base,
                    "tcp://localhost",
                ),
            )
            p.start()
            ps.append(p)

        # compute preschedule
        preschedule = precompute(job)

        # check processes started healthy
        for i, p in enumerate(ps):
            if not p.is_alive():
                # TODO ideally we would somehow connect this with the Register message
                # consumption in the Controller -- but there we don't assume that
                # executors are on the same physical host
                raise ValueError(f"executor {i} failed to live due to {p.exitcode}")

        # start bridge itself
        logger.debug("starting bridge")
        b = Bridge(c, hosts)
        start = perf_counter_ns()
        result = run(job, b, preschedule, report_address=report_address)
        end = perf_counter_ns()
        print(
            f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s"
        )
        if os.environ.get("CASCADE_DEBUG_PRINT"):
            for key, value in result.outputs.items():
                print(f"{key} => {value}")
    except Exception:
        # NOTE we log this to get the stacktrace into the logfile
        logger.exception("controller failure, proceed with executor shutdown")
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
        gpu_count = get_gpu_count(0, workers_per_host)
        launch_executor(
            jobInstance,
            controller_url,
            workers_per_host,
            12345,
            idx,
            shm_vol_gb,
            gpu_count,
            f"tcp://{platform.get_bindabble_self()}",
        )
