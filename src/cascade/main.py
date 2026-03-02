# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Main entrypoints for cluster or local starting for executors and controllers"""

import logging
import logging.config
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns
from typing import Any

import fire
import orjson

import cascade.executor.platform as platform
from cascade.controller.impl import run
from cascade.deployment.logging import DefaultLoggingConfig, LoggingConfig, init_from_cliparam, init_from_obj
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.executor import Executor
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.core import DatasetId, JobInstance, JobInstanceRich
from cascade.low.func import msum
from cascade.scheduler.precompute import precompute

logger = logging.getLogger(__name__)


def _get_cuda_count() -> int:
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


def _get_gpu_count(host_idx: int, worker_count: int) -> int:
    if sys.platform == "darwin":
        # we should inspect some gpu capabilities details to prevent overcommit
        return worker_count
    else:
        if host_idx == 0:
            return _get_cuda_count()
        else:
            return 0


def launch_executor(
    job: JobInstanceRich,
    controller_address: BackboneAddress,
    workers_per_host: int,
    portBase: int,
    i: int,
    shm_vol_gb: int | None,
    gpu_count: int,
    loggingConfig: LoggingConfig,
    url_base: str,
):
    init_from_obj(loggingConfig, "executor")
    try:
        logger.info(f"will set {gpu_count} gpus on host {i}")
        os.environ["CASCADE_GPU_COUNT"] = str(gpu_count)
        executor = Executor(
            job.jobInstance,
            controller_address,
            workers_per_host,
            f"h{i}",
            portBase,
            shm_vol_gb,
            loggingConfig,
            url_base,
        )
        executor.register()
        executor.recv_loop()
    except Exception:
        # NOTE we log this to get the stacktrace into the logfile
        logger.exception("executor failure")
        raise


def run_locally(
    job: JobInstanceRich,
    hosts: int,
    workers: int,
    portBase: int = 12345,
    loggingConfigSer: str | None = None,
    report_address: str | None = None,
) -> dict[DatasetId, Any]:
    # NOTE the provided job may cary traces of imports we dont want to pollute executor with
    job = JobInstanceRich(**orjson.loads(job.model_dump_json().encode()))
    loggingConfig = init_from_cliparam(loggingConfigSer, "controller")
    logger.debug(f"local run starting with {hosts=} and {workers=} on {portBase=}")
    launch = perf_counter_ns()
    c = f"tcp://localhost:{portBase}"
    m = f"tcp://localhost:{portBase+1}"
    ps = []
    try:
        # executors forking
        for i, executor in enumerate(range(hosts)):
            gpu_count = _get_gpu_count(i, workers)
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
                    loggingConfig.withContext(f"host_{i}"),
                    "tcp://localhost",
                ),
            )
            p.start()
            ps.append(p)

        # compute preschedule
        preschedule = precompute(job.jobInstance)

        # check processes started healthy
        for i, p in enumerate(ps):
            if not p.is_alive():
                # TODO ideally we would somehow connect this with the Register message
                # consumption in the Controller -- but there we don't assume that
                # executors are on the same physical host
                raise ValueError(f"executor {i} failed to live due to {p.exitcode}")

        # start bridge itself
        logger.debug("starting bridge")
        b = Bridge(c, hosts, job.checkpointSpec)
        start = perf_counter_ns()
        result = run(job, b, preschedule, report_address=report_address)
        end = perf_counter_ns()
        print(
            f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s"
        )
        if os.environ.get("CASCADE_DEBUG_PRINT"):
            for key, value in result.outputs.items():
                print(f"{key} => {value}")
        return result.outputs
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

def _deserialize(instance_path: str) -> JobInstanceRich:
    with open(instance_path, "rb") as f:
        d = orjson.loads(f.read())
        return JobInstanceRich(**d)

def main_local(
    workers_per_host: int,
    instance: str,
    hosts: int = 1,
    report_address: str | None = None,
    port_base: int = 12345,
    loggingConfigSer: str | None = None,
) -> None:
    jobInstanceRich = _deserialize(instance)
    run_locally(
        jobInstanceRich,
        hosts,
        workers_per_host,
        report_address=report_address,
        portBase=port_base,
        loggingConfigSer=loggingConfigSer,
    )


def main_dist(
    idx: int,
    controller_url: str,
    instance: str,
    hosts: int = 3,
    workers_per_host: int = 10,
    shm_vol_gb: int = 64,
    report_address: str | None = None,
) -> None:
    """Entrypoint for *both* controller and worker -- they are on different hosts! Distinguished by idx: 0 for
    controller, 1+ for worker. Assumed to come from slurm procid.
    """
    launch = perf_counter_ns()

    jobInstanceRich = _deserialize(instance)

    if idx == 0:
        logging.config.dictConfig(logging_config)
        tp = ThreadPoolExecutor(max_workers=1)
        preschedule_fut = tp.submit(precompute, jobInstanceRich.jobInstance)
        b = Bridge(controller_url, hosts, jobInstanceRich.checkpointSpec)
        preschedule = preschedule_fut.result()
        tp.shutdown()
        start = perf_counter_ns()
        run(jobInstanceRich, b, preschedule, report_address=report_address)
        end = perf_counter_ns()
        print(
            f"compute took {(end-start)/1e9:.3f}s, including startup {(end-launch)/1e9:.3f}s"
        )
    else:
        gpu_count = _get_gpu_count(0, workers_per_host)
        launch_executor(
            jobInstanceRich,
            controller_url,
            workers_per_host,
            12345,
            idx,
            shm_vol_gb,
            gpu_count,
            loggingConfig = DefaultLoggingConfig, # TODO handle logging for dist scenario
            url_base = f"tcp://{platform.get_bindabble_self()}",
        )

if __name__ == "__main__":
    fire.Fire({"local": main_local, "dist": main_dist})
