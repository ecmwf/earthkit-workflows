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
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns
from typing import Any

import fire
import orjson

import cascade.executor.platform as platform
from cascade.controller.impl import run
from cascade.controller.report import Reporter
from cascade.deployment.logging import DefaultLoggingConfig, LoggingConfig, init_from_cliparam, init_from_obj
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.executor import Executor, address_of
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.core import DatasetId, HostId, JobInstance, JobInstanceRich, globalIdx2hostId, localIdx2hostId
from cascade.low.exceptions import CascadeInfrastructureError
from cascade.low.func import msum
from cascade.scheduler.precompute import precompute

logger = logging.getLogger(__name__)


def launch_executor(
    job: JobInstanceRich,
    controller_address: BackboneAddress,
    workers_per_host: int,
    portBase: int,
    host: HostId,
    shm_vol_gb: int | None,
    loggingConfig: LoggingConfig,
    url_base: str,
):
    init_from_obj(loggingConfig, "executor")
    # NOTE we install explicitly as we may be a forkserver/spawn child, or a standalone process
    platform.install_sigterm_exit()
    executor: Executor | None = None
    try:
        executor = Executor(
            job,
            controller_address,
            workers_per_host,
            host,
            portBase,
            shm_vol_gb,
            loggingConfig,
            url_base,
        )
        executor.register()
        executor.recv_loop()
    except Exception as e:
        # NOTE we log this to get the stacktrace into the logfile
        # NOTE we do *not* raise -- we keep system exit 0. Otherwise, the orchestrator (eg slurm) could have killed the whole job before the controller has the chance to receive and report the message to the gateway
        logger.exception("executor failure, swallowing")
    finally:
        # NOTE safe to call even if already terminated
        if executor is not None:
            executor.terminate()


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
    c = f"tcp://localhost:{portBase}"
    # NOTE the reporter makes sure the gateway learns of failure even if we die before `run` starts
    reporter = Reporter(report_address)
    try:
        ps = []
        # executors forking
        for i, executor in enumerate(range(hosts)):
            # NOTE forkserver/spawn seem to forget venv, we need fork
            logger.debug(f"forking into executor on host {i}")
            p = platform.get_mp_ctx("executor-loc").Process(
                target=launch_executor,
                args=(
                    job,
                    c,
                    workers,
                    portBase + 1 + i * 10,
                    localIdx2hostId(i),
                    None,
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
                raise CascadeInfrastructureError(description=f"executor {i} failed to live due to {p.exitcode}")

        # start bridge itself
        logger.debug("starting bridge")
        b = Bridge(c, hosts, job.checkpointSpec)
    except BaseException as e:
        # NOTE includes eg SystemExit due to sigterm
        reporter.send_failure_and_log(e)
        raise
    result = run(job, b, preschedule, reporter)
    return result.outputs


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
    platform.install_sigterm_exit()
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
    loggingConfigSer: str | None = None,
) -> None:
    """Entrypoint for *both* controller and worker -- they are on different hosts! Distinguished by idx: 0 for
    controller, 1+ for worker. Assumed to come from slurm procid.
    """
    platform.install_sigterm_exit()

    jobInstanceRich = _deserialize(instance)

    if idx == 0:
        loggingConfig = init_from_cliparam(loggingConfigSer, "controller")
        reporter = Reporter(report_address)
        b = None
        tp = None
        try:
            tp = ThreadPoolExecutor(max_workers=1)
            preschedule_fut = tp.submit(precompute, jobInstanceRich.jobInstance)
            b = Bridge(controller_url, hosts, jobInstanceRich.checkpointSpec)
            preschedule = preschedule_fut.result()
        except BaseException as e:
            # NOTE includes eg SystemExit due to sigterm
            reporter.send_failure_and_log(e)
            if b is not None:
                b.shutdown()
            raise
        finally:
            tp.shutdown()
        run(jobInstanceRich, b, preschedule, reporter)

    else:
        loggingConfig = init_from_cliparam(loggingConfigSer, f"executor_{idx}")
        launch_executor(
            jobInstanceRich,
            controller_url,
            workers_per_host,
            12345,
            globalIdx2hostId(idx),
            shm_vol_gb,
            loggingConfig=loggingConfig,
            url_base=f"tcp://{platform.get_bindabble_self()}",
        )


if __name__ == "__main__":
    fire.Fire({"local": main_local, "dist": main_dist})
