# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Represents information about submitted jobs. The main business logic of `cascade.gateway`"""

import itertools
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from socket import getfqdn
from typing import Iterable

import orjson
import zmq

from cascade.controller.report import (
    JobId,
    JobProgress,
    JobProgressShutdown,
    JobProgressStarted,
)
from cascade.executor.comms import get_context
from cascade.gateway.api import JobSpec
from cascade.low.core import DatasetId
from cascade.low.func import next_uuid

logger = logging.getLogger(__name__)


@dataclass
class Job:
    socket: zmq.Socket
    progress: JobProgress
    last_seen: int
    results: dict[DatasetId, bytes]


def _spawn_local(job_spec: JobSpec, addr: str, job_id: JobId) -> None:
    base = [
        "python",
        "-m",
        "cascade.benchmarks",
        "local",
    ]
    if job_spec.benchmark_name is not None:
        if job_spec.job_instance is not None:
            raise TypeError("specified both benchmark name and job instance")
        base += ["--job", job_spec.benchmark_name]
    else:
        if job_spec.job_instance is None:
            raise TypeError("specified neither benchmark name nor job instance")
        with open(f"/tmp/{job_id}.json", "wb") as f:
            f.write(orjson.dumps(job_spec.job_instance.dict()))
        base += ["--instance", f"/tmp/{job_id}.json"]

    infra = [
        "--workers_per_host",
        f"{job_spec.workers_per_host}",
        "--hosts",
        f"{job_spec.hosts}",
    ]
    report = ["--report_address", f"{addr},{job_id}"]
    subprocess.Popen(base + infra + report, env={**os.environ, **job_spec.envvars})


def _spawn_slurm(job_spec: JobSpec, addr: str, job_id: JobId) -> None:
    extra_vars = {
        "EXECUTOR_HOSTS": str(job_spec.hosts),
        "WORKERS_PER_HOST": str(job_spec.workers_per_host),
        # NOTE put to infra specs
        "SHM_VOL_GB": "64",
        "REPORT_ADDRESS": f"{addr},{job_id}",
    }
    if job_spec.benchmark_name is not None:
        if job_spec.job_instance is not None:
            raise TypeError("specified both benchmark name and job instance")
        extra_vars["JOB"] = job_spec.benchmark_name
    else:
        if job_spec.job_instance is None:
            raise TypeError("specified neither benchmark name nor job instance")
        with open(f"./localConfigs/_tmp/{job_id}.json", "wb") as f:
            f.write(orjson.dumps(job_spec.job_instance.dict()))
        extra_vars["INSTANCE"] = f"./localConfigs/_tmp/{job_id}.json"
    subprocess.run(
        ["cp", "localConfigs/gateway.sh", f"localConfigs/_tmp/{job_id}"], check=True
    )
    with open(f"./localConfigs/_tmp/{job_id}", "a") as f:
        for k, v in itertools.chain(job_spec.envvars.items(), extra_vars.items()):
            f.write(f"export {k}={v}\n")
    subprocess.Popen(["./scripts/launch_slurm.sh", f"localConfigs/_tmp/{job_id}"])


def _spawn_subprocess(job_spec: JobSpec, addr: str, job_id: JobId) -> None:
    if job_spec.use_slurm:
        _spawn_slurm(job_spec, addr, job_id)
    else:
        _spawn_local(job_spec, addr, job_id)


class JobRouter:
    def __init__(self, poller: zmq.Poller):
        self.poller = poller
        self.jobs: dict[str, Job] = {}

    def spawn_job(self, job_spec: JobSpec) -> JobId:
        job_id = next_uuid(self.jobs.keys(), lambda: str(uuid.uuid4()))
        base_addr = f"tcp://{getfqdn()}"
        socket = get_context().socket(zmq.PULL)
        port = socket.bind_to_random_port(base_addr)
        full_addr = f"{base_addr}:{port}"
        logger.debug(f"will spawn job {job_id} and listen on {full_addr}")
        self.poller.register(socket, flags=zmq.POLLIN)
        self.jobs[job_id] = Job(socket, JobProgressStarted, -1, {})
        _spawn_subprocess(job_spec, full_addr, job_id)
        return job_id

    def progress_of(self, job_ids: Iterable[JobId]) -> dict[JobId, JobProgress]:
        if not job_ids:
            job_ids = self.jobs.keys()
        return {job_id: self.jobs[job_id].progress for job_id in job_ids}

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> bytes:
        return self.jobs[job_id].results[dataset_id]

    def maybe_update(
        self, job_id: JobId, progress: JobProgress | None, timestamp: int
    ) -> None:
        if progress is None:
            return
        job = self.jobs[job_id]
        if progress == JobProgressShutdown:
            self.poller.unregister(job.socket)
            return
        if job.last_seen >= timestamp:
            return
        else:
            job.progress = progress

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
        self.jobs[job_id].results[dataset_id] = result
