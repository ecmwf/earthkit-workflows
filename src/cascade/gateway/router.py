"""
Represents information about submitted jobs. The main business logic of `cascade.gateway`
"""

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from socket import getfqdn

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

def _spawn_local_job(job_spec: JobSpec, addr: str, job_id: JobId) -> None:
    base = ["python", "-m", "cascade.benchmarks", "local", "--job", job_spec.benchmark_name]
    infra = ["--workers_per_host", f"{job_spec.workers_per_host}", "--hosts", f"{job_spec.hosts}"]
    report = ["--report_address", f"{addr},{job_id}"]
    job_env = {
        "GENERATORS_N": "8",
        "GENERATORS_K": "10",
        "GENERATORS_L": "4",
    } # TODO this must be generic
    subprocess.Popen(base + infra + report, env = {**os.environ, **job_env})

class JobRouter():
    def __init__(self, poller: zmq.Poller):
        self.poller = poller
        self.jobs = {}

    def spawn_job(self, job_spec: JobSpec) -> JobId:
        job_id = next_uuid(self.jobs.keys(), lambda : str(uuid.uuid4()))
        base_addr = f"tcp://{getfqdn()}"
        socket = get_context().socket(zmq.PULL)
        port = socket.bind_to_random_port(base_addr)
        full_addr = f"{base_addr}:{port}"
        logger.debug(f"will spawn job {job_id} and listen on {full_addr}")
        self.poller.register(socket, flags=zmq.POLLIN)
        self.jobs[job_id] = Job(socket, JobProgressStarted, -1, {})
        _spawn_local_job(job_spec, full_addr, job_id)
        return job_id

    def progres_of(self, job_id: JobId) -> JobProgress:
        return self.jobs[job_id].progress

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> bytes:
        return self.jobs[job_id].results[dataset_id]

    def maybe_update(self, job_id: JobId, progress: JobProgress|None, timestamp: int) -> None:
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
