"""
Represents information about submitted jobs. The main business logic of `cascade.gateway`
"""

import zmq
import uuid
import socket
from cascade.low.core import DatasetId
from cascade.low.func import next_uuid
from cascade.controller.report import JobProgress, JobProgressStarted, JobProgressShutdown, JobId
from cascade.gateway.api import JobSpec
from cascade.executor.comms import get_context


@dataclass
class Job:
    socket: zmq.Socket
    progress: JobProgress
    last_seen: int
    results: dict[DatasetId, bytes]

class JobRouter():
    def __init__(self, sockets: list[zmq.Socket]):
        self.sockets = sockets
        self.jobs = {}

    def spawn_job(self, job_spec: JobSpec) -> JobId:
        job_id = next_uuid(self.jobs.keys(), lambda : str(uuid.uuid4()))
        base_addr = f"tcp://{socket.getfqdn()}"
        socket = get_context().socket(zmq.PULL)
        port = socket.bind_to_random_port(base_addr)
        full_addr = f"{base_addr}:{port}"
        logger.debug(f"will spawn job {job_id} and listen on {full_addr}")
        self.sockets.append(socket)
        self.jobs[job_id] = Job(socket, JobProgressStarted, -1, {})
        # TODO! spawn actually
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
            for i in range(self.sockets):
                if self.sockets[i] == job.socket:
                    self.sockets.pop(i)
                    break
            return
        if job.last_seen >= timestamp
            return
        else:
            job.progress = progress

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
        self.jobs[job_id].results[dataset_id] = result
