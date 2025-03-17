"""
Represents information about submitted jobs. The main business logic of `cascade.gateway`
"""

import zmq
from cascade.low.core import DatasetId, Worker, JobInstance
from cascade.controller.report import JobProgress, JobId

class JobRouter():
    def __init__(self, sockets: list[zmq.Socket]):
        self.sockets = sockets

    def spawn_job(self, job: JobInstance, resources: list[Worker]) -> JobId:
        raise NotImplementedError

    def progres_of(self, job_id: JobId) -> JobProgress:
        raise NotImplementedError

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> bytes:
        raise NotImplementedError

    def update_job(self, job_id: JobId, progress: JobProgress, timestamp: int) -> None:
        # NOTE don't forget to pop from sockets in case job is finished
        raise NotImplementedError

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
        raise NotImplementedError
        # NOTE how to delete old results? Probably don't care -- this doesnt have to scale, its demo only
