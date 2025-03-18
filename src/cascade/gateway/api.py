from dataclasses import dataclass
from pydantic import BaseModel

from cascade.low.core import DatasetId, Worker
from cascade.controller.report import JobProgress, JobId

CascadeGatewayAPI = BaseModel

@dataclass
class JobSpec:
    benchmark_name: str
    # job: JobInstance
    workers_per_host: int
    hosts: int

class SubmitJobRequest(CascadeGatewayAPI):
    job: JobSpec

class SubmitJobResponse(CascadeGatewayAPI):
    job_id: JobId | None
    error: str | None


class JobProgressRequest(CascadeGatewayAPI):
    job_id: JobId


class JobProgressResponse(CascadeGatewayAPI):
    progress: JobProgress | None
    error: str | None


class ResultRetrievalRequest(CascadeGatewayAPI):
    job_id: JobId
    dataset_id: DatasetId


class ResultRetrievalResponse(CascadeGatewayAPI):
    result: bytes | None
    error: str | None
