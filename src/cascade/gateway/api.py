from pydantic import BaseModel

from cascade.low.core import DatasetId, Worker, JobInstance
from cascade.controller.report import JobProgress, JobId

CascadeGatewayAPI = BaseModel


class SubmitJobRequest(CascadeGatewayAPI):
    resources: list[Worker]
    job: JobInstance


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
