from dataclasses import dataclass

from pydantic import BaseModel

from cascade.controller.report import JobId, JobProgress
from cascade.low.core import DatasetId

CascadeGatewayAPI = BaseModel


@dataclass
class JobSpec:
    # job -- atm its "catalog" approach, alternatively we just job: JobInstance
    benchmark_name: str
    envvars: dict[str, str]
    # example values:
    # benchmark_name="generators"
    # envvars={"GENERATORS_N": "8", "GENERATORS_K": "10", "GENERATORS_L": "4"}

    # infra
    workers_per_host: int
    hosts: int
    use_slurm: bool


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
