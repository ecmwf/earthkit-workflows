from dataclasses import dataclass

from pydantic import BaseModel

from cascade.controller.report import JobId, JobProgress
from cascade.low.core import DatasetId, JobInstance

CascadeGatewayAPI = BaseModel


@dataclass
class JobSpec:
    # job benchmark + envvars -- set to None/{} if using custom jobs instead
    benchmark_name: str | None
    envvars: dict[str, str]
    # example values:
    # benchmark_name="generators"
    # envvars={"GENERATORS_N": "8", "GENERATORS_K": "10", "GENERATORS_L": "4"}
    job_instance: JobInstance | None

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
    job_ids: list[JobId]  # on empty list, return all


class JobProgressResponse(CascadeGatewayAPI):
    progresses: dict[JobId, JobProgress]
    error: str | None  # top level error


class ResultRetrievalRequest(CascadeGatewayAPI):
    job_id: JobId
    dataset_id: DatasetId


class ResultRetrievalResponse(CascadeGatewayAPI):
    result: bytes | None
    error: str | None
