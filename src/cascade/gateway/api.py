# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
from dataclasses import dataclass
from typing import Any

import cloudpickle
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
    datasets: dict[JobId, list[DatasetId]]
    error: str | None  # top level error


class ResultRetrievalRequest(CascadeGatewayAPI):
    job_id: JobId
    dataset_id: DatasetId


class ResultRetrievalResponse(CascadeGatewayAPI):
    result: str | None
    error: str | None


class ResultDeletionRequest(CascadeGatewayAPI):
    datasets: dict[JobId, list[DatasetId]]
    # empty dict: delete all datasets present at the moment
    # empty list for jobId: delete all for that job (present at the moment)


class ResultDeletionResponse(CascadeGatewayAPI):
    error: str | None


class ShutdownRequest(CascadeGatewayAPI):
    pass


class ShutdownResponse(CascadeGatewayAPI):
    error: str | None


def decoded_result(result: ResultRetrievalResponse, job: JobInstance) -> Any:
    # TODO dont base64, instead skip the whole json business and send two zmq frames
    # TODO dont cloudpickle, instead use the JobInstance's registered serde
    if not result.result:
        raise ValueError(result.error)
    as_bytes = base64.b64decode(result.result)
    as_value = cloudpickle.loads(as_bytes)
    return as_value
