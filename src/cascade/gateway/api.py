# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
from dataclasses import dataclass, field
from typing import Any, Union

import cloudpickle

from cascade.controller.report import JobId, JobProgress
from cascade.low.core import DatasetId, JobInstance, JobInstanceRich, TaskId
from cascade.low.exceptions import CascadeInternalError
from cascade.low.func import CascadeBaseModel

CascadeGatewayAPI = CascadeBaseModel


@dataclass
class TroikaSpec:
    """Requires the gateway to have been started with --troika_config pointing
    to some config.yml troika file. The connection must work (passwordlessly),
    and must allow for script being copied. The remote host must have a venv
    already in place, and must be able to resolve gateway's fqdn
    """

    venv: str  # remote host path to venv -- *do* include the bin/activate
    conn: str  # which connection from config.yml to pick


@dataclass
class LocalProcesses:
    """Run controller and executors as local subprocesses on the gateway host."""

    workers_per_host: int
    hosts: int


@dataclass
class SlurmCluster:
    """Launch via Slurm (or troika wrapping Slurm) from the gateway host."""

    workers_per_host: int
    hosts: int
    troika: TroikaSpec | None = None
    wheel_path: str | None = None


@dataclass
class SshCluster:
    """Launch controller and executors on remote nodes via SSH.

    Each URL is of the form [<username>@]<hostname>. The hostnames must be
    SSH-accessible from the gateway and reachable among each other for ZMQ
    communications (controller <-> executors).

    The earthkit-workflows package is installed on the fly via ``uv run --with``.
    """

    controller_url: str  # e.g. "root@plain-cluster-main"
    worker_urls: list[str]  # e.g. ["root@plain-cluster-worker1", "root@plain-cluster-worker2"]
    workers_per_host: int
    ssh_key_path: str | None = None
    ssh_config_path: str | None = None
    wheel_path: str | None = None


InfraSpec = Union[LocalProcesses, SlurmCluster, SshCluster]


@dataclass
class JobSpec:
    job_instance: JobInstanceRich
    envvars: dict[str, str]
    infra_spec: InfraSpec


class SubmitJobRequest(CascadeGatewayAPI):
    job: JobSpec


class SubmitJobResponse(CascadeGatewayAPI):
    job_id: JobId | None
    error: str | None


class JobProgressRequest(CascadeGatewayAPI):
    job_ids: list[JobId]  # on empty list, return all
    detailed_report: bool = False


class JobProgressResponse(CascadeGatewayAPI):
    progresses: dict[JobId, JobProgress | None]
    datasets: dict[JobId, list[DatasetId]]
    queue_length: int
    error: str | None  # top level error
    completed_task_ids: dict[JobId, list[TaskId]] | None = None
    planned_task_ids: dict[JobId, list[TaskId]] | None = None


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
        # we are in control of encoding -> InternalError
        raise CascadeInternalError(f"result retrieval failed: {result.error}")
    as_bytes = base64.b64decode(result.result)
    as_value = cloudpickle.loads(as_bytes)
    return as_value
