# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning for LocalProcesses infra spec."""

import logging
import os
import subprocess

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.api import JobSpec, LocalProcesses
from cascade.gateway.spawning.common import allocate_port_range

logger = logging.getLogger(__name__)


def spawn_local(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: LocalProcesses,
) -> subprocess.Popen[bytes]:
    base = ["python", "-m", "cascade.main", "local"]

    with open(f"/tmp/{job_id}.json", "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    base += ["--instance", f"/tmp/{job_id}.json"]

    infra_args = ["--workers_per_host", f"{infra.workers_per_host}", "--hosts", f"{infra.hosts}"]
    report = ["--report_address", f"{addr},{job_id}"]
    logs = ["--loggingConfigSer", loggingConfig.withContext(f"job_{job_id}").ser_cliparam()]
    port_base = allocate_port_range(1 + infra.hosts * infra.workers_per_host * 10)
    return subprocess.Popen(
        base + infra_args + report + ["--port_base", str(port_base)] + logs,
        env={**os.environ, **job_spec.envvars},
        close_fds=True,
    )
