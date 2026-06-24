# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning for Troika (Slurm via Troika) infra spec."""

import logging
import os
import stat
import subprocess

import orjson

from cascade.controller.report import JobId
from cascade.gateway.api import JobSpec, SlurmCluster, TroikaSpec
from cascade.gateway.spawning.common import allocate_port_range

logger = logging.getLogger(__name__)


def spawn_troika_singlehost(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    infra: SlurmCluster,
    troika: TroikaSpec,
    troika_config: str,
) -> subprocess.Popen[bytes]:
    script = "#!/bin/bash\n"
    script += f"source {troika.venv}\n"
    for k, v in job_spec.envvars.items():
        script += f"export {k}={v}\n"

    job_json_path = f"/tmp/cascJob.{job_id}.json"
    with open(job_json_path, "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    script += f"python -m cascade.main local"
    script += f" --instance {job_json_path}"

    script += f" --workers_per_host {infra.workers_per_host} --hosts {infra.hosts}"
    script += f" --report_address {addr},{job_id}"
    # NOTE technically not needed to be globally unique, but we cant rely on troika environment isolation...
    port_base = allocate_port_range(1 + infra.hosts * infra.workers_per_host * 10)
    script += f" --port_base {port_base}"
    script += "\n"
    script_path = f"/tmp/troikascade.{job_id}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(
        script_path,
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    return subprocess.Popen(
        [
            "troika",
            "-c",
            troika_config,
            "submit",
            "-o",
            f"/tmp/output.{job_id}.txt",
            troika.conn,
            script_path,
        ]
    )
