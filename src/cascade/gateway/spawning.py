# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning new processes for each SubmitJobRequest, locally or remotely"""

import base64
import itertools
import logging
import os
import stat
import subprocess

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.api import JobSpec, TroikaSpec

logger = logging.getLogger(__name__)

# TODO this is a hotfix to not port collide on local jobs. There should be way more
# bind-to-random-port overall, but the current code often needs to use the port number
# before the actual bind happens -- this should be inverted
local_job_port = 12345


def _spawn_troika_singlehost(
    job_spec: JobSpec, addr: str, job_id: JobId, troika: TroikaSpec, troika_config: str
) -> subprocess.Popen:
    script = "#!/bin/bash\n"
    script += f"source {troika.venv}\n"
    for k, v in job_spec.envvars.items():
        script += f"export {k}={v}\n"

    job_desc_raw = orjson.dumps(job_spec.job_instance.model_dump())
    job_desc_enc = base64.b64encode(job_desc_raw).decode("ascii")
    script += f'JOB_ENC="{job_desc_enc}"'
    job_json_path = f"/tmp/cascJob.{job_id}.json"
    script += f'echo "$JOB_ENC" | base64 --decode > {job_json_path}'
    script += "python -m cascade.main local"
    script += f" --instance {job_json_path}"

    script += (
        f" --workers_per_host {job_spec.workers_per_host} --hosts {job_spec.hosts}"
    )
    script += f" --report_address {addr},{job_id}"
    # NOTE technically not needed to be globally unique, but we cant rely on troika environment isolation...
    global local_job_port
    script += f" --port_base {local_job_port}"
    local_job_port += 1 + job_spec.hosts * job_spec.workers_per_host * 10
    script += "\n"
    script_path = f"/tmp/troikascade.{job_id}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(
        script_path,
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH,
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


def _spawn_local(
    job_spec: JobSpec, addr: str, job_id: JobId, loggingConfig: LoggingConfig,
) -> subprocess.Popen:
    base = [
        "python",
        "-m",
        "cascade.main",
        "local",
    ]

    with open(f"/tmp/{job_id}.json", "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    base += ["--instance", f"/tmp/{job_id}.json"]

    infra = [
        "--workers_per_host",
        f"{job_spec.workers_per_host}",
        "--hosts",
        f"{job_spec.hosts}",
    ]
    report = ["--report_address", f"{addr},{job_id}"]
    logs = ["--loggingConfigSer", loggingConfig.withContext(f"job_{job_id}").ser_cliparam()]
    global local_job_port
    portBase = ["--port_base", str(local_job_port)]
    local_job_port += 1 + job_spec.hosts * job_spec.workers_per_host * 10
    return subprocess.Popen(
        base + infra + report + portBase + logs, env={**os.environ, **job_spec.envvars}
    )


def _spawn_slurm(job_spec: JobSpec, addr: str, job_id: JobId) -> subprocess.Popen:
    extra_vars = {
        "EXECUTOR_HOSTS": str(job_spec.hosts),
        "WORKERS_PER_HOST": str(job_spec.workers_per_host),
        # NOTE put to infra specs
        "SHM_VOL_GB": "64",
        "REPORT_ADDRESS": f"{addr},{job_id}",
    }
    with open(f"./localConfigs/_tmp/{job_id}.json", "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    extra_vars["INSTANCE"] = f"./localConfigs/_tmp/{job_id}.json"
    subprocess.run(
        ["cp", "localConfigs/gateway.sh", f"localConfigs/_tmp/{job_id}"], check=True
    )
    with open(f"./localConfigs/_tmp/{job_id}", "a") as f:
        for k, v in itertools.chain(job_spec.envvars.items(), extra_vars.items()):
            f.write(f"export {k}={v}\n")
    return subprocess.Popen(
        ["./scripts/launch_slurm.sh", f"localConfigs/_tmp/{job_id}"]
    )


def spawn_subprocess(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    troika_config: str | None,
) -> subprocess.Popen:
    if job_spec.troika is not None:
        # TODO support logging config properly
        if troika_config is None:
            raise ValueError("cant spawn troika job without troika config")
        if not job_spec.use_slurm:
            return _spawn_troika_singlehost(
                job_spec, addr, job_id, job_spec.troika, troika_config
            )
        else:
            # TODO create a slurm script like in spawn_slurm, but dont refer to any other file
            raise NotImplementedError

    elif job_spec.use_slurm:
        # TODO support logging config properly
        return _spawn_slurm(job_spec, addr, job_id)
    else:
        return _spawn_local(job_spec, addr, job_id, loggingConfig)
