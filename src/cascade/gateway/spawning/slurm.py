# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning for SlurmCluster infra spec."""

import logging
import shlex
import stat
import subprocess
import tempfile
from importlib import resources
from pathlib import Path

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.api import JobSpec, SlurmCluster
from cascade.gateway.spawning.common import allocate_port_range
from cascade.gateway.spawning.wheels import EkwInstallSpec
from cascade.low.exceptions import CascadeUserError

logger = logging.getLogger(__name__)
_SLURM_TEMPLATE_PACKAGE = "cascade.gateway.slurm_templates"


def stage_text_resource(resource_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = resources.files(_SLURM_TEMPLATE_PACKAGE).joinpath(resource_name).read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp_path.replace(dest)


def write_slurm_exports(dest: Path, exports: dict[str, str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in exports.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    body = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def stage_slurm_scripts(shared_path: str) -> Path:
    slurm_root = Path(shared_path) / "cascade-slurm"
    scripts_dir = slurm_root / "scripts"
    stage_text_resource("launch_slurm.sh", scripts_dir / "launch_slurm.sh")
    stage_text_resource("slurm_entrypoint.sh", scripts_dir / "slurm_entrypoint.sh")
    return slurm_root


def spawn_slurm(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: SlurmCluster,
    shared_path: str | None,
    install_spec: EkwInstallSpec | None,
) -> subprocess.Popen[bytes]:
    if shared_path is None:
        raise CascadeUserError("Slurm jobs require gateway shared_path")
    if install_spec is None or install_spec.shared_spec is None:
        raise CascadeUserError("Slurm jobs require a shared-disk install spec; start the gateway with --shared-path")

    install_spec_to_use = install_spec.shared_spec

    slurm_root = stage_slurm_scripts(shared_path)
    job_root = slurm_root / "jobs" / str(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    scripts_dir = slurm_root / "scripts"

    controller_port = allocate_port_range(1 + (infra.hosts + 1) * infra.workers_per_host * 10)

    job_instance_path = job_root / "instance.json"
    with open(job_instance_path, "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))

    logging_ser = loggingConfig.withContext(f"job_{job_id}").ser_cliparam()
    exports = {
        **job_spec.envvars,
        "EXECUTOR_HOSTS": str(infra.hosts),
        "WORKERS_PER_HOST": str(infra.workers_per_host),
        "SHM_VOL_GB": "64",
        "INSTANCE": str(job_instance_path),
        "REPORT_ADDRESS": f"{addr},{job_id}",
        "LOGGING_CONFIG_SER": logging_ser,
        "UV_RUN_WITH": install_spec_to_use,
        "CONTROLLER_PORT": str(controller_port),
        "JOB_ROOT": str(job_root),
    }
    config_path = job_root / "config.sh"
    write_slurm_exports(config_path, exports)

    launcher = scripts_dir / "launch_slurm.sh"
    return subprocess.Popen([str(launcher), str(config_path)])
