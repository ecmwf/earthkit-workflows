# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning new processes for each SubmitJobRequest, locally or remotely."""

import logging
import subprocess

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.api import JobSpec, LocalProcesses, SlurmCluster, SshCluster
from cascade.gateway.spawning.local import spawn_local
from cascade.gateway.spawning.slurm import spawn_slurm
from cascade.gateway.spawning.ssh import spawn_ssh
from cascade.gateway.spawning.troika import spawn_troika_singlehost
from cascade.gateway.spawning.wheels import EkwInstallSpec, prepare_install_spec  # re-exported
from cascade.low.exceptions import CascadeUserError

logger = logging.getLogger(__name__)

__all__ = ["EkwInstallSpec", "prepare_install_spec", "spawn_subprocess"]


def spawn_subprocess(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    troika_config: str | None,
    shared_path: str | None,
    install_spec: EkwInstallSpec | None,
) -> subprocess.Popen[bytes]:
    infra = job_spec.infra_spec
    if isinstance(infra, SlurmCluster):
        if infra.troika is not None:
            # TODO support logging config properly
            if troika_config is None:
                raise CascadeUserError("cant spawn troika job without troika config")
            return spawn_troika_singlehost(job_spec, addr, job_id, infra, infra.troika, troika_config)
        else:
            # TODO support logging config properly
            return spawn_slurm(job_spec, addr, job_id, loggingConfig, infra, shared_path, install_spec)
    elif isinstance(infra, LocalProcesses):
        return spawn_local(job_spec, addr, job_id, loggingConfig, infra)
    elif isinstance(infra, SshCluster):
        return spawn_ssh(job_spec, addr, job_id, loggingConfig, infra, install_spec)
    else:
        raise CascadeUserError(f"unsupported infra_spec type: {type(infra)}")
