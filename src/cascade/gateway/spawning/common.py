# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared utilities for gateway spawning submodules."""

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SpawnedJob:
    """Local handles of whatever was spawned for a job. Termination signals go to `procs`; if `pgid` is
    set, the whole process group is additionally killed if anything remains after the termination grace.

    NOTE for remote spawns (ssh, troika, ...), the procs are just the local clients, so terminating them
    does not necessarily terminate the remote processes.
    """

    procs: list[subprocess.Popen[bytes]]
    pgid: int | None = None


# TODO this is a hotfix to not port collide on local jobs. There should be way more
# bind-to-random-port overall, but the current code often needs to use the port number
# before the actual bind happens -- this should be inverted
_local_job_port = 12345


def allocate_port_range(size: int) -> int:
    """Allocate a contiguous range of ports; return the base port."""
    global _local_job_port
    base = _local_job_port
    _local_job_port += size
    return base


def ssh_args(ssh_key_path: str | None, ssh_config_path: str | None) -> list[str]:
    """Common SSH/SCP flags: disable host-key prompts, use key if provided."""
    args = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key_path is not None:
        args += ["-i", ssh_key_path]
    if ssh_config_path is not None:
        args += ["-F", ssh_config_path]
    return args
