# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Extending venv with packages required by the executed job

Note that venv itself is left untouched after the run finishes -- we extend sys path
with a temporary directory and install in there
"""

import logging
import os
import site
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from typing import Literal

logger = logging.getLogger(__name__)


class Commands:
    venv_command = lambda name: ["uv", "venv", name]
    install_command = lambda name: [
        "uv",
        "pip",
        "install",
        "--prefix",
        name,
        "--prerelease",
        "explicit",
    ]


def run_command(command: list[str]) -> None:
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError as ex:
        raise ValueError(f"command failure: {ex}")
    if result.returncode != 0:
        msg = f"command failed with {result.returncode}. Stderr: {result.stderr}, Stdout: {result.stdout}, Args: {result.args}"
        logger.error(msg)
        raise ValueError(msg)


class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "PackagesEnv":
        logger.debug("creating a new venv")
        self.td = tempfile.TemporaryDirectory(prefix="cascade_runner_venv_", delete = False)
        # NOTE we create a venv instead of just plain directory, because some of the packages create files
        # outside of site-packages. Thus we then install with --prefix, not with --target
        run_command(Commands.venv_command(self.td.name))

        # NOTE not sure if getsitepackages was intended for this -- if issues, attempt replacing
        # with something like f"{self.td.name}/lib/python*/site-packages" + globbing
        extra_sp = site.getsitepackages(prefixes=[self.td.name])
        # NOTE this makes the explicit packages go first, in case of a different version
        logger.debug(f"extending sys.path with {extra_sp}")
        sys.path = extra_sp + sys.path
        logger.debug(f"new sys.path: {sys.path}")

        return self
    
    def extend(self, packages: list[str]) -> None:
        if not packages:
            return
        assert self.td is not None         

        logger.debug(
            f"installing {len(packages)} packages: {','.join(packages[:3])}{',...' if len(packages) > 3 else ''}"
        )
        install_command = Commands.install_command(self.td.name)
        if os.environ.get("VENV_OFFLINE", "") == "YES":
            install_command += ["--offline"]
        if cache_dir := os.environ.get("VENV_CACHE", ""):
            install_command += ["--cache-dir", cache_dir]
        install_command.extend(set(packages))
        run_command(install_command)

        self._prepare_env()

    def _prepare_env(self) -> None:
        """Prepare the environment after installing packages"""
        import importlib
        importlib.invalidate_caches()

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        sys.path = [p for p in sys.path if self.td is None or not p.startswith(self.td.name)]
        if self.td is not None:
            self.td.cleanup()
        return False
