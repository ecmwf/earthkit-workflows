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


class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory | None = None

    def extend(self, packages: list[str]) -> None:
        if not packages:
            return
        if self.td is None:
            logger.debug("creating a new venv")
            self.td = tempfile.TemporaryDirectory()
            venv_command = ["uv", "venv", self.td.name]
            # NOTE we create a venv instead of just plain directory, because some of the packages create files
            # outside of site-packages. Thus we then install with --prefix, not with --target
            subprocess.run(venv_command, check=True)

        logger.debug(
            f"installing {len(packages)} packages: {','.join(packages[:3])}{',...' if len(packages) > 3 else ''}"
        )
        install_command = ["uv", "pip", "install", "--prefix", self.td.name]
        if os.environ.get("VENV_OFFLINE", "") == "YES":
            install_command += ["--offline"]
        if cache_dir := os.environ.get("VENV_CACHE", ""):
            install_command += ["--cache-dir", cache_dir]
        install_command.extend(set(packages))
        subprocess.run(install_command, check=True)
        # NOTE not sure if getsitepackages was intended for this -- if issues, attempt replacing
        # with something like f"{self.td.name}/lib/python*/site-packages" + globbing
        extra_sp = site.getsitepackages(prefixes=[self.td.name])
        # NOTE this makes the explicit packages go first, in case of a different version
        sys.path = extra_sp + sys.path

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self.td is not None:
            self.td.cleanup()
        return False
