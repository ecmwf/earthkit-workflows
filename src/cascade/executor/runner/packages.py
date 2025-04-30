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

import glob
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from typing import Literal

logger = logging.getLogger(__name__)


def _find_site_packages(venv_dir: str) -> str:
    """Find the site-packages directory in a virtual environment."""
    site_packages_pattern = os.path.join(venv_dir, "lib", "python*", "site-packages")
    site_packages_dirs = glob.glob(site_packages_pattern)
    if site_packages_dirs:
        return site_packages_dirs[0]
    raise ValueError(f"Could not find site-packages in {venv_dir}")


class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory | None = None

    def extend(self, packages: list[str]) -> None:
        if not packages:
            return

        python_version = None

        if any(map(lambda p: "python" in p, packages)):
            python_version = [p.split("==", 2)[1] for p in packages if "python" in p][0]
            packages = [p for p in packages if "python" not in p]

        if self.td is None:
            logger.debug("creating a new venv")
            self.td = tempfile.TemporaryDirectory()
            if python_version:
                venv_command = [
                    "uv",
                    "venv",
                    self.td.name,
                    "--python",
                    f"{python_version}",
                ]
            else:
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
        sys.path = [_find_site_packages(self.td.name), *sys.path]

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self.td is not None:
            self.td.cleanup()
        return False
