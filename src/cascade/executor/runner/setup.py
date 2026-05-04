# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Worker venv creation and process launching.

Each worker owns its own temporary venv. This module handles:
 - creation of a temporary venv with the same Python version and an initial earthkit-workflows install
 - launching a Python module as a subprocess inside that venv
 - cleanup/termination of both the process and the venv directory
"""

import logging
import os
import subprocess
import sys
import tempfile

from cascade.executor.runner.packages import check_run_result, initial_venv_packages, run_command

logger = logging.getLogger(__name__)

CONTEXT_ENVVAR = "CASCADE_RUNNER_CONTEXT"
# NOTE on some systems, default /tmp can be mounted with noexec, leading to issues at runtime
# like 'failed to map segment from shared object' for binary dependencies like zmq
# Thus override this envvar to some exec-mounted filesystem
venv_root = os.environ.get("CASCADE_VENV_ROOT", None)

_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def create_venv() -> tempfile.TemporaryDirectory:  # type: ignore[type-arg]
    """Creates a new temporary venv with earthkit-workflows installed at the same version as the parent process."""
    td = tempfile.TemporaryDirectory(prefix="cascade_worker_venv_", dir=venv_root)
    logger.debug(f"creating a new worker venv at {td}")
    run_command(["uv", "venv", "--python", _python_version, td.name], check_run_result)
    python = _venv_python(td.name)
    for install_spec in initial_venv_packages():
        logger.debug(f"installing {install_spec} into worker venv")
        run_command(
            ["uv", "pip", "install", "--python", python, install_spec],
            check_run_result,
        )
    return td


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "bin", "python")


def launch_in_venv(module: str, venv_dir: str, envvars: dict[str, str]) -> "subprocess.Popen[bytes]":
    """Launches `python -m module` inside the given venv with the provided environment variables.

    The parent process's non-venv sys.path entries are forwarded via PYTHONPATH so that
    cloudpickle-serialized callables referencing caller-side modules (e.g. test modules,
    editable source trees) can be unpickled in the worker.
    """
    python = _venv_python(venv_dir)
    # Forward non-venv paths so cloudpickle can resolve module references from the parent
    parent_venv = sys.prefix
    extra_paths = [p for p in sys.path if p and not p.startswith(parent_venv)]
    pythonpath = os.pathsep.join(extra_paths)
    env = {**os.environ, **envvars, "VIRTUAL_ENV": venv_dir}
    if pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{existing}" if existing else pythonpath
    logger.debug(f"launching {module} in {venv_dir}")
    return subprocess.Popen([python, "-m", module], env=env)


def terminate_worker(process: subprocess.Popen[bytes], venv_dir: tempfile.TemporaryDirectory[str]) -> None:
    """Terminates the worker process and cleans up its venv directory."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    try:
        venv_dir.cleanup()
    except Exception as e:
        logger.warning(f"failed to cleanup worker venv: {repr(e)}")
