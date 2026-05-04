# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Worker venv creation and process launching, plus RunnerContext/WorkerSetup definitions.

Each worker owns its own temporary venv. This module handles:
 - creation of a temporary venv with the same Python version and an initial earthkit-workflows install
 - launching a Python module as a subprocess inside that venv
 - cleanup/termination of both the process and the venv directory
 - RunnerContext: the shared per-executor context passed to all workers via shared memory
 - WorkerSetup: the per-worker identity passed via environment variable
 - save/load helpers for the shared RunnerContext in POSIX shared memory
"""

import logging
import multiprocessing.resource_tracker
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import cloudpickle
from typing_extensions import Self

from cascade.deployment.logging import LoggingConfig
from cascade.executor.msg import BackboneAddress
from cascade.executor.runner.packages import check_run_result, initial_venv_packages, run_command
from cascade.executor.runner.runner import ExecutionContext
from cascade.low.core import DatasetId, JobInstance, TaskId, WorkerId
from cascade.low.exceptions import CascadeInternalError

logger = logging.getLogger(__name__)

WORKER_SETUP_ENVVAR = "CASCADE_WORKER_SETUP"
# NOTE on some systems, default /tmp can be mounted with noexec, leading to issues at runtime
# like 'failed to map segment from shared object' for binary dependencies like zmq
# Thus override this envvar to some exec-mounted filesystem
venv_root = os.environ.get("CASCADE_VENV_ROOT", None)

_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Shared memory unregistering pattern: Python's resource tracker automatically unlinks
# shared memory it created, which is wrong for memory we share across processes.
# On 3.13+ we can disable tracking at creation time; on older versions we unregister manually.
if (sys.version_info.major, sys.version_info.minor) >= (3, 13):
    _is_unregister = False
    _shm_kwargs: dict[str, Any] = {"track": False}
else:
    _is_unregister = True
    _shm_kwargs = {}


@dataclass(frozen=True)
class WorkerSetup:
    """Per-worker identity: what is factored out of RunnerContext and passed via envvar."""

    workerId: WorkerId
    workerAttemptCnt: int
    shm_key: str

    def to_str(self) -> str:
        return f"{repr(self.workerId)}|{self.workerAttemptCnt}|{self.shm_key}"

    @classmethod
    def from_str(cls, s: str) -> Self:
        worker_repr, attempt_str, shm_key = s.split("|", 2)
        return cls(
            workerId=WorkerId.from_repr(worker_repr),
            workerAttemptCnt=int(attempt_str),
            shm_key=shm_key,
        )


@dataclass(frozen=True, slots=True)
class RunnerContext:
    """The static runner configuration shared across all workers on an executor."""

    job: JobInstance
    callback: BackboneAddress
    param_source: dict[TaskId, dict[int | str, DatasetId]]
    loggingConfig: LoggingConfig
    schema_lookup: dict[DatasetId, str]

    @staticmethod
    def build_schema_lookup(job: JobInstance) -> dict[DatasetId, str]:
        return {
            DatasetId(task_id, output): fqn
            for task_id, task_instance in job.tasks.items()
            for output, fqn in task_instance.definition.output_schema
        }

    def project(self, taskSequence: Any) -> ExecutionContext:
        param_source_ext: dict[TaskId, dict[int | str, tuple[DatasetId, str]]] = {
            task: {k: (dataset_id, self.schema_lookup[dataset_id]) for k, dataset_id in self.param_source[task].items()}
            for task in taskSequence.tasks
        }
        return ExecutionContext(
            tasks={task: self.job.tasks[task] for task in taskSequence.tasks},
            param_source=param_source_ext,
            callback=self.callback,
            publish=taskSequence.publish,
        )


def save_runner_ctx_to_shm(ctx: RunnerContext, key: str) -> SharedMemory:
    """Cloudpickle ctx and store it in a new POSIX shared memory block named key.

    Returns the SharedMemory object; the caller owns it and must call .close()/.unlink() on exit.
    If a stale block with the same name exists, it is deleted and recreated.
    """
    data = cloudpickle.dumps(ctx)
    size = len(data)
    try:
        shm = SharedMemory(key, create=True, size=size, **_shm_kwargs)
    except FileExistsError:
        logger.error(f"runner ctx shm {key!r} already existed; deleting and recreating")
        _old = SharedMemory(key, create=False, **_shm_kwargs)
        _old.close()
        _old.unlink()
        shm = SharedMemory(key, create=True, size=size, **_shm_kwargs)
    if _is_unregister:
        multiprocessing.resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    assert shm.buf is not None
    shm.buf[:size] = data
    return shm


def load_runner_ctx_from_shm(key: str) -> RunnerContext:
    """Open the shared memory block named key, deserialize the RunnerContext, and close the block.

    The worker calls this once during init; the memory stays alive until the executor frees it.
    """
    shm = SharedMemory(key, create=False, **_shm_kwargs)
    if _is_unregister:
        multiprocessing.resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    assert shm.buf is not None
    data = bytes(shm.buf[: shm.size])
    shm.close()
    return cloudpickle.loads(data)


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
    try:
        return subprocess.Popen([python, "-m", module], env=env)
    except OSError as e:
        raise CascadeInternalError(f"failed to launch worker process (env may be too large): {repr(e)}", parent=e) from e


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
