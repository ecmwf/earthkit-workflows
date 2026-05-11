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
 - launching a Python module inside that venv
 - cleanup/termination of both the process and the venv directory
 - RunnerContext: the shared per-executor context passed to all workers via shared memory
 - WorkerSetup: the per-worker identity passed via environment variable
 - save/load helpers for the shared RunnerContext in POSIX shared memory
"""

import importlib
import logging
import multiprocessing as mp
import multiprocessing.resource_tracker as resource_tracker
import os
import runpy
import site
import subprocess
import sys
import sysconfig
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import cloudpickle
import orjson
from packaging.version import Version
from typing_extensions import Self

import cascade.executor.platform as platform
from cascade.deployment.logging import LoggingConfig
from cascade.executor.msg import BackboneAddress
from cascade.executor.runner.packages import _parse_pip_install, check_run_result, initial_venv_packages, run_command
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


class WorkerProcessHandle(ABC):
    @property
    @abstractmethod
    def pid(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def wait(self, timeout: float | None = None) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def kill(self) -> None:
        raise NotImplementedError


@dataclass
class PopenWorkerProcessHandle(WorkerProcessHandle):
    process: subprocess.Popen[bytes]

    @property
    def pid(self) -> int:
        assert self.process.pid is not None
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        return self.process.wait(timeout=timeout)

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()


@dataclass
class MpWorkerProcessHandle(WorkerProcessHandle):
    process: BaseProcess

    @property
    def pid(self) -> int:
        assert self.process.pid is not None
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.exitcode if not self.process.is_alive() else None

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def wait(self, timeout: float | None = None) -> int | None:
        self.process.join(timeout)
        return self.process.exitcode if not self.process.is_alive() else None

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()


@dataclass(frozen=True)
class WorkerSetup:
    """Per-worker identity: together with RunnerContext below, forms complete worker startup config"""

    workerId: WorkerId
    workerAttemptCnt: int
    shm_key: str
    # Packages installed in the worker venv at creation time, as {dist_name: version_str}.
    # Pre-populates PackagesEnv._installed so it can skip pip for already-present packages.
    initial_installed: dict[str, str]

    def to_str(self) -> str:
        installed_json = orjson.dumps(self.initial_installed).decode()
        return f"{repr(self.workerId)}|{self.workerAttemptCnt}|{self.shm_key}|{installed_json}"

    @classmethod
    def from_str(cls, s: str) -> Self:
        worker_repr, attempt_str, shm_key, installed_json = s.split("|", 3)
        return cls(
            workerId=WorkerId.from_repr(worker_repr),
            workerAttemptCnt=int(attempt_str),
            shm_key=shm_key,
            initial_installed=orjson.loads(installed_json),
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
        # this should not happen thanks to uuid, but if it does we handle gracefully
        logger.error(f"runner ctx shm {key!r} already existed; deleting and recreating")
        _old = SharedMemory(key, create=False, **_shm_kwargs)
        _old.close()
        _old.unlink()
        if sys.platform == "darwin":
            time.sleep(1)  # on mac, create right after unlink leads to not found
        shm = SharedMemory(key, create=True, size=size, **_shm_kwargs)
    if _is_unregister:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    assert shm.buf is not None
    shm.buf[:size] = data
    return shm


def load_runner_ctx_from_shm(key: str) -> RunnerContext:
    """Open the shared memory block named key, deserialize the RunnerContext, and close the block.

    The worker calls this once during init; the memory stays alive until the executor frees it.
    """
    shm = SharedMemory(key, create=False, **_shm_kwargs)
    if _is_unregister:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    assert shm.buf is not None
    data = bytes(shm.buf[: shm.size])
    shm.close()
    return cloudpickle.loads(data)


def create_venv() -> tuple[tempfile.TemporaryDirectory[str], dict[str, str]]:
    """Creates a new temporary venv with earthkit-workflows installed at the same version as the parent process.

    Returns the TemporaryDirectory for the venv and a {dist_name: version_str} dict of every
    package pip reported as installed, so callers can pre-populate PackagesEnv._installed.
    """
    td = tempfile.TemporaryDirectory(prefix="cascade_worker_venv_", dir=venv_root)
    logger.debug(f"creating a new worker venv at {td}")
    run_command(["uv", "venv", "--python", _python_version, td.name], check_run_result)
    python = _venv_python(td.name)
    installed: dict[str, str] = {}
    for install_spec in initial_venv_packages():
        logger.debug(f"installing {install_spec} into worker venv")
        result = run_command(
            ["uv", "pip", "install", "--python", python, install_spec],
            check_run_result,
        )
        for dist_name, version in _parse_pip_install(result.stderr).items():
            installed[dist_name] = str(version)
    return td, installed


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "bin", "python")


def _venv_site_paths(venv_dir: str) -> list[str]:
    paths = sysconfig.get_paths(vars={"base": venv_dir, "platbase": venv_dir})
    result: list[str] = []
    for key in ("purelib", "platlib"):
        path = paths.get(key)
        if path and path not in result:
            result.append(path)
    return result


def _build_worker_env(venv_dir: str, envvars: dict[str, str]) -> dict[str, str]:
    python = _venv_python(venv_dir)
    env = {**os.environ, **envvars, "VIRTUAL_ENV": venv_dir}
    env["PATH"] = f"{os.path.dirname(python)}{os.pathsep}{env.get('PATH', '')}"
    return env


def _activate_worker_venv(venv_dir: str, envvars: dict[str, str]) -> None:
    env = _build_worker_env(venv_dir, envvars)
    os.environ.clear()
    os.environ.update(env)
    sys.executable = _venv_python(venv_dir)
    sys.prefix = venv_dir
    sys.exec_prefix = venv_dir
    for site_path in reversed(_venv_site_paths(venv_dir)):
        site.addsitedir(site_path)
        while site_path in sys.path:
            sys.path.remove(site_path)
        sys.path.insert(0, site_path)
    importlib.invalidate_caches()


def _launch_worker_module(module: str, venv_dir: str, envvars: dict[str, str]) -> None:
    _activate_worker_venv(venv_dir, envvars)
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def _launch_via_popen(module: str, venv_dir: str, envvars: dict[str, str]) -> WorkerProcessHandle:
    python = _venv_python(venv_dir)
    env = _build_worker_env(venv_dir, envvars)
    logger.debug(f"launching {module} in {venv_dir} via popen")
    try:
        process = subprocess.Popen([python, "-m", module], env=env)
    except OSError as e:
        raise CascadeInternalError(f"failed to launch worker process (env may be too large): {repr(e)}", parent=e) from e
    return PopenWorkerProcessHandle(process)


def _launch_via_multiprocessing(module: str, venv_dir: str, envvars: dict[str, str]) -> WorkerProcessHandle:
    ctx = platform.get_mp_ctx("worker")
    process = ctx.Process(target=_launch_worker_module, args=(module, venv_dir, envvars))
    logger.debug(f"launching {module} in {venv_dir} via multiprocessing")
    process.start()
    return MpWorkerProcessHandle(process)


def launch_in_venv(module: str, venv_dir: str, envvars: dict[str, str]) -> WorkerProcessHandle:
    """Launches `python -m module` inside the given venv with the provided environment variables."""
    method = platform.get_new_worker_method()
    if method == "popen":
        return _launch_via_popen(module, venv_dir, envvars)
    return _launch_via_multiprocessing(module, venv_dir, envvars)


def terminate_worker(process: WorkerProcessHandle, venv_dir: tempfile.TemporaryDirectory[str]) -> None:
    """Terminates the worker process and cleans up its venv directory."""
    if process.is_alive():
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            logger.debug("worker did not exit cleanly during graceful shutdown wait", exc_info=True)
        if process.is_alive():
            process.kill()
        try:
            process.wait()
        except Exception:
            logger.debug("worker wait after shutdown raised", exc_info=True)
    try:
        venv_dir.cleanup()
    except Exception as e:
        logger.warning(f"failed to cleanup worker venv: {repr(e)}")
