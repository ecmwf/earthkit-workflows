# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import pytest

import cascade.executor.platform as platform
import cascade.executor.runner.setup as setup


def test_get_new_worker_method_defaults_by_platform(monkeypatch) -> None:
    monkeypatch.setattr(platform.sys, "platform", "darwin")
    monkeypatch.delenv("CASCADE_NEW_WORKER_METHOD", raising=False)
    assert platform.get_new_worker_method() == "multiprocessing"
    monkeypatch.setattr(platform.sys, "platform", "linux")
    assert platform.get_new_worker_method() == "multiprocessing"
    monkeypatch.setattr(platform.sys, "platform", "plan9")
    assert platform.get_new_worker_method() == "popen"
    monkeypatch.setenv("CASCADE_NEW_WORKER_METHOD", "popen")
    assert platform.get_new_worker_method() == "popen"
    monkeypatch.setenv("CASCADE_NEW_WORKER_METHOD", "multiprocessing")
    assert platform.get_new_worker_method() == "multiprocessing"
    monkeypatch.setenv("CASCADE_NEW_WORKER_METHOD", "invalid")
    with pytest.raises(ValueError):
        platform.get_new_worker_method()


@dataclass
class _FakePopen:
    args: list[str]
    env: dict[str, str]
    alive: bool = True
    killed: bool = False
    terminated: bool = False

    @property
    def pid(self) -> int:
        return 1234

    def poll(self) -> int | None:
        return None if self.alive else 0

    def is_alive(self) -> bool:
        return self.alive

    def wait(self, timeout: float | None = None) -> int:
        self.alive = False
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False


def test_launch_in_venv_popen(monkeypatch) -> None:
    captured_args: list[str] | None = None
    captured_env: dict[str, str] | None = None

    def fake_popen(args, env):  # type: ignore[no-untyped-def]
        nonlocal captured_args, captured_env
        captured_args = list(args)
        captured_env = dict(env)
        return _FakePopen(args=list(args), env=dict(env))

    monkeypatch.setattr(setup.subprocess, "Popen", fake_popen)

    monkeypatch.setenv("CASCADE_NEW_WORKER_METHOD", "popen")
    handle = setup.launch_in_venv("demo.module", "/tmp/demo-venv", {"X_TEST": "1"})

    assert isinstance(handle, setup.PopenWorkerProcessHandle)
    assert captured_args == ["/tmp/demo-venv/bin/python", "-m", "demo.module"]
    assert captured_env is not None
    assert captured_env["VIRTUAL_ENV"] == "/tmp/demo-venv"
    assert captured_env["X_TEST"] == "1"
    assert captured_env["PATH"].startswith("/tmp/demo-venv/bin")
    assert handle.pid == 1234


class _FakeProcess:
    def __init__(self) -> None:
        self.started = False
        self.alive = True
        self.exitcode = None
        self.pid = 4321
        self.terminated = False
        self.killed = False
        self.target: object | None = None
        self.args: tuple[object, ...] | None = None

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.alive = False
        if self.exitcode is None:
            self.exitcode = 0

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9


def test_launch_in_venv_multiprocessing(monkeypatch) -> None:
    fake_process = _FakeProcess()

    class _FakeContext:
        def Process(self, target, args):  # type: ignore[no-untyped-def]
            fake_process.target = target
            fake_process.args = tuple(args)
            return fake_process

    monkeypatch.setattr(setup.mp, "get_context", lambda method: _FakeContext())

    monkeypatch.setenv("CASCADE_NEW_WORKER_METHOD", "multiprocessing")
    handle = setup.launch_in_venv("demo.module", "/tmp/demo-venv", {"X_TEST": "1"})

    assert isinstance(handle, setup.MpWorkerProcessHandle)
    assert fake_process.started
    assert fake_process.args == ("demo.module", "/tmp/demo-venv", {"X_TEST": "1"})
    assert fake_process.target is setup._launch_worker_module
    assert handle.pid == 4321
    assert handle.poll() is None
    assert handle.is_alive()
    handle.terminate()
    assert fake_process.terminated
    assert handle.poll() == -15
    assert handle.wait() == -15


def test_terminate_worker_grace_period(tmp_path) -> None:
    calls: list[tuple[str, float | None]] = []

    @dataclass
    class _TermProc(setup.WorkerProcessHandle):
        alive: bool = True
        terminated: bool = False
        killed: bool = False

        @property
        def pid(self) -> int:
            return 9999

        def poll(self) -> int | None:
            return None if self.alive else 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            calls.append(("terminate", None))
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int | None:
            calls.append(("wait", timeout))
            if timeout is not None:
                raise TimeoutError
            self.alive = False
            return 0

        def kill(self) -> None:
            calls.append(("kill", None))
            self.killed = True
            self.alive = False

    proc = _TermProc()
    venv_dir = setup.tempfile.TemporaryDirectory(dir=tmp_path)

    setup.terminate_worker(proc, venv_dir)

    assert calls == [("terminate", None), ("wait", 5), ("kill", None), ("wait", None)]
