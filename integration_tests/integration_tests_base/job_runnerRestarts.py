"""
A graph where each node requires a different version of numpy.
An environment with a single host.

This validates that venv corruption is not happening -- ie, a previous
task's pip installs can be erased reliably when a new task with
conflicting requirements is executed.

Additionally, a pair of jobs validates the requires_new_worker flag: set_global modifies the process
state, check_global_empty fails if it observes that modification -- ie, it succeeds only in a new worker.
"""

from collections.abc import Mapping
from typing import Any

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from integration_tests_base.base import JobSpec, TestCase, expect_success


def job() -> JobInstanceRich:
    def fac(version: str) -> TaskBuilder:
        return TaskBuilder.from_entrypoint(
            "integration_tests_runtime.check_numpy_version",
            {"expected": "str"},
            "bool",
            [f"numpy=={version}"],
        ).with_values(expected=version)

    ji = JobBuilder().with_node("t1", fac("2.3.5")).with_node("t2", fac("2.4.1")).with_node("t3", fac("2.4.2")).build().get_or_raise()
    ji.ext_outputs = [
        DatasetId(task=TaskId("t1"), output=DefaultTaskOutput),
        DatasetId(task=TaskId("t2"), output=DefaultTaskOutput),
        DatasetId(task=TaskId("t3"), output=DefaultTaskOutput),
    ]
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    if not outputs:
        raise AssertionError("expected outputs")
    if not all(value is True for value in outputs.values()):
        raise AssertionError(f"unexpected outputs: {list(outputs.values())!r}")


def _global_job(requires_new_worker: bool) -> JobInstanceRich:
    set_task = TaskBuilder.from_entrypoint("integration_tests_runtime.set_global", {}, "int", [])
    check_task = TaskBuilder.from_entrypoint("integration_tests_runtime.check_global_empty", {"a": "int"}, "bool", [])
    check_task = check_task.model_copy(
        update={"definition": check_task.definition.model_copy(update={"requires_new_worker": requires_new_worker})}
    )
    ji = (
        JobBuilder()
        .with_node("set_global", set_task)
        .with_node("check_global_empty", check_task)
        .with_edge("set_global", "check_global_empty", "a")
        .build()
        .get_or_raise()
    )
    ji.ext_outputs = [DatasetId(task=TaskId("check_global_empty"), output=DefaultTaskOutput)]
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def outputFails(result: Mapping[Any, Any] | Exception) -> None:
    if not isinstance(result, Exception):
        raise AssertionError(f"expected failure, got outputs: {result!r}")


def cases() -> list[TestCase]:
    return [
        TestCase(job=job(), spec=spc(), outputOk=expect_success(outputOk)),
        # new worker is started for the second task -> clean state
        TestCase(job=_global_job(True), spec=spc(), outputOk=expect_success(outputOk)),
        # fused into the same worker, the global state leaks -> expected failure
        TestCase(job=_global_job(False), spec=spc(), outputOk=outputFails),
    ]
