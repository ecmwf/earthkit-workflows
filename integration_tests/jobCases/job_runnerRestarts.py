"""
A graph where each node requires a different version of numpy.
An environment with a single host.

This validates that venv corruption is not happening -- ie, a previous
task's pip installs can be erased reliably when a new task with
conflicting requirements is executed.
"""

from collections.abc import Mapping

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from integration_tests.jobCases.base import JobSpec


def job() -> JobInstanceRich:
    def fac(version: str) -> TaskBuilder:
        return TaskBuilder.from_entrypoint(
            "runtime.check_numpy_version",
            {"expected": "str"},
            "bool",
            [f"numpy=={version}"],
        ).with_values(expected=version)

    ji = JobBuilder().with_node("t1", fac("2.0.1")).with_node("t2", fac("2.4.1")).with_node("t3", fac("2.4.2")).build().get_or_raise()
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
