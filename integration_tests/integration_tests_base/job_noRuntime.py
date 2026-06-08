"""Needs no runtime at all -- the simplest possible job to run"""

from collections.abc import Mapping

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from integration_tests_base.base import JobSpec


def job() -> JobInstanceRich:
    b = JobBuilder()
    b = b.with_node("source", TaskBuilder.from_entrypoint("cascade.executor.platform.get_bindabble_self", {}, "str", []))
    ji = b.build().get_or_raise()
    ji.ext_outputs = [DatasetId(task=TaskId("source"), output=DefaultTaskOutput)]
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    if not any(isinstance(value, str) and value for value in outputs.values()):
        raise AssertionError(f"unexpected outputs: {list(outputs.values())!r}")
