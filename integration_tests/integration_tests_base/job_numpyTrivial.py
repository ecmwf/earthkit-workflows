"""
A line of numpy operations
"""

from collections.abc import Mapping

import numpy

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from integration_tests_base.base import JobSpec


def job() -> JobInstanceRich:
    b = JobBuilder()
    b = b.with_node("source", TaskBuilder.from_entrypoint("integration_tests_runtime.source_numpy", {}, "numpy.ndarray", ["numpy"]))
    b = b.with_node(
        "t1", TaskBuilder.from_entrypoint("integration_tests_runtime.transform_numpy", {"a": "numpy.ndarray"}, "numpy.ndarray", ["numpy"])
    ).with_edge("source", "t1", "a")
    ji = b.build().get_or_raise()
    ji.ext_outputs = [DatasetId(task=TaskId("t1"), output=DefaultTaskOutput)]
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    if not any(numpy.array_equal(numpy.asarray(value), numpy.array([2])) for value in outputs.values()):
        raise AssertionError(f"unexpected outputs: {list(outputs.values())!r}")
