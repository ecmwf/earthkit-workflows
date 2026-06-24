"""
A line of writing grib outputs
"""

# TODO support distributed in the outputOk

from collections.abc import Mapping

import numpy

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from integration_tests_base.base import JobSpec


def job() -> JobInstanceRich:
    b = JobBuilder()
    ekdType = "earthkit.data.readers.grib.file.GRIBReader"
    deps = []
    b = b.with_node("t0", TaskBuilder.from_entrypoint("integration_tests_runtime.source_ekd", {}, ekdType, deps))
    N = 100
    for i in range(N):
        b = b.with_node(
            f"t{i + 1}",
            TaskBuilder.from_entrypoint("integration_tests_runtime.write_grib", {"a": ekdType, "i": "int"}, ekdType, deps).with_values(i=i),
        ).with_edge(f"t{i}", f"t{i + 1}", "a")
    ji = b.build().get_or_raise()
    ji.ext_outputs = [DatasetId(task=TaskId(f"t{N}"), output=DefaultTaskOutput)]
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    pass
    # TODO check that the files exist
