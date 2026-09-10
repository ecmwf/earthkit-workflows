"""
A trivial job based on earthkit-workflows fluent custom actions.
"""

from collections.abc import Mapping
from pathlib import Path

from integration_tests_runtime import product_add, sink_file, source_42, transform_increment  # ty: ignore

from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstance, JobInstanceRich, TaskId
from earthkit.workflows.compilers import graph2job
from earthkit.workflows.fluent import create_task_instance, from_source
from integration_tests_base.base import JobSpec


def _task_id(ji: JobInstance, prefix: str) -> TaskId:
    for task_id in ji.tasks:
        if str(task_id).startswith(prefix):
            return task_id
    raise AssertionError(f"missing task with prefix {prefix!r}")


def job() -> JobInstanceRich:
    source = from_source(source_42)
    trans = source.map(transform_increment)
    prod = trans.join(source, dim="inputs").reduce(product_add)
    sink = prod.map(create_task_instance(sink_file, static_input_kw={"fname": "/tmp/ekwTrivial.txt"}))

    graph = sink.graph()
    ji = graph2job(graph)
    ji.ext_outputs = [
        DatasetId(task=_task_id(ji, "sink_file:"), output=DefaultTaskOutput),
        DatasetId(task=_task_id(ji, "product_add:"), output=DefaultTaskOutput),
    ]

    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    if not any(value == 85 for value in outputs.values()):
        raise AssertionError(f"expected product output, got {list(outputs.values())!r}")

    file_path = Path("/tmp/ekwTrivial.txt")
    if file_path.exists():
        content = file_path.read_text()
        if content != "85":
            raise AssertionError(f"unexpected file content: {content!r}")
