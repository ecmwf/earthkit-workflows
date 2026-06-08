"""
A trivial job with torch tensor sum over two steps
"""

from collections.abc import Mapping
from pathlib import Path

from integration_tests_runtime import sink_file, source_tensor, transform_tensorsum  # ty: ignore

from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstance, JobInstanceRich, TaskId
from earthkit.workflows.compilers import graph2job
from earthkit.workflows.fluent import Payload, from_source
from integration_tests_base.base import JobSpec


def _task_id(ji: JobInstance, prefix: str) -> TaskId:
    for task_id in ji.tasks:
        if str(task_id).startswith(prefix):
            return task_id
    raise AssertionError(f"missing task with prefix {prefix!r}")


def job() -> JobInstanceRich:
    source = from_source(source_tensor)
    trans = source.map(transform_tensorsum)
    sink = trans.map(Payload(sink_file, kwargs={"fname": "/tmp/torchTrivial.txt"}))

    graph = sink.graph()
    ji = graph2job(graph)
    for task in ji.tasks:
        if "tensor" in task:
            ji.tasks[task].definition.environment = ["torch"]
    ji.ext_outputs = [
        DatasetId(task=_task_id(ji, "sink_file:"), output=DefaultTaskOutput),
        DatasetId(task=_task_id(ji, "transform_tensorsum:"), output=DefaultTaskOutput),
    ]

    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)


def outputOk(outputs: Mapping[object, object]) -> None:
    if not any(value == 6 for value in outputs.values()):
        raise AssertionError(f"expected tensor sum, got {list(outputs.values())!r}")

    file_path = Path("/tmp/torchTrivial.txt")
    if file_path.exists():
        content = file_path.read_text()
        if content != "6":
            raise AssertionError(f"unexpected file content: {content!r}")
