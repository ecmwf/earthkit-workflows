"""
A trivial job with torch tensor sum over two steps
"""

from base import JobSpec  # ty:ignore[unresolved-import]
from runtime import sink_file, source_tensor, transform_tensorsum  # ty:ignore[unresolved-import]

from cascade.low.core import JobInstanceRich
from earthkit.workflows.compilers import graph2job
from earthkit.workflows.fluent import Action, Payload, from_source


def job() -> JobInstanceRich:
    source = from_source(source_tensor)
    trans = source.map(transform_tensorsum)
    sink = trans.map(Payload(sink_file, kwargs={"fname": "/tmp/file.txt"}))

    graph = sink.graph()
    ji = graph2job(graph)
    for task in ji.tasks:
        if "tensor" in task:
            ji.tasks[task].definition.environment = ["torch"]

    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)
