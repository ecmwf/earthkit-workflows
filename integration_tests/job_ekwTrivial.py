"""
A trivial job based on earthkit-workflows fluent custom actions.
"""

from base import JobSpec  # ty:ignore[unresolved-import]
from runtime import product_add, sink_file, source_42, transform_increment  # ty:ignore[unresolved-import]

from cascade.low.core import JobInstanceRich
from earthkit.workflows.compilers import graph2job
from earthkit.workflows.fluent import Action, Payload, from_source


def job() -> JobInstanceRich:
    source = from_source(source_42)
    trans = source.map(transform_increment)
    prod = trans.join(source, dim="inputs").reduce(product_add)
    sink = prod.map(Payload(sink_file, kwargs={"fname": "/tmp/file.txt"}))

    graph = sink.graph()
    ji = graph2job(graph)

    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)
