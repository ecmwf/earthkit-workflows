"""
A trivial job based on earthkit-workflows fluent custom actions.
"""

from cascade.low.core import JobInstanceRich
from base import JobSpec # ty:ignore[unresolved-import]

from earthkit.workflows.fluent import Action, Payload, from_source
from earthkit.workflows.compilers import graph2job

from runtime import source_42, transform_increment, product_add, sink_file # ty:ignore[unresolved-import]


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
