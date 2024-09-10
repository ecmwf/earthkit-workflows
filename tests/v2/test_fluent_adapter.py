from dask.threaded import get

from cascade.graph.samplegraphs import linear
from cascade.v2.dask import job2dask
from cascade.v2.fluent import graph2job


def test_fluent():
    """Tests that a linear dag with process-i being a function +1, when converted to v2.api and then to dask gives
    correct results upon execution"""
    N = 5
    g = linear(N)
    for node in g.nodes():
        if node.is_source():
            node.payload = (lambda input: input, [], {"input": 0})
        elif node.is_processor():
            node.payload = (lambda input: input + 1, ["input"], {})
        elif node.is_sink():
            node.payload = (lambda input: input, ["input"], {})
    j = graph2job(g)
    d = job2dask(j)
    r = get(d, "writer")
    assert r == N
