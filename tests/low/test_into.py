from dask.threaded import get

from cascade.graph.samplegraphs import linear
from cascade.low.into import graph2job
from cascade.low.delayed import job2delayed


def test_fluent():
    """Tests that a linear dag with process-i being a function +1, when lowered and then executed in dask gives
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
    d = job2delayed(j)
    r = get(d, "writer")
    assert r == N
