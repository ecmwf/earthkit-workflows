import numpy as np
import xarray as xr
import pytest

from cascade.transformers import to_task_graph, to_dask_graph
from cascade.fluent import Fluent, Payload, Node, MultiAction
from cascade.backends.arrayapi import ArrayApiBackend


def test_taskgraph_transform():
    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = Payload(np.random.rand, [100, 100])
    graph = (
        Fluent()
        .source(payloads, ["x", "y"])
        .mean("x")
        .minimum("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )
    result = to_task_graph(graph, None)
    assert np.any([node.memory > 0 for node in result.nodes()])
    assert np.any([node.cost > 0 for node in result.nodes()])


def test_dask_transform():
    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = Payload(np.random.rand, [2, 3])
    example = Fluent().source(payloads, ["x", "y"]).mean("x").expand("z", 3, 1, 0)

    dask_graph = to_dask_graph(example.graph())
    assert all([isinstance(x, tuple) for x in dask_graph.items()])

    # If graph contains nodes with same name then check conversion
    # to dask causes raise
    node = Node(Payload(np.random.rand, [2, 3]))
    graph = (
        MultiAction(None, xr.DataArray([node, node], dims=["x"]), ArrayApiBackend)
        .mean("x")
        .graph
    )
    with pytest.raises(Exception):
        to_dask_graph(graph)
