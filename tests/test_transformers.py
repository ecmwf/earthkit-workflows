import numpy as np
import xarray as xr
import pytest
import functools

from cascade.transformers import to_task_graph, to_dask_graph
from cascade.fluent import Payload, Node, Action

from helpers import mock_graph


def test_taskgraph_transform():
    graph = mock_graph(functools.partial(np.random.rand, 100, 100))
    result = to_task_graph(graph, None)
    assert np.any([node.memory > 0 for node in result.nodes()])
    assert np.any([node.cost > 0 for node in result.nodes()])


def test_dask_transform():
    graph = mock_graph(functools.partial(np.random.rand, 100, 100))
    dask_graph = to_dask_graph(graph)
    assert all([isinstance(x, tuple) for x in dask_graph.items()])

    # If graph contains nodes with same name then check conversion
    # to dask causes raise
    node = Node(Payload(np.random.rand, [2, 3]))
    graph = Action(xr.DataArray([node, node], dims=["x"])).mean("x").graph
    with pytest.raises(Exception):
        to_dask_graph(graph)
