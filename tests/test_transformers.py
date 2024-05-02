import functools

import numpy as np
import pytest
import xarray as xr
from helpers import mock_graph

from cascade.fluent import Action, Node, Payload
from cascade.transformers import to_dask_graph


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
