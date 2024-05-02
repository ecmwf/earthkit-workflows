import functools

import numpy as np
import pytest
import xarray as xr

from cascade.contextgraph import ContextGraph
from cascade.fluent import Payload, from_source


@pytest.fixture
def execution_context():
    context_graph = ContextGraph()
    context_graph.add_node("worker_1", type="CPU", speed=10, memory=400)
    context_graph.add_node("worker_2", type="CPU", speed=10, memory=200)
    context_graph.add_edge("worker_1", "worker_2", bandwidth=0.1, latency=1)
    context_graph

    task_graph = (
        from_source(
            [
                np.fromiter(
                    [functools.partial(np.random.rand, 2, 3) for _ in range(6)],
                    dtype=object,
                )
                for _ in range(7)
            ],
            ["x", "y"],
        )
        .mean("x")
        .min("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )

    return task_graph, context_graph
