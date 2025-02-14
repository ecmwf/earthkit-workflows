import functools

import numpy as np
import pytest

from cascade.contextgraph import ContextGraph
from cascade.fluent import Payload, from_source


@pytest.fixture(scope="function")
def context_graph():
    cont_graph = ContextGraph()
    cont_graph.add_node("worker_1", type="CPU", speed=10, memory=400)
    cont_graph.add_node("worker_2", type="CPU", speed=10, memory=200)
    cont_graph.add_node("worker_3", type="CPU", speed=10, memory=200)
    cont_graph.add_edge("worker_1", "worker_2", bandwidth=0.1, latency=1)
    cont_graph.add_edge("worker_2", "worker_3", bandwidth=0.1, latency=1)
    cont_graph.add_edge("worker_1", "worker_3", bandwidth=0.1, latency=1)
    cont_graph
    return cont_graph


@pytest.fixture(scope="function")
def task_graph(request):
    func = getattr(request, "param", functools.partial(np.random.rand, 2, 3))
    return (
        from_source(
            [
                np.fromiter(
                    [func for _ in range(6)],
                    dtype=object,
                )
                for _ in range(7)
            ],
            dims=["x", "y"],
        )
        .mean("x")
        .min("y")
        .expand("z", internal_dim=1, dim_size=3, axis=0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )
