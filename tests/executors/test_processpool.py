import numpy as np
import xarray as xr

from cascade.executors.processpool import ProcessPoolExecutor
from cascade.schedulers.depthfirst import DepthFirstScheduler
from cascade.fluent import Fluent, Payload
from cascade.contextgraph import ContextGraph

context_graph = ContextGraph()
context_graph.add_node("worker_1", type="CPU", speed=10, memory=400)
context_graph.add_node("worker_2", type="CPU", speed=10, memory=200)
context_graph.add_edge("worker_1", "worker_2", bandwidth=0.1, latency=1)
context_graph


task_graph = (
    Fluent()
    .source(
        np.random.rand,
        xr.DataArray(
            [np.fromiter([(2, 3) for _ in range(6)], dtype=object) for _ in range(7)],
            dims=["x", "y"],
        ),
    )
    .mean("x")
    .min("y")
    .expand("z", 3, 1, 0)
    .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
    .graph()
)


def test_without_schedule():
    ProcessPoolExecutor.execute(task_graph, n_workers=2)


def test_with_schedule():
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)
    ProcessPoolExecutor.execute(schedule)
