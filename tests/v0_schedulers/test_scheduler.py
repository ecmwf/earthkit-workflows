import random

import pytest
from schedule_utils import example_graph

from cascade.graph import Graph
from cascade.taskgraph import Resources
from cascade.transformers import to_task_graph
from cascade.v0_schedulers.anneal import AnnealingScheduler
from cascade.v0_schedulers.depthfirst import DepthFirstScheduler


def resource_map(graph: Graph) -> dict[str, Resources]:
    res_map = {}
    for node in graph.nodes():
        res_map[node.name] = Resources(random.randrange(1, 100), random.randrange(1, 2))
    return res_map


def test_depth_first_scheduler(context_graph):
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    DepthFirstScheduler().schedule(task_graph, context_graph)


@pytest.mark.parametrize(
    "cost_function",
    [AnnealingScheduler.total_idle_time, AnnealingScheduler.total_execution_time],
)
def test_annealing_scheduler(context_graph, cost_function):
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    AnnealingScheduler(cost_function, num_temp_levels=10, num_tries=10).schedule(
        task_graph, context_graph
    )
