import pytest
import random

from cascade.schedulers.schedule import Schedule
from cascade.graph import Graph, Graph
from cascade.taskgraph import Resources, Task
from cascade.contextgraph import ContextGraph
from cascade.transformers import to_task_graph

from cascade.schedulers.anneal import AnnealingScheduler
from cascade.schedulers.depthfirst import DepthFirstScheduler


def setup_context():
    context = ContextGraph()
    context.add_node("gpu_1", type="GPU", speed=10, memory=40)
    context.add_node("gpu_2", type="GPU", speed=10, memory=20)
    context.add_node("gpu_3", type="GPU", speed=5, memory=40)
    context.add_node("gpu_4", type="GPU", speed=5, memory=20)
    context.add_edge("gpu_1", "gpu_2", bandwidth=0.1, latency=1)
    context.add_edge("gpu_1", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_1", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_3", "gpu_4", bandwidth=0.1, latency=1)
    return context


def resource_map(graph: Graph) -> dict[str, Resources]:
    res_map = {}
    for node in graph.nodes():
        res_map[node] = Resources(random.randrange(1, 100), random.randrange(1, 2))
    return res_map


def example_graph(num_inputs: int):
    mean = Task(name="mean", payload="mean")
    for index in range(num_inputs):
        read = Task(name=f"read-{index}", payload=f"read-{index}")
        sh2gp = Task(name=f"sh2gp-{index}", payload=f"sh2gp-{index}")
        sh2gp.inputs[f"inputs0"] = read.get_output()
        mean.inputs[f"inputs{index}"] = sh2gp.get_output()
    return Graph([mean])


@pytest.mark.parametrize(
    "allocations, exp",
    [
        [
            {"p1": ["read-0", "sh2gp-0", "mean"], "p2": ["read-1", "sh2gp-1", "mean"]},
            True,
        ],
        [
            {"p1": ["read-0", "sh2gp-1", "mean"], "p2": ["sh2gp-0", "read-1", "mean"]},
            True,
        ],
        [
            {"p1": ["sh2gp-1", "read-0", "mean"], "p2": ["sh2gp-0", "read-1", "mean"]},
            False,
        ],
    ],
)
def test_valid_allocations(allocations, exp):
    task_graph = example_graph(2)
    assert not task_graph.has_cycle()
    assert Schedule.valid_allocations(task_graph, allocations) == exp


def test_depth_first_scheduler():
    context = setup_context()
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    DepthFirstScheduler().schedule(task_graph, context)


@pytest.mark.skip(
    "Initial cost from schedule can sometimes 0 resulting in ZeroDivisionError error"
)
def test_annealing_scheduler():
    context = setup_context()
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    AnnealingScheduler().schedule(task_graph, context, num_temp_levels=10, num_tries=10)
