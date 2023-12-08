import pytest
import random

from cascade.cascade import Cascade
from cascade.scheduler import Schedule
from cascade.graph import Graph, Node, Sink, Graph, Transformer
from cascade.taskgraph import Task, TaskGraph
from cascade.contextgraph import ContextGraph
from cascade.scheduler import AnnealingScheduler


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


class _AssignRandomResources(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        newnode.cost = random.randrange(1, 100)
        newnode.in_memory = random.randrange(1, 2)
        newnode.out_memory = random.randrange(1, 2)
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def add_resources(graph: Graph) -> TaskGraph:
    return _AssignRandomResources().transform(graph)


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
    schedule = Cascade.schedule(add_resources(graph), context)
    Cascade.simulate(schedule)


@pytest.mark.skip("Initial cost from schedule can sometimes 0 resulting in ZeroDivisionError error")
def test_annealing_scheduler():
    context = setup_context()
    graph = example_graph(5)
    scheduler = AnnealingScheduler(add_resources(graph), context)
    schedule = scheduler.create_schedule(num_temp_levels=10, num_tries=10)
    Cascade.simulate(schedule)
