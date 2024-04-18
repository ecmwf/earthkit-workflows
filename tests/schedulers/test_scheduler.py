import pytest
import random
from contextlib import nullcontext as does_not_raise

from cascade.schedulers.schedule import Schedule
from cascade.graph import Graph, Graph
from cascade.taskgraph import Resources, Task
from cascade.contextgraph import ContextGraph
from cascade.transformers import to_task_graph

from cascade.schedulers.anneal import AnnealingScheduler
from cascade.schedulers.depthfirst import DepthFirstScheduler

from schedule_utils import context, example_graph


def resource_map(graph: Graph) -> dict[str, Resources]:
    res_map = {}
    for node in graph.nodes():
        res_map[node.name] = Resources(random.randrange(1, 100), random.randrange(1, 2))
    return res_map


def example_graph(num_inputs: int):
    mean = Task(name="mean", payload="mean")
    for index in range(num_inputs):
        read = Task(name=f"read-{index}", payload=f"read-{index}")
        sh2gp = Task(name=f"sh2gp-{index}", payload=f"sh2gp-{index}")
        sh2gp.inputs[f"inputs0"] = read.get_output()
        mean.inputs[f"inputs{index}"] = sh2gp.get_output()
    return Graph([mean])


def test_depth_first_scheduler(context):
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    DepthFirstScheduler().schedule(task_graph, context)


@pytest.mark.parametrize(
    "cost_function",
    [AnnealingScheduler.total_idle_time, AnnealingScheduler.total_execution_time],
)
def test_annealing_scheduler(context, cost_function):
    graph = example_graph(5)
    task_graph = to_task_graph(graph, resource_map(graph))
    AnnealingScheduler(cost_function, num_temp_levels=10, num_tries=10).schedule(
        task_graph, context
    )
