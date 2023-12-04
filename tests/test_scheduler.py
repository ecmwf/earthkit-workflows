import pytest
import networkx as nx

from cascade.scheduler import Schedule
from cascade.graph import Graph
from cascade.taskgraph import Task


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
    mean = Task(name="mean", payload="mean")
    for index in range(2):
        read = Task(name=f"read-{index}", payload=f"read-{index}")
        sh2gp = Task(name=f"sh2gp-{index}", payload=f"sh2gp-{index}")
        sh2gp.inputs[f"inputs0"] = read.get_output()
        mean.inputs[f"inputs{index}"] = sh2gp.get_output()

    task_graph = Graph([mean])
    assert not task_graph.has_cycle()

    assert Schedule.valid_allocations(task_graph, allocations) == exp
