import pytest
import networkx as nx

from ppgraph import Graph
from ppgraph.networkx import to_networkx

from cascade.scheduler import Schedule
from cascade.graphs import Task


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
    assert not nx.dag.has_cycle(to_networkx(task_graph))

    assert Schedule.valid_allocations(task_graph, allocations) == exp
