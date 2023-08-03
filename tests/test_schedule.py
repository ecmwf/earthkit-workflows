import pytest
import networkx as nx

from cascade.graphs import TaskGraph, ContextGraph
from cascade.scheduler import Schedule


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
    task_graph = TaskGraph()
    task_graph.add_task(1, 1, 1, "mean")
    for index in range(2):
        task_graph.add_task(1, 1, 1, f"read-{index}")
        task_graph.add_task(1, 1, 1, f"sh2gp-{index}")
        task_graph.add_comm_edge(f"read-{index}", f"sh2gp-{index}", 1)
        task_graph.add_comm_edge(f"sh2gp-{index}", "mean", 1)

    assert not nx.dag.has_cycle(task_graph)

    assert Schedule.valid_allocations(task_graph, allocations) == exp
