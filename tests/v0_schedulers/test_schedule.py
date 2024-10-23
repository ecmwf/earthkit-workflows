from contextlib import nullcontext as does_not_raise

import pytest
from schedule_utils import example_graph

from cascade.v0_schedulers.schedule import Schedule


@pytest.mark.parametrize(
    "allocations, expectation",
    [
        [
            {"p1": ["read-0", "sh2gp-0", "mean"], "p2": ["read-1", "sh2gp-1", "mean"]},
            does_not_raise(),
        ],
        [
            {"p1": ["read-0", "sh2gp-1", "mean"], "p2": ["sh2gp-0", "read-1", "mean"]},
            does_not_raise(),
        ],
        [
            {"p1": ["sh2gp-1", "read-0", "mean"], "p2": ["sh2gp-0", "read-1", "mean"]},
            pytest.raises(ValueError),
        ],
    ],
)
def test_valid_allocations(context_graph, allocations, expectation):
    task_graph = example_graph(2)
    assert not task_graph.has_cycle()
    with expectation:
        Schedule(task_graph, context_graph, allocations)
