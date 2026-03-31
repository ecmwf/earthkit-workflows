# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from typing import cast

from cascade.low.core import TaskId
from cascade.scheduler.precompute import _decompose, _enrich


def _oedge2iedge(edge_o: dict[TaskId, set[TaskId]]) -> dict[TaskId, set[TaskId]]:
    edge_i: dict[TaskId, set[TaskId]] = defaultdict(set)
    for v, inps in edge_o.items():
        for i in inps:
            edge_i[i] = edge_i[i].union({v})
    return edge_i


def test_decompose():
    # comp1: v0 -> v1 -> v2 + v3 -> v1
    # comp2: v4 -> v5, v4 -> v6
    nodes = [TaskId(f"v{i}") for i in range(7)]
    edge_o: dict[TaskId, set[TaskId]] = defaultdict(set)
    edge_o.update(
        {
            TaskId("v0"): {TaskId("v1")},
            TaskId("v1"): {TaskId("v2")},
            TaskId("v3"): {TaskId("v1")},
            TaskId("v4"): {TaskId("v5"), TaskId("v6")},
        }
    )
    edge_i = _oedge2iedge(edge_o)

    expected = {
        (frozenset({TaskId("v0"), TaskId("v1"), TaskId("v2"), TaskId("v3")}), frozenset({TaskId("v0"), TaskId("v3")})),
        (frozenset({TaskId("v4"), TaskId("v5"), TaskId("v6")}), frozenset({TaskId("v4")})),
    }
    for component in _decompose(nodes, edge_i, edge_o):
        e = (frozenset(component[0]), frozenset(component[1]))
        expected.remove(e)

    assert expected == set()


def test_enrich():
    # v0 -> v1 -> v2
    # v3 -> v1
    # v4 -> v5 -> v2
    # v4 -> v6
    edge_o: dict[TaskId, set[TaskId]] = defaultdict(set)
    edge_o.update(
        {
            TaskId("v0"): {TaskId("v1")},
            TaskId("v1"): {TaskId("v2")},
            TaskId("v3"): {TaskId("v1")},
            TaskId("v4"): {TaskId("v5"), TaskId("v6")},
            TaskId("v5"): {TaskId("v2")},
        }
    )
    edge_i = _oedge2iedge(edge_o)
    component = (list(set(edge_o.keys()).union(set(edge_i.keys()))), [TaskId("v0"), TaskId("v3"), TaskId("v4")])

    res = _enrich(component, edge_i, edge_o, set(), set())

    assert res.nodes == component[0]
    assert res.sources == component[1]
    assert res.weight() == len(component[0])
    value: dict[TaskId, int] = {
        TaskId("v0"): 1,
        TaskId("v1"): 2,
        TaskId("v2"): 3,
        TaskId("v3"): 1,
        TaskId("v4"): 2,
        TaskId("v5"): 2,
        TaskId("v6"): 3,
    }
    assert res.value == value
    distance_matrix: dict[TaskId, dict[TaskId, int]] = {
        TaskId("v0"): {
            TaskId("v0"): 0,
            TaskId("v1"): 1,
            TaskId("v2"): 2,
            TaskId("v3"): 1,
            TaskId("v4"): 2,
            TaskId("v5"): 2,
            TaskId("v6"): 3,
        },
        TaskId("v1"): {
            TaskId("v0"): 1,
            TaskId("v1"): 0,
            TaskId("v2"): 1,
            TaskId("v3"): 1,
            TaskId("v4"): 2,
            TaskId("v5"): 1,
            TaskId("v6"): 3,
        },
        TaskId("v2"): {
            TaskId("v0"): 2,
            TaskId("v1"): 1,
            TaskId("v2"): 0,
            TaskId("v3"): 2,
            TaskId("v4"): 2,
            TaskId("v5"): 1,
            TaskId("v6"): 3,
        },
        TaskId("v3"): {
            TaskId("v0"): 1,
            TaskId("v1"): 1,
            TaskId("v2"): 2,
            TaskId("v3"): 0,
            TaskId("v4"): 2,
            TaskId("v5"): 2,
            TaskId("v6"): 3,
        },
        TaskId("v4"): {
            TaskId("v0"): 2,
            TaskId("v1"): 2,
            TaskId("v2"): 2,
            TaskId("v3"): 2,
            TaskId("v4"): 0,
            TaskId("v5"): 1,
            TaskId("v6"): 1,
        },
        TaskId("v5"): {
            TaskId("v0"): 2,
            TaskId("v1"): 1,
            TaskId("v2"): 1,
            TaskId("v3"): 2,
            TaskId("v4"): 1,
            TaskId("v5"): 0,
            TaskId("v6"): 3,
        },
        TaskId("v6"): {
            TaskId("v0"): 3,
            TaskId("v1"): 3,
            TaskId("v2"): 3,
            TaskId("v3"): 3,
            TaskId("v4"): 1,
            TaskId("v5"): 3,
            TaskId("v6"): 0,
        },
    }
    assert res.distance_matrix == distance_matrix
