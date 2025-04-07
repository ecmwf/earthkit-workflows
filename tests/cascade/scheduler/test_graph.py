# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

from cascade.low.core import TaskId
from cascade.scheduler.graph import decompose, enrich


def _oedge2iedge(edge_o: dict[TaskId, set[TaskId]]) -> dict[TaskId, set[TaskId]]:
    edge_i: dict[TaskId, set[TaskId]] = defaultdict(set)
    for v, inps in edge_o.items():
        for i in inps:
            edge_i[i] = edge_i[i].union({v})
    return edge_i


def test_decompose():
    # comp1: v0 -> v1 -> v2 + v3 -> v1
    # comp2: v4 -> v5, v4 -> v6
    nodes = [f"v{i}" for i in range(7)]
    edge_o = defaultdict(
        set,
        **{
            "v0": {"v1"},
            "v1": {"v2"},
            "v3": {"v1"},
            "v4": {"v5", "v6"},
        },
    )
    edge_i = _oedge2iedge(edge_o)

    expected = {
        (frozenset({"v0", "v1", "v2", "v3"}), frozenset({"v0", "v3"})),
        (frozenset({"v4", "v5", "v6"}), frozenset({"v4"})),
    }
    for component in decompose(nodes, edge_i, edge_o):
        e = (frozenset(component[0]), frozenset(component[1]))
        expected.remove(e)

    assert expected == set()


def test_enrich():
    # v0 -> v1 -> v2
    # v3 -> v1
    # v4 -> v5 -> v2
    # v4 -> v6
    edge_o = defaultdict(
        set,
        **{
            "v0": {"v1"},
            "v1": {"v2"},
            "v3": {"v1"},
            "v4": {"v5", "v6"},
            "v5": {"v2"},
        },
    )
    edge_i = _oedge2iedge(edge_o)
    component = (list(set(edge_o.keys()).union(set(edge_i.keys()))), ["v0", "v3", "v4"])

    res = enrich(component, edge_i, edge_o)

    assert res.nodes == component[0]
    assert res.sources == component[1]
    assert res.weight() == len(component[0])
    value = {
        "v0": 1,
        "v1": 2,
        "v2": 3,
        "v3": 1,
        "v4": 2,
        "v5": 2,
        "v6": 3,
    }
    assert res.value == value
    distance_matrix = {
        "v0": {"v0": 0, "v1": 1, "v2": 2, "v3": 1, "v4": 2, "v5": 2, "v6": 3},
        "v1": {"v0": 1, "v1": 0, "v2": 1, "v3": 1, "v4": 2, "v5": 1, "v6": 3},
        "v2": {"v0": 2, "v1": 1, "v2": 0, "v3": 2, "v4": 2, "v5": 1, "v6": 3},
        "v3": {"v0": 1, "v1": 1, "v2": 2, "v3": 0, "v4": 2, "v5": 2, "v6": 3},
        "v4": {"v0": 2, "v1": 2, "v2": 2, "v3": 2, "v4": 0, "v5": 1, "v6": 1},
        "v5": {"v0": 2, "v1": 1, "v2": 1, "v3": 2, "v4": 1, "v5": 0, "v6": 3},
        "v6": {"v0": 3, "v1": 3, "v2": 3, "v3": 3, "v4": 1, "v5": 3, "v6": 0},
    }
    assert res.distance_matrix == distance_matrix
