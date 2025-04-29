# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.workflows.graph import Graph, Node, Transformer
from earthkit.workflows.graph.samplegraphs import (
    disconnected,
    empty,
    linear,
    multi,
    simple,
)

D = Node.DEFAULT_OUTPUT


class ToSimple(Transformer):
    def source(self, s: Node) -> str:
        return s.name

    def processor(self, p: Node, **inputs: Any) -> tuple[str, dict[str, Any]]:
        return (p.name, inputs)

    def sink(self, s: Node, **inputs: Any) -> tuple[str, dict[str, Any]]:
        return s.name, inputs

    def output(
        self, node: str | tuple[str, Any], name: str
    ) -> str | tuple[str, dict[str, Any]]:
        if isinstance(node, str):  # source
            return f"{node}.{name}"
        pname, inputs = node  # processor
        return (f"{pname}.{name}", inputs)


def to_simple(g: Graph) -> list[tuple[str, dict[str, Any]]]:
    return ToSimple().transform(g)


def test_transform_empty():
    g = empty()
    gt = to_simple(g)
    assert gt == []


def test_transform_linear():
    g = linear(2)
    gt = to_simple(g)
    assert gt == [
        (
            "writer",
            {
                "input": (
                    f"process-1.{D}",
                    {
                        "input": (
                            f"process-0.{D}",
                            {
                                "input": f"reader.{D}",
                            },
                        ),
                    },
                ),
            },
        )
    ]


def test_transform_disconnected():
    g = disconnected(5)
    gt = to_simple(g)
    assert gt == [
        (
            f"writer-{i}",
            {
                "input": (
                    f"process-{i}.{D}",
                    {
                        "input": f"reader-{i}.{D}",
                    },
                )
            },
        )
        for i in range(5)
    ]


# node -> input -> (node, output)
DepDict = dict[str, dict[str, tuple[str, str]]]


class ToDepDict(Transformer):
    def node(self, n: Node, **inputs: tuple[str, str, DepDict]) -> tuple[str, DepDict]:
        res = {}
        ndeps = {}
        for iname, (nname, oname, odeps) in inputs.items():
            res.update(odeps)
            ndeps[iname] = (nname, oname)
        res[n.name] = ndeps
        return n.name, res

    def output(
        self, ntrans: tuple[str, DepDict], oname: str
    ) -> tuple[str, str, DepDict]:
        nname, ndeps = ntrans
        return nname, oname, ndeps

    def graph(self, g: Graph, sinks: list[tuple[str, DepDict]]) -> DepDict:
        res = {}
        for _, sdeps in sinks:
            res.update(sdeps)
        return res


def to_depdict(g: Graph) -> DepDict:
    return ToDepDict().transform(g)


def test_depdict_empty():
    g = empty()
    deps = to_depdict(g)
    assert deps == {}


def test_depdict_linear():
    g = linear(4)
    deps = to_depdict(g)
    assert deps == {
        "writer": {"input": ("process-3", D)},
        "process-3": {"input": ("process-2", D)},
        "process-2": {"input": ("process-1", D)},
        "process-1": {"input": ("process-0", D)},
        "process-0": {"input": ("reader", D)},
        "reader": {},
    }


def test_depdict_disc():
    g = disconnected(6)
    deps = to_depdict(g)
    edeps = {}
    for i in range(6):
        edeps.update(
            {
                f"writer-{i}": {"input": (f"process-{i}", D)},
                f"process-{i}": {"input": (f"reader-{i}", D)},
                f"reader-{i}": {},
            }
        )
    assert deps == edeps


def test_depdict_simple():
    g = simple(2, 3)
    deps = to_depdict(g)
    assert deps == {
        "writer-0": {"input": ("process-0", D)},
        "writer-1": {"input": ("process-1", D)},
        "writer-2": {"input": ("process-2", D)},
        "process-0": {"input0": ("reader-0", D), "input1": ("reader-1", D)},
        "process-1": {"input0": ("reader-0", D), "input1": ("reader-1", D)},
        "process-2": {"input0": ("reader-0", D), "input1": ("reader-1", D)},
        "reader-0": {},
        "reader-1": {},
    }


def test_depdict_multi():
    g = multi(2, 3, 2)
    deps = to_depdict(g)
    assert deps == {
        "writer-1": {"input1": ("process-1", D), "input2": ("process-2", "output1")},
        "writer-0": {"input1": ("process-1", D), "input2": ("process-2", "output0")},
        "process-2": {"input1": ("process-0", "output1"), "input2": ("reader-1", D)},
        "process-1": {
            "input1": ("process-0", "output0"),
            "input2": ("process-0", "output2"),
        },
        "process-0": {"input0": ("reader-0", D), "input1": ("reader-1", D)},
        "reader-0": {},
        "reader-1": {},
    }
