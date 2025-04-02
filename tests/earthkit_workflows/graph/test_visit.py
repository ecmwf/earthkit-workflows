# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.workflows.graph import Graph, Node, Visitor
from earthkit.workflows.graph.samplegraphs import (
    disconnected,
    empty,
    linear,
    multi,
    simple,
)


class Lister(Visitor):
    nodes: list[Node]

    def __init__(self):
        self.nodes = []

    def node(self, n: Node):
        self.nodes.append(n)


def gnames(g: Graph) -> list[str]:
    lis = Lister()
    lis.visit(g)
    return [n.name for n in lis.nodes]


def test_visit_empty():
    g = empty()
    assert gnames(g) == []


def test_visit_linear():
    g = linear(5)
    names = gnames(g)
    assert names == [
        "writer",
        "process-4",
        "process-3",
        "process-2",
        "process-1",
        "process-0",
        "reader",
    ]


def test_visit_disc():
    g = disconnected(5)
    names = gnames(g)
    assert len(names) == 5 + 5 + 5
    assert set(names) == set(
        [f"reader-{i}" for i in range(5)]
        + [f"process-{i}" for i in range(5)]
        + [f"writer-{i}" for i in range(5)]
    )


def test_visit_simple():
    g = simple(5, 3)
    names = gnames(g)
    assert len(names) == 5 + 3 + 3
    assert set(names) == set(
        [f"reader-{i}" for i in range(5)]
        + [f"process-{i}" for i in range(3)]
        + [f"writer-{i}" for i in range(3)]
    )


def test_visit_multi():
    g = multi(5, 3, 2)
    names = gnames(g)
    assert len(names) == 5 + 3 + 2 * (3 - 2)
    assert set(names) == set(
        [f"reader-{i}" for i in range(5)]
        + [f"process-{i}" for i in range(3)]
        + [f"writer-{i}" for i in range(2)]
    )


class SourceLister(Visitor):
    sources: list[Node]
    others: list[Node]

    def __init__(self):
        self.sources = []
        self.others = []

    def source(self, s: Node):
        self.sources.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_sources():
    g = simple(6, 2)
    v = SourceLister()
    v.visit(g)
    for s in v.sources:
        assert s.is_source()
    for n in v.others:
        assert not n.is_source()
    snames = [s.name for s in v.sources]
    assert len(snames) == 6
    assert set(snames) == set(f"reader-{i}" for i in range(6))
    onames = [n.name for n in v.others]
    assert len(onames) == 2 + 2
    assert set(onames) == set(
        [f"process-{i}" for i in range(2)] + [f"writer-{i}" for i in range(2)]
    )


class ProcessorLister(Visitor):
    procs: list[Node]
    others: list[Node]

    def __init__(self):
        self.procs = []
        self.others = []

    def processor(self, p: Node):
        self.procs.append(p)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_processors():
    g = simple(7, 4)
    v = ProcessorLister()
    v.visit(g)
    for p in v.procs:
        assert p.is_processor()
    for n in v.others:
        assert not n.is_processor()
    pnames = [p.name for p in v.procs]
    assert len(pnames) == 4
    assert set(pnames) == set(f"process-{i}" for i in range(4))
    onames = [n.name for n in v.others]
    assert len(onames) == 7 + 4
    assert set(onames) == set(
        [f"reader-{i}" for i in range(7)] + [f"writer-{i}" for i in range(4)]
    )


class SinkLister(Visitor):
    sinks: list[Node]
    others: list[Node]

    def __init__(self):
        self.sinks = []
        self.others = []

    def sink(self, s: Node):
        self.sinks.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_sinks():
    g = simple(9, 5)
    v = SinkLister()
    v.visit(g)
    for s in v.sinks:
        assert s.is_sink()
    for n in v.others:
        assert not n.is_sink()
    snames = [s.name for s in v.sinks]
    assert len(snames) == 5
    assert set(snames) == set(f"writer-{i}" for i in range(5))
    onames = [n.name for n in v.others]
    assert len(onames) == 9 + 5
    assert set(onames) == set(
        [f"reader-{i}" for i in range(9)] + [f"process-{i}" for i in range(5)]
    )


class SegregatedLister(Visitor):
    sources: list[Node]
    procs: list[Node]
    sinks: list[Node]
    others: list[Node]

    def __init__(self):
        self.sources = []
        self.procs = []
        self.sinks = []
        self.others = []

    def source(self, s: Node):
        self.sources.append(s)

    def processor(self, p: Node):
        self.procs.append(p)

    def sink(self, s: Node):
        self.sinks.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_segregated():
    g = simple(8, 3)
    v = SegregatedLister()
    v.visit(g)
    for p in v.sources:
        assert p.is_source()
    for p in v.procs:
        assert p.is_processor()
    for p in v.sinks:
        assert p.is_sink()
    assert v.others == []
    sonames = [s.name for s in v.sources]
    assert len(sonames) == 8
    assert set(sonames) == set(f"reader-{i}" for i in range(8))
    pnames = [p.name for p in v.procs]
    assert len(pnames) == 3
    assert set(pnames) == set(f"process-{i}" for i in range(3))
    sinames = [s.name for s in v.sinks]
    assert len(sinames) == 3
    assert set(sinames) == set(f"writer-{i}" for i in range(3))
