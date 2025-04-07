# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import codecs

import pytest

from earthkit.workflows.graph import Graph, Node, Output, Transformer
from earthkit.workflows.graph.samplegraphs import disconnected, empty, linear, multi

D = Node.DEFAULT_OUTPUT


def test_equals_empty():
    g1 = empty()
    g2 = empty()
    assert g1 == g2
    assert not (g1 != g2)


def test_neq_linear_empty():
    g1 = empty()
    g2 = linear()
    assert g1 != g2
    assert not (g1 == g2)


def test_eq_multi():
    g1 = multi()
    g2 = multi()
    assert g1 == g2
    assert not (g1 != g2)


def test_eq_self():
    g1 = multi()
    assert g1 == g1
    assert not (g1 != g1)


class NameMangler(Transformer):
    def __init__(
        self,
        mangle_names: bool = True,
        mangle_outputs: bool = True,
        mangle_inputs: bool = True,
    ):
        self.mangle_names = mangle_names
        self.mangle_outputs = mangle_outputs
        self.mangle_inputs = mangle_inputs

    @staticmethod
    def mangle(s: str) -> str:
        return codecs.encode(s, "rot13")

    def mangle_name(self, name: str) -> str:
        return self.mangle(name) if self.mangle_names else name

    def mangle_output(self, oname: str) -> str:
        return self.mangle(oname) if self.mangle_outputs else oname

    def mangle_input(self, iname: str) -> str:
        return self.mangle(iname) if self.mangle_inputs else iname

    def source(self, source: Node) -> Node:
        return Node(
            self.mangle_name(source.name),
            [self.mangle_output(out) for out in source.outputs],
        )

    def processor(self, processor: Node, **inputs: Output) -> Node:
        return Node(
            self.mangle_name(processor.name),
            [self.mangle_output(out) for out in processor.outputs],
            **{self.mangle_input(iname): isrc for iname, isrc in inputs.items()},
        )

    def sink(self, sink: Node, **inputs: Output) -> Node:
        return Node(
            self.mangle_name(sink.name),
            outputs=[],
            **{self.mangle_input(iname): isrc for iname, isrc in inputs.items()},
        )

    def output(self, node: Node, name: str) -> Output:
        return Output(node, self.mangle_output(name))

    def graph(self, graph: Graph, sinks: list[Node]):
        return Graph(sinks)


def mangle(
    graph: Graph,
    mangle_names: bool = True,
    mangle_outputs: bool = True,
    mangle_inputs: bool = True,
) -> Graph:
    return NameMangler(
        mangle_names=mangle_names,
        mangle_outputs=mangle_outputs,
        mangle_inputs=mangle_inputs,
    ).transform(graph)


def test_neq_allmangled():
    g1 = multi()
    g2 = mangle(g1)
    assert g1 != g2
    assert not (g1 == g2)


def test_neq_inmangled():
    g1 = multi()
    g2 = mangle(g1, mangle_names=False, mangle_outputs=False)
    assert g1 != g2
    assert not (g1 == g2)


def test_neq_outmangled():
    g1 = multi()
    g2 = mangle(g1, mangle_names=False, mangle_inputs=False)
    assert g1 != g2
    assert not (g1 == g2)


def test_neq_outrenamed():
    r = Node("reader", outputs=["foo", "bar"])
    w = Node("writer", outputs=[], input1=r.foo, input2=r.bar)
    g1 = Graph([w])
    foom = NameMangler.mangle("foo")
    rm = Node("reader", outputs=[foom, "bar"])
    wm = Node("writer", outputs=[], input1=rm.get_output(foom), input2=rm.bar)
    g2 = Graph([wm])
    assert g1 != g2
    assert not (g1 == g2)


def test_neq_payload():
    r = Node("reader", outputs=["foo", "bar"])
    w = Node("writer", outputs=[], input1=r.foo, input2=r.bar)
    g1 = Graph([w])
    rp = Node("reader", outputs=["foo", "bar"], payload="Hello")
    wp = Node("writer", outputs=[], input1=rp.foo, input2=rp.bar)
    g2 = Graph([wp])
    assert g1 != g2
    assert not (g1 == g2)


def test_add():
    graphs = []
    for i in range(7):
        r = Node(f"reader-{i}")
        p = Node(f"process-{i}", input=r)
        w = Node(f"writer-{i}", outputs=[], input=p)
        graphs.append(Graph([w]))
    j1 = graphs[0] + graphs[1]
    assert j1 == disconnected(2)
    j2 = j1 + graphs[2]
    assert j2 == disconnected(3)
    j3 = graphs[3] + graphs[4]
    j4 = j2 + j3
    assert j4 == disconnected(5)
    j4 += graphs[5] + graphs[6]
    assert j4 == disconnected(7)


def test_iter_nodes():
    g = multi(5, 3, 2)
    nodes = [n.name for n in g.nodes()]
    assert nodes == [
        "writer-1",
        "process-2",
        "reader-2",
        "process-0",
        "reader-4",
        "reader-3",
        "reader-1",
        "reader-0",
        "process-1",
        "writer-0",
    ]


def test_iter_nodes_forwards():
    g = multi(5, 3, 2)
    nodes = [n.name for n in g.nodes(forwards=True)]
    assert nodes == [
        "reader-0",
        "reader-1",
        "reader-2",
        "reader-3",
        "reader-4",
        "process-0",
        "process-1",
        "process-2",
        "writer-1",
        "writer-0",
    ]


def test_sources():
    N = 10
    g = multi(N, 7, 4)
    sources = [s.name for s in g.sources()]
    assert len(sources) == N
    assert set(sources) == set(f"reader-{i}" for i in range(N))


def test_get_node():
    N = 7
    g = multi(N, 4, 6)
    node = g.get_node("process-0")
    assert node.name == "process-0"
    pnames = {iname: isrc.parent.name for iname, isrc in node.inputs.items()}
    assert pnames == {f"input{i}": f"reader-{i}" for i in range(N)}

    with pytest.raises(KeyError):
        g.get_node("nonexistent")


def test_predecessors():
    g = multi(5, 3, 2)

    r2 = g.get_node("reader-2")
    assert g.get_predecessors(r2) == {}

    p0 = g.get_node("process-0")
    p1 = g.get_node("process-1")
    assert g.get_predecessors(p1) == {
        "input1": (p0, "output0"),
        "input2": (p0, "output2"),
    }

    p2 = g.get_node("process-2")
    assert g.get_predecessors(p2) == {"input1": (p0, "output1"), "input2": r2}


def test_successors():
    g = multi(5, 3, 2)

    p1 = g.get_node("process-1")
    sp1 = g.get_successors(p1)
    assert list(sp1.keys()) == [D]
    assert len(sp1[D]) == len(g.sinks)
    assert set(sp1[D]) == set((s, "input1") for s in g.sinks)

    p0 = g.get_node("process-0")
    p2 = g.get_node("process-2")
    assert g.get_successors(p0) == {
        "output0": [(p1, "input1")],
        "output1": [(p2, "input1")],
        "output2": [(p1, "input2")],
    }

    w0 = g.sinks[0]
    assert g.get_successors(w0) == {}


def test_has_cycle():
    NO1 = 3
    g = multi(5, NO1, 2)
    assert not g.has_cycle()

    g.get_node("process-0").inputs["input0"] = g.get_node(
        f"process-{NO1-1}"
    ).get_output("output0")
    assert g.has_cycle()
