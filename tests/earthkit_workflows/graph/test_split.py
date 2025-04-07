# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import astuple

from earthkit.workflows.graph import Node, serialise, split_graph
from earthkit.workflows.graph.samplegraphs import disconnected, multi

D = Node.DEFAULT_OUTPUT


def test_split_trivial():
    def splitter(node: Node) -> int:
        return int(node.name.rsplit("-", 1)[1])

    N = 5
    g = disconnected(N)
    parts, cuts = split_graph(splitter, g)
    assert set(parts.keys()) == set(range(N))
    for i in range(N):
        assert serialise(parts[i]) == {
            f"reader-{i}": {"inputs": {}, "outputs": [D]},
            f"process-{i}": {"inputs": {"input": f"reader-{i}"}, "outputs": [D]},
            f"writer-{i}": {"inputs": {"input": f"process-{i}"}, "outputs": []},
        }
    assert cuts == []


def test_split_multi():
    def splitter(node: Node) -> int:
        if node.name.startswith("reader-"):
            return 0
        num = int(node.name.rsplit("-", 1)[1])
        if node.name.startswith("process-"):
            return num
        return num + 1

    NR = 5
    NO1 = 3
    g = multi(NR, 3, 2)
    parts, cuts = split_graph(splitter, g)
    assert len(cuts) == 6
    assert set(astuple(cut) for cut in cuts) == {
        (0, "process-0", "output0", 1, "process-1", "input1"),
        (0, "process-0", "output2", 1, "process-1", "input2"),
        (0, "process-0", "output1", 2, "process-2", "input1"),
        (0, "reader-2", D, 2, "process-2", "input2"),
        (1, "process-1", D, 2, "writer-1", "input1"),
        (2, "process-2", "output0", 1, "writer-0", "input2"),
    }
    cuts_from = {}
    cuts_to = {}
    for cut in cuts:
        cuts_from.setdefault(cut.source_key, []).append(cut)
        cuts_to.setdefault(cut.dest_key, []).append(cut)
    assert set(parts.keys()) == {0, 1, 2}

    s0 = serialise(parts[0])
    print(s0)
    for i in range(NR):
        assert s0.pop(f"reader-{i}") == {"inputs": {}, "outputs": [D]}
    assert s0.pop("process-0") == {
        "inputs": {f"input{i}": f"reader-{i}" for i in range(NR)},
        "outputs": [f"output{i}" for i in range(NO1)],
    }
    assert s0 == {
        cut.name: {"inputs": {"input": cut.source}, "outputs": []}
        for cut in cuts_from[0]
    }

    s1 = serialise(parts[1])
    p1_inp = {}
    w0_inp = {"input1": "process-1"}
    for cut in cuts_to[1]:
        assert s1.pop(cut.name) == {"inputs": {}, "outputs": [D]}
        if cut.dest_node == "process-1":
            p1_inp[cut.dest_input] = cut.name
        elif cut.dest_node == "writer-0":
            w0_inp[cut.dest_input] = cut.name
    assert s1.pop("process-1") == {"inputs": p1_inp, "outputs": [D]}
    assert s1.pop("writer-0") == {"inputs": w0_inp, "outputs": []}
    assert s1 == {
        cut.name: {"inputs": {"input": cut.source}, "outputs": []}
        for cut in cuts_from[1]
    }

    s2 = serialise(parts[2])
    p2_inp = {}
    w1_inp = {"input2": ("process-2", "output1")}
    for cut in cuts_to[2]:
        assert s2.pop(cut.name) == {"inputs": {}, "outputs": [D]}
        if cut.dest_node == "process-2":
            p2_inp[cut.dest_input] = cut.name
        elif cut.dest_node == "writer-1":
            w1_inp[cut.dest_input] = cut.name
    assert s2.pop("process-2") == {"inputs": p2_inp, "outputs": ["output0", "output1"]}
    assert s2.pop("writer-1") == {"inputs": w1_inp, "outputs": []}
    assert s2 == {
        cut.name: {"inputs": {"input": cut.source}, "outputs": []}
        for cut in cuts_from[2]
    }
