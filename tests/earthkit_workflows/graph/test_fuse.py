# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.workflows.graph import Node, fuse_nodes, serialise
from earthkit.workflows.graph.samplegraphs import comb, linear

D = Node.DEFAULT_OUTPUT


def fuse_linear(parent: Node, pout: str, child: Node, cin: str) -> Node | None:
    if not parent.is_processor():
        return None
    if not child.is_processor():
        return None
    if pout != D:
        return None
    if len(parent.outputs) != 1:
        return None
    if len(child.inputs) != 1:
        return None
    payload = []
    if parent.payload is None:
        payload.append(parent.name)
    else:
        payload.extend(parent.payload)
    payload.append(child.name)
    print(f"Fuse {parent.name} ({parent.payload}) with {child.name} -> {payload}")
    return Node(
        f"{parent.name}+{child.name}", child.outputs, payload=payload, **parent.inputs
    )


def test_fuse_linear():
    N = 5
    g = linear(N)
    gf = fuse_nodes(fuse_linear, g)
    pnames = [f"process-{i}" for i in range(N)]
    fname = "+".join(pnames)
    assert serialise(gf) == {
        "reader": {"inputs": {}, "outputs": [D]},
        fname: {"inputs": {"input": "reader"}, "outputs": [D], "payload": pnames},
        "writer": {"inputs": {"input": fname}, "outputs": []},
    }


def fuse_accum(parent: Node, pout: str, child: Node, cin: str) -> Node | None:
    ptype = parent.name.split("-")[0]
    if ptype not in ["join", "accum"]:
        return None
    ctype = child.name.split("-")[0]
    if ctype != "join":
        return None
    if len(child.inputs) != 2:
        return None
    joined: list[Any] = []
    joined.extend(parent.inputs.values())
    joined.extend(isrc for iname, isrc in child.inputs.items() if iname != cin)
    inputs = {f"input{i}": src for i, src in enumerate(joined)}
    return Node(f"accum-{len(joined)}", child.outputs, **inputs)


def test_fuse_comb():
    N = 7
    g = comb(N)
    gf = fuse_nodes(fuse_accum, g)
    assert serialise(gf) == {
        **{f"reader-{i}": {"inputs": {}, "outputs": [D]} for i in range(N)},
        f"accum-{N}": {
            "inputs": {f"input{i}": f"reader-{i}" for i in range(N)},
            "outputs": [D],
        },
        "writer": {"inputs": {"input": f"accum-{N}"}, "outputs": []},
    }


def test_nofuse_comb():
    N = 8
    g = comb(N)
    g.sinks.append(Node("writer-1", outputs=[], input=g.get_node("join-2")))
    gf = fuse_nodes(fuse_accum, g)
    assert serialise(gf) == {
        **{f"reader-{i}": {"inputs": {}, "outputs": [D]} for i in range(N)},
        "accum-4": {
            "inputs": {f"input{i}": f"reader-{i}" for i in range(4)},
            "outputs": [D],
        },
        "accum-5": {
            "inputs": {
                "input0": "accum-4",
                **{f"input{i-3}": f"reader-{i}" for i in range(4, N)},
            },
            "outputs": [D],
        },
        "writer": {"inputs": {"input": "accum-5"}, "outputs": []},
        "writer-1": {"inputs": {"input": "accum-4"}, "outputs": []},
    }
