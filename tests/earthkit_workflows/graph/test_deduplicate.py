# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from payload_utils import add_payload

from earthkit.workflows.graph import (
    Graph,
    Node,
    deduplicate_nodes,
    expand_graph,
    serialise,
)
from earthkit.workflows.graph.samplegraphs import disconnected, simple

D = Node.DEFAULT_OUTPUT


def test_dedup_disc():
    g = disconnected()
    mg = deduplicate_nodes(g)
    assert len(mg.sinks) == 1
    w = mg.sinks[0]
    assert w.name.startswith("writer-")
    assert w.is_sink()
    p = w.inputs["input"].parent
    assert p.name.startswith("process-")
    assert p.is_processor()
    r = p.inputs["input"].parent
    assert r.name.startswith("reader-")
    assert r.is_source()


def test_no_dedup():
    g = disconnected()
    add_payload(g)
    mg = deduplicate_nodes(g)
    assert g == mg


def test_dedup_part():
    NR1 = 6
    NP1 = 2
    NR2 = 5
    NP2 = 3
    g1 = simple(NR1, NP1)
    g2 = simple(NR2, NP2)
    add_payload(g1)
    add_payload(g2)

    def expander(node: Node) -> Graph | None:
        return {"g1": g1, "g2": g2}.get(node.name, None)

    templ = Graph([Node("g1"), Node("g2")])
    g = expand_graph(expander, templ)
    mg = deduplicate_nodes(g)
    s = serialise(mg)
    exp = {}
    p1i = {}
    p1i.update({f"input{j}": f"g2.reader-{j}" for j in range(min(NR1, NR2))})
    p1i.update({f"input{j}": f"g1.reader-{j}" for j in range(min(NR1, NR2), NR1)})
    p2i = {}
    p2i.update({f"input{j}": f"g2.reader-{j}" for j in range(NR2)})
    for i in range(NP1):
        exp[f"g1.writer-{i}"] = {
            "inputs": {"input": f"g1.process-{i}"},
            "outputs": [],
            "payload": i,
        }
        exp[f"g1.process-{i}"] = {
            "inputs": p1i,
            "outputs": [D],
            "payload": i,
        }
    for i in range(NP2):
        exp[f"g2.writer-{i}"] = {
            "inputs": {"input": f"g2.process-{i}"},
            "outputs": [],
            "payload": i,
        }
        exp[f"g2.process-{i}"] = {
            "inputs": p2i,
            "outputs": [D],
            "payload": i,
        }
    for i in range(min(NR1, NR2)):
        exp[f"g2.reader-{i}"] = {"inputs": {}, "outputs": [D], "payload": i}
    for i in range(min(NR1, NR2), NR1):
        exp[f"g1.reader-{i}"] = {"inputs": {}, "outputs": [D], "payload": i}
    assert s == exp
