from cascade.graph import (
    expand_graph,
    deduplicate_nodes,
    serialise,
    Graph,
    Node,
    Processor,
    Sink,
    Source,
)
from cascade.graph.samplegraphs import disconnected, simple

from payload_utils import add_payload


D = Node.DEFAULT_OUTPUT


def test_dedup_disc():
    g = disconnected()
    mg = deduplicate_nodes(g)
    assert len(mg.sinks) == 1
    w = mg.sinks[0]
    assert w.name.startswith("writer-")
    assert isinstance(w, Sink)
    p = w.inputs["input"].parent
    assert p.name.startswith("process-")
    assert isinstance(p, Processor)
    r = p.inputs["input"].parent
    assert r.name.startswith("reader-")
    assert isinstance(r, Source)


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

    templ = Graph([Sink("g1"), Sink("g2")])
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
