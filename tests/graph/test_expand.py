from cascade.graph import (
    Graph,
    Node,
    Processor,
    Sink,
    Source,
    Splicer,
    expand_graph,
    serialise,
)
from cascade.graph.expand import _Subgraph
from cascade.graph.samplegraphs import disconnected, linear, multi, simple

D = Node.DEFAULT_OUTPUT


def test_splice_linear():
    inp = Source("reader")
    g = linear(2)
    sp = Splicer("main", {"reader": inp.get_output()}, None, ["writer"], None)
    spliced: _Subgraph = sp.transform(g)
    assert list(spliced.leaves.keys()) == ["writer"]
    out = spliced.get_output("writer")
    w = out.parent
    assert w.name == "main.writer"
    assert isinstance(w, Processor)
    p1 = w.inputs["input"].parent
    assert p1.name == "main.process-1"
    p0 = p1.inputs["input"].parent
    assert p0.name == "main.process-0"
    r = p0.inputs["input"].parent
    assert r.name == "main.reader"
    assert isinstance(r, Processor)
    i = r.inputs["input"].parent
    assert i == inp


def test_splice_disc():
    N = 5
    inps = [Source(f"reader-{i}") for i in range(N)]
    g = disconnected(N)
    sp = Splicer(
        "main",
        {f"input{i}": inp for i, inp in enumerate(inps)},
        {f"reader-{i}": f"input{i}" for i in range(N)},
        [f"output{i}" for i in range(N)],
        {f"output{i}": f"writer-{i}" for i in range(N)},
    )
    spliced: _Subgraph = sp.transform(g)
    assert list(spliced.leaves.keys()) == [f"writer-{i}" for i in range(N)]
    for i in range(N):
        out = spliced.get_output(f"output{i}")
        w = out.parent
        assert isinstance(w, Processor)
        assert w.name == f"main.writer-{i}"
        p = w.inputs["input"].parent
        assert p.name == f"main.process-{i}"
        r = p.inputs["input"].parent
        assert r.name == f"main.reader-{i}"
        assert isinstance(r, Processor)
        inp = r.inputs["input"].parent
        assert inp == inps[i]


def test_expand_linear():
    def linear_expander(
        node: Node,
    ) -> tuple[Graph, dict[str, str], dict[str, str]] | None:
        if not node.name.startswith("process-"):
            return None
        sg = linear(2 + int(node.name.split("-")[1]))
        inp_map = {"reader": "input"}
        out_map = {D: "writer"}
        return sg, inp_map, out_map

    NC = 3
    coarse = linear(NC)
    expanded = expand_graph(linear_expander, coarse)
    assert len(expanded.sinks) == 1
    wr = expanded.sinks[0]
    assert wr.name == "writer"
    assert isinstance(wr, Sink)
    tip = wr
    for i in range(NC - 1, -1, -1):
        w = tip.inputs["input"].parent
        assert isinstance(w, Processor)
        assert w.name == f"process-{i}.writer"
        cur = w
        for j in range(i + 1, -1, -1):
            p = cur.inputs["input"].parent
            assert isinstance(p, Processor)
            assert p.name == f"process-{i}.process-{j}"
            cur = p
        r = cur.inputs["input"].parent
        assert isinstance(r, Processor)
        assert r.name == f"process-{i}.reader"
        tip = r
    rd = tip.inputs["input"].parent
    assert rd.name == "reader"
    assert isinstance(rd, Source)


def test_expand_multi():
    def expander(node: Node) -> tuple[Graph, dict[str, str], dict[str, str]] | None:
        if node.name.startswith("reader-"):
            return linear(2), None, {D: "writer"}
        if node.name.startswith("process-"):
            if node.name == "process-0":
                nread = len(node.inputs)
                nproc = len(node.outputs)
                sg = simple(nread, nproc)
                imap = {f"reader-{i}": f"input{i}" for i in range(nread)}
                omap = {f"output{i}": f"writer-{i}" for i in range(nproc)}
                return sg, imap, omap
        if node.name.startswith("writer-"):
            i1 = Source("input1")
            i2 = Source("input2")
            w = Sink("writer", input1=i1, input2=i2)
            return Graph([w])
        return None

    NR = 5
    NO1 = 3
    NO2 = 2
    coarse = multi(NR, NO1, NO2)
    expanded = expand_graph(expander, coarse)
    s = serialise(expanded)
    exp = {}
    for i in range(NR):
        exp[f"reader-{i}.reader"] = {"inputs": {}, "outputs": [D]}
        exp[f"reader-{i}.process-0"] = {
            "inputs": {"input": f"reader-{i}.reader"},
            "outputs": [D],
        }
        exp[f"reader-{i}.process-1"] = {
            "inputs": {"input": f"reader-{i}.process-0"},
            "outputs": [D],
        }
        exp[f"reader-{i}.writer"] = {
            "inputs": {"input": f"reader-{i}.process-1"},
            "outputs": [D],
        }

        exp[f"process-0.reader-{i}"] = {
            "inputs": {f"input": f"reader-{i}.writer"},
            "outputs": [D],
        }
    for i in range(NO1):
        exp[f"process-0.process-{i}"] = {
            "inputs": {f"input{j}": f"process-0.reader-{j}" for j in range(NR)},
            "outputs": [D],
        }
        exp[f"process-0.writer-{i}"] = {
            "inputs": {"input": f"process-0.process-{i}"},
            "outputs": [D],
        }
    exp["process-1"] = {
        "inputs": {"input1": "process-0.writer-0", "input2": "process-0.writer-2"},
        "outputs": [D],
    }
    exp["process-2"] = {
        "inputs": {"input1": "process-0.writer-1", "input2": "reader-2.writer"},
        "outputs": [f"output{i}" for i in range(NO2)],
    }
    for i in range(NO2):
        exp[f"writer-{i}.input1"] = {"inputs": {"input": "process-1"}, "outputs": [D]}
        exp[f"writer-{i}.input2"] = {
            "inputs": {"input": ("process-2", f"output{i}")},
            "outputs": [D],
        }
        exp[f"writer-{i}.writer"] = {
            "inputs": {"input1": f"writer-{i}.input1", "input2": f"writer-{i}.input2"},
            "outputs": [],
        }
    assert s == exp
