from cascade.graph import Graph, Node, Processor, Sink, Source, Visitor
from cascade.graph.samplegraphs import disconnected, empty, linear, multi, simple


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
    sources: list[Source]
    others: list[Node]

    def __init__(self):
        self.sources = []
        self.others = []

    def source(self, s: Source):
        self.sources.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_sources():
    g = simple(6, 2)
    v = SourceLister()
    v.visit(g)
    for s in v.sources:
        assert isinstance(s, Source)
    for n in v.others:
        assert not isinstance(n, Source)
    snames = [s.name for s in v.sources]
    assert len(snames) == 6
    assert set(snames) == set(f"reader-{i}" for i in range(6))
    onames = [n.name for n in v.others]
    assert len(onames) == 2 + 2
    assert set(onames) == set(
        [f"process-{i}" for i in range(2)] + [f"writer-{i}" for i in range(2)]
    )


class ProcessorLister(Visitor):
    procs: list[Processor]
    others: list[Node]

    def __init__(self):
        self.procs = []
        self.others = []

    def processor(self, p: Processor):
        self.procs.append(p)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_processors():
    g = simple(7, 4)
    v = ProcessorLister()
    v.visit(g)
    for p in v.procs:
        assert isinstance(p, Processor)
    for n in v.others:
        assert not isinstance(n, Processor)
    pnames = [p.name for p in v.procs]
    assert len(pnames) == 4
    assert set(pnames) == set(f"process-{i}" for i in range(4))
    onames = [n.name for n in v.others]
    assert len(onames) == 7 + 4
    assert set(onames) == set(
        [f"reader-{i}" for i in range(7)] + [f"writer-{i}" for i in range(4)]
    )


class SinkLister(Visitor):
    sinks: list[Sink]
    others: list[Processor]

    def __init__(self):
        self.sinks = []
        self.others = []

    def sink(self, s: Sink):
        self.sinks.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_sinks():
    g = simple(9, 5)
    v = SinkLister()
    v.visit(g)
    for s in v.sinks:
        assert isinstance(s, Sink)
    for n in v.others:
        assert not isinstance(n, Sink)
    snames = [s.name for s in v.sinks]
    assert len(snames) == 5
    assert set(snames) == set(f"writer-{i}" for i in range(5))
    onames = [n.name for n in v.others]
    assert len(onames) == 9 + 5
    assert set(onames) == set(
        [f"reader-{i}" for i in range(9)] + [f"process-{i}" for i in range(5)]
    )


class SegregatedLister(Visitor):
    sources: list[Source]
    procs: list[Processor]
    sinks: list[Sink]
    others: list[Node]

    def __init__(self):
        self.sources = []
        self.procs = []
        self.sinks = []
        self.others = []

    def source(self, s: Source):
        self.sources.append(s)

    def processor(self, p: Processor):
        self.procs.append(p)

    def sink(self, s: Sink):
        self.sinks.append(s)

    def node(self, n: Node):
        self.others.append(n)


def test_visit_segregated():
    g = simple(8, 3)
    v = SegregatedLister()
    v.visit(g)
    for p in v.sources:
        assert isinstance(p, Source)
    for p in v.procs:
        assert isinstance(p, Processor)
    for p in v.sinks:
        assert isinstance(p, Sink)
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
