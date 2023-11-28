from . import Graph, Processor, Sink, Source


def empty() -> Graph:
    """Empty graph"""
    return Graph([])


def linear(nproc: int = 5) -> Graph:
    """Linear graph

    reader -> process-0 -> ... -> process-{nproc-1} -> writer
    """
    r = Source("reader")
    p = r
    for i in range(nproc):
        p = Processor(f"process-{i}", input=p)
    w = Sink("writer", input=p)
    return Graph([w])


def disconnected(nchains: int = 5, nproc: int = 1) -> Graph:
    """Disconnected graph

    reader-0 -> process-0.0 -> ... -> process-0.{nproc-1} -> writer-0
    :
    reader-{nchains-1} -> process-{nchains-1}.0 -> ... -> process-{nchains-1}.{nproc-1} -> writer-{nchains-1}

    If nproc is 1 the `.0` suffix of processors is omitted
    """
    rs = [Source(f"reader-{i}") for i in range(nchains)]
    ps = rs
    for j in range(nproc):
        suffix = "" if nproc == 1 else f".{j}"
        ps = [Processor(f"process-{i}{suffix}", input=p) for i, p in enumerate(ps)]
    ws = [Sink(f"writer-{i}", input=p) for i, p in enumerate(ps)]
    return Graph(ws)


def simple(nread: int = 5, nproc: int = 3) -> Graph:
    """Simple graph

    `nread` sources (reader-{i} for i in range(nread))
    `nproc` processors reading from all readers (process-{i} for i in range(nproc))
    `nproc` writers writing the corresponding processor output (writer-{i} for i in range(nproc))
    """
    rs = [Source(f"reader-{i}") for i in range(nread)]
    ps = []
    ws = []
    for i in range(nproc):
        pi = {f"input{j}": r for j, r in enumerate(rs)}
        p = Processor(f"process-{i}", **pi)
        ps.append(p)
        ws.append(Sink(f"writer-{i}", input=p))
    return Graph(ws)


def multi(nread: int = 5, nout1: int = 3, nout2: int = 2) -> Graph:
    """Graph with multi-output nodes

    `nread` readers (reader-{i} for i in range(nread))
    `nout1` processors (process-{i} for i in range(nout1), nout1 must be at least 3)
    `nout2 * (nout1 - 2)` writers (writer-{i} for i in range(...))

    process-0 reads from all readers, has `nout` outputs
    process-{i} (i in range(1, nout1-1)) reads process-0's output0 and output{i+1}
    process-{nout1-1} reads process0's output1 and one of the readers (third, or last if nread<3)
    writer-* read from one of the process-{i} (i in range(1, nout1-1)) and one output of process-{nout1-1}
    """
    assert nout1 >= 3
    rs = [Source(f"reader-{i}") for i in range(nread)]
    p0i = {f"input{i}": r for i, r in enumerate(rs)}
    p0 = Processor("process-0", outputs=[f"output{i}" for i in range(nout1)], **p0i)
    p1s = []
    for i in range(2, nout1):
        p = Processor(
            f"process-{i-1}", input1=p0.output0, input2=p0.get_output(f"output{i}")
        )
        p1s.append(p)
    p2 = Processor(
        f"process-{nout1-1}",
        outputs=[f"output{i}" for i in range(nout2)],
        input1=p0.output1,
        input2=rs[min(2, nread - 1)],
    )
    ws = []
    for i in range(nout2):
        for p1 in p1s:
            w = Sink(f"writer-{len(ws)}", input1=p1, input2=p2.get_output(f"output{i}"))
            ws.append(w)
    return Graph(ws)


def comb(nteeth: int = 5, nproc: int = 0):
    """Comb graph

    `nteeth` readers (reader-{i} for i in range(nteeth))
    `(nproc + 1) * nteeth - 1` processors (
        process-{i}.{j} for i in range(nteeth) for j in range(nproc),
        join-{i} for i in range(nteeth - 1)
    )
    1 writer (writer)

    Each tooth has the form reader-{i} -> process-{i}.0 -> ... -> process-{i}.{nproc-1}
    Teeth are joined by processors: tooth-0 -> join-0.input1
    join-{i} -> join-{i+1}.input1 / tooth-{i+1} -> join-{i}.input2
    where tooth-{i} is reader-{i} if nproc is 0, else process-{i}.{nproc-1}
    The last join feeds into the writer: join-{nteeth - 2} -> writer
    """
    tip = None
    for i in range(nteeth):
        tooth = Source(f"reader-{i}")
        for j in range(nproc):
            tooth = Processor(f"process-{i}.{j}", input=tooth)
        if tip is None:
            tip = tooth
        else:
            tip = Processor(f"join-{i-1}", input1=tip, input2=tooth)
    assert tip is not None
    return Graph([Sink("writer", input=tip)])
