from typing import Callable

from pyvis.network import Network

from .contextgraph import ContextGraph, Processor, Communicator
from .graph import Graph
from .graph.pyvis import _make_attr_func, to_pyvis, node_info, edge_info


def node_info_ext(node):
    info = node_info(node)
    info["color"] = "#648FFF"
    if not node.inputs:
        info["shape"] = "diamond"
        info["color"] = "#DC267F"
    elif not node.outputs:
        info["shape"] = "triangle"
        info["color"] = "#FFB000"
    if node.payload is not None:
        t = []
        if "title" in info:
            t.append(info["title"])
        func, args, kwargs = node.payload
        t.append(f"Function: {func}")
        if args:
            t.append("Arguments:")
            t.extend(f"- {arg!r}" for arg in args)
        if kwargs:
            t.append("Keyword arguments:")
            t.extend(f"- {k!r}: {v!r}" for k, v in kwargs.items())
        info["title"] = "\n".join(t)

    return info


def visualise(g: Graph, dest: str, **kwargs):
    """Visualise a graph with PyVis

    Parameters
    ----------
    g: Graph
        Input graph
    dest: str
        Path to the generated HTML file
    **kwargs
        Passed to the `pyvis.Network` constructor

    Returns
    -------
    IFrame
        Jupyter IFrame to visualise the graph
    """
    gv = to_pyvis(
        g, notebook=True, node_attrs=node_info_ext, edge_attrs=edge_info, **kwargs
    )
    return gv.show(dest)


def cg_proc_info(proc: Processor) -> dict:
    return {"title": f"Type: {proc.type}\nSpeed: {proc.speed}\nMemory: {proc.memory}"}


def cg_comm_info(comm: Communicator) -> dict:
    return {
        "title": f"From: {comm.source}\nTo: {comm.target}\nBandwidth: {comm.bandwidth}\nLatency: {comm.latency}"
    }


def visualise_contextgraph(
    g: ContextGraph,
    dest: str,
    proc_attrs: dict | Callable[[Processor], dict] | None = cg_proc_info,
    comm_attrs: dict | Callable[[Communicator], dict] | None = cg_comm_info,
    **kwargs,
):
    """Visualise a context graph with PyVis

    Parameters
    ----------
    g: ContextGraph
        Input context graph
    dest: str
        Path to the generated HTML file
    **kwargs
        Passed to the `pyvis.Network` constructor

    Returns
    -------
    IFrame
        Jupyter IFrame to visualise the graph
    """
    proc_func = _make_attr_func(proc_attrs)
    comm_func = _make_attr_func(comm_attrs)
    gv = Network(directed=True, notebook=True, **kwargs)
    for proc in g.nodes:
        gv.add_node(proc.name, **proc_func(proc))
    for comm in g.communicators():
        gv.add_edge(comm.source, comm.target, **comm_func(comm))
    return gv.show(dest)
