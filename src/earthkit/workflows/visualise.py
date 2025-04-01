from typing import Callable

from .graph import Graph, Node
from .graph.pyvis import PRESET_OPTIONS, edge_info, node_info, to_pyvis
from .taskgraph import Task


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
        if isinstance(node, Task):
            t.append(f"Duration: {node.duration}")
            t.append(f"Memory: {node.memory}")
        info["title"] = "\n".join(t)

    return info


def visualise(
    g: Graph,
    dest: str,
    node_attrs: dict | Callable[[Node], dict] | None = node_info_ext,
    edge_attrs: dict | Callable[[str, Node, str, Node], dict] | None = edge_info,
    preset: PRESET_OPTIONS = "hierarchical",
    **kwargs,
):
    """Visualise a graph with PyVis

    Parameters
    ----------
    g: Graph
        Input graph
    dest: str
        Path to the generated HTML file
    preset: str
        Name of the preset to use for network options.
        Can be 'hierarchical', 'quick', 'blob', or 'none'.
    **kwargs
        Passed to the `pyvis.Network` constructor

    Returns
    -------
    IFrame
        Jupyter IFrame to visualise the graph
    """
    gv = to_pyvis(
        g,
        notebook=True,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        preset=preset,
        **kwargs,
    )
    return gv.show(dest)
