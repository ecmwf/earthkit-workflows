from .graph import Graph
from .graph.pyvis import to_pyvis, node_info, edge_info

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
    gv = to_pyvis(g, notebook=True, node_attrs=node_info_ext, edge_attrs=edge_info, **kwargs)
    return gv.show(dest)