import textwrap

from . import Graph, Node


def _quote(s: str) -> str:
    res = ['"']
    for c in s:
        if c == "\\":
            res.append("\\\\")
        elif c == '"':
            res.append('\\"')
        else:
            res.append(c)
    res.append('"')
    return "".join(res)


def to_dot(graph: Graph) -> str:
    out = []
    for node in graph.nodes():
        nname = node.name
        for iname, isrc in node.inputs.items():
            pname = isrc.parent.name
            oname = isrc.name
            attrs = {}
            if oname != Node.DEFAULT_OUTPUT:
                attrs["taillabel"] = _quote(oname)
            attrs["headlabel"] = _quote(iname)
            astr = (
                " [" + ", ".join(f"{k}={v}" for k, v in attrs.items()) + "]"
                if attrs
                else ""
            )
            out.append(f"{_quote(pname)} -> {_quote(nname)}{astr}")
    return "digraph {\n" + textwrap.indent("\n".join(out), "  ") + "\n}"


def render_graph(graph: Graph, **kwargs) -> str:
    import graphviz

    dot = to_dot(graph)
    src = graphviz.Source(dot)
    return src.render(**kwargs)
