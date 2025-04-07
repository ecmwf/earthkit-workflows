# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
    """Convert a graph to GraphViz's 'dot' format"""
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
    """Render a graph using GraphViz

    Keyword arguments are passed to `graphviz.Source.render`.
    """
    import graphviz

    dot = to_dot(graph)
    src = graphviz.Source(dot)
    return src.render(**kwargs)
