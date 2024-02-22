import json
from typing import Any, Callable, ParamSpec, cast

from pyvis.network import Network

from .graph import Graph
from .nodes import Node


_P = ParamSpec("_P")


def _make_attr_func(attr: dict | Callable[_P, dict] | None) -> Callable[_P, dict]:
    if attr is None:
        return cast(Callable[_P, dict], lambda *args: {})
    if isinstance(attr, dict):
        return cast(Callable[_P, dict], lambda *args: attr)
    return attr


def node_info(node):
    labels = []
    inputs_s = ", ".join(inp for inp in node.inputs.keys()) if node.inputs else "None"
    labels.append(f"Input{'' if len(node.inputs) == 1 else 's'}: {inputs_s}")
    if node.outputs != [Node.DEFAULT_OUTPUT]:
        outputs_s = ", ".join(node.outputs) if node.outputs else "None"
        labels.append(f"Output{'' if len(node.outputs) == 1 else 's'}: {outputs_s}")
    return {"title": "\n".join(labels)}


def edge_info(sname, snode, dname, dnode):
    return {"title": f"From: {sname}\nTo: {dname}"}


def hierarchical_layout_options() -> dict:
    return {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "shakeTowards": "roots",
            }
        }
    }


def to_pyvis(
    graph: Graph,
    node_attrs: dict | Callable[[Node], dict] | None = None,
    edge_attrs: dict | Callable[[str, Node, str, Node], dict] | None = None,
    hierarchical_layout: bool = True,
    **kwargs: Any,
) -> Network:
    node_func = _make_attr_func(node_attrs)
    edge_func = _make_attr_func(edge_attrs)
    net = Network(directed=True, **kwargs)
    for node in graph.nodes(forwards=True):
        net.add_node(node.name, **node_func(node))
        for iname, isrc in node.inputs.items():
            eattrs = edge_func(str(isrc), isrc.parent, f"{node.name}.{iname}", node)
            net.add_edge(isrc.parent.name, node.name, **eattrs)
    if hierarchical_layout:
        net.set_options(json.dumps(hierarchical_layout_options()))
    return net
