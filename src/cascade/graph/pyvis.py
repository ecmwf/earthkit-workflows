import json
from typing import Any, Callable, Literal, ParamSpec, cast

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
    """Simple node info function showing the node inputs and outputs as node title"""
    labels = []
    inputs_s = ", ".join(inp for inp in node.inputs.keys()) if node.inputs else "None"
    labels.append(f"Input{'' if len(node.inputs) == 1 else 's'}: {inputs_s}")
    if node.outputs != [Node.DEFAULT_OUTPUT]:
        outputs_s = ", ".join(node.outputs) if node.outputs else "None"
        labels.append(f"Output{'' if len(node.outputs) == 1 else 's'}: {outputs_s}")
    return {"title": "\n".join(labels)}


def edge_info(sname, snode, dname, dnode):
    """Simple edge info function showing the names of the output and input as edge title"""
    return {"title": f"From: {sname}\nTo: {dname}"}


class VisualisationPresets:
    @staticmethod
    def hierarchical() -> dict:
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

    @staticmethod
    def quick() -> dict:
        return {
            "layout": {
                "randomSeed": 42,
                "improvedLayout": True,
                "hierarchical": {
                    "enabled": False,
                },
            },
            "physics": {
                "enabled": False,
            },
        }

    @staticmethod
    def blob() -> dict:
        return {
            "layout": {
                "randomSeed": 42,
                "improvedLayout": True,
                "hierarchical": {
                    "enabled": False,
                },
            },
            "physics": {
                "forceAtlas2Based": {"gravitationalConstant": -28, "springLength": 100},
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.25,
            },
        }

    @staticmethod
    def none() -> dict:
        return {}


PRESET_OPTIONS = Literal["hierarchical", "quick", "blob", "none"]


def to_pyvis(
    graph: Graph,
    node_attrs: dict | Callable[[Node], dict] | None = None,
    edge_attrs: dict | Callable[[str, Node, str, Node], dict] | None = None,
    preset: PRESET_OPTIONS = "hierarchical",
    options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Network:
    """Convert a graph to a PyVis network

    Parameters
    ----------
    graph: Graph
        Input graph
    node_attrs: None | dict | (Node -> dict)
        Node attributes, or function to set per-node attributes
    edge_attrs: None | dict | ((str, Node, str, Node) -> dict)
        Edge attributes, or function to set per-edge attributes. The function
        arguments are (from_output_name, from_node, to_input_name, to_node)
    options: dict[str, Any] | None, optional
        Options to pass to the `set_options` method of the network.
        Has priority over `hierarchical_layout`
    preset:
        Visualisation preset to use. Options are 'hierarchical', 'blob', 'quick' and 'none'
        Defaults to 'hierarchical'
    **kwargs
        Passed to the `pyvis.Network` constructor

    Returns
    -------
    pyvis.Network
        Output network
    """
    node_func = _make_attr_func(node_attrs)
    edge_func = _make_attr_func(edge_attrs)
    net = Network(directed=True, **kwargs)
    for node in graph.nodes(forwards=True):
        net.add_node(node.name, **node_func(node))
        for iname, isrc in node.inputs.items():
            eattrs = edge_func(str(isrc), isrc.parent, f"{node.name}.{iname}", node)
            net.add_edge(isrc.parent.name, node.name, **eattrs)

    options = options or {}
    preset_options = getattr(VisualisationPresets, preset)()
    preset_options.update(options)
    net.set_options(json.dumps(preset_options))

    return net
