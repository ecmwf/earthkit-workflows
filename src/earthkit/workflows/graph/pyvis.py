# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import random
from typing import Any, Callable, Literal, ParamSpec, cast

from pyvis.network import Network

from .graph import Graph
from .nodes import Node

_P = ParamSpec("_P")


AVAILABLE_SHAPES = ["box", "dot", "triangleDown", "square"]
AVAILABLE_COLOURS = [
    "#6ba472",
    "#726bf9",
    "#fefc6e",
    "#fb69b9",
    "#74fbf1",
    "#c1e7b4",
    "#d68569",
    "#c8a5f4",
    "#9c6ca5",
    "#8afe6f",
    "#79aacc",
    "#73f8ae",
    "#fec3a5",
    "#f76cf8",
    "#c4d67c",
    "#bef7f0",
    "#986e6f",
    "#c26dd2",
    "#98b29b",
    "#89bffe",
    "#fdb1d9",
    "#d0979b",
    "#f8fbc4",
    "#69899c",
    "#f6ba73",
    "#c1b2c9",
    "#6c69c8",
    "#f0698b",
    "#92dcd3",
    "#a3ae6c",
    "#72d385",
    "#d8fe93",
]


def _make_attr_func(attr: dict | Callable[_P, dict] | None) -> Callable[_P, dict]:
    if attr is None:
        return cast(Callable[_P, dict], lambda *args: {})
    if isinstance(attr, dict):
        return cast(Callable[_P, dict], lambda *args: attr)
    return attr


def node_info(node):
    """Simple node info function showing the node inputs and outputs as node title.

    Additionally uses function name as the seed to select the shape and colour.
    """
    labels = []
    inputs_s = ", ".join(inp for inp in node.inputs.keys()) if node.inputs else "None"
    labels.append(f"Input{'' if len(node.inputs) == 1 else 's'}: {inputs_s}")
    if node.outputs != [Node.DEFAULT_OUTPUT]:
        outputs_s = ", ".join(node.outputs) if node.outputs else "None"
        labels.append(f"Output{'' if len(node.outputs) == 1 else 's'}: {outputs_s}")

    shape = random.Random(node.name.split(":")[0]).choice(AVAILABLE_SHAPES)
    colour = random.Random(f"{node.name.split(':')[0]}-{str(node.outputs)}").choice(
        AVAILABLE_COLOURS
    )

    return {
        "title": "\n".join(labels),
        "shape": shape,
        "color": colour,
    }


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


def truncate_name(name: str, max_length: int = 10) -> str:
    """Truncate a name to a maximum length"""

    if ":" not in name:
        return name

    func, id = name.split(":", maxsplit=1)
    if len(id) <= max_length:
        return name

    return f"{func}:{id[:max_length]}"


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
        net.add_node(truncate_name(node.name), **node_func(node))
        for iname, isrc in node.inputs.items():
            eattrs = edge_func(
                str(isrc),
                isrc.parent,
                f"{truncate_name(node.name)}.{truncate_name(iname)}",
                node,
            )
            net.add_edge(
                truncate_name(isrc.parent.name), truncate_name(node.name), **eattrs
            )

    options = options or {}
    preset_options = getattr(VisualisationPresets, preset)()
    preset_options.update(options)
    net.set_options(json.dumps(preset_options))

    return net
