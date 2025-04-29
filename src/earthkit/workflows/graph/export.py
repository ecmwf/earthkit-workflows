# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import graphlib
import json
from typing import Any, Protocol

from .graph import Graph
from .nodes import Node, Output


class NodeFactory(Protocol):
    def __call__(
        self, name: str, outputs: list[str], payload: Any, **inputs: Output
    ) -> Node:
        pass


def default_node_factory(
    name: str, outputs: list[str], payload: Any, **inputs: Output
) -> Node:
    # NOTE this logic is rather fragile; necessary due to the existence of default output. Remove it and simplify here
    if not outputs:
        return Node(name, payload=payload, outputs=[], **inputs)
    else:
        return Node(name, outputs, payload, **inputs)


def _deserialise_node(
    name: str,
    data: dict,
    node_factory: NodeFactory = default_node_factory,
    **inputs: Output,
) -> "Node":
    payload = data.get("payload", None)
    outputs = data.get("outputs", [])
    return node_factory(name, outputs, payload, **inputs)


def serialise(graph: Graph) -> dict:
    """Convert a graph to a serialisable representation

    See also `Node.serialise`
    """
    data = {}
    for node in graph.nodes():
        assert node.name not in data
        data[node.name] = node.serialise()
    return data


def to_json(graph: Graph) -> str:
    """Serialise a graph as JSON

    See also `serialise`
    """
    return json.dumps(serialise(graph))


def deserialise(data: dict, node_factory: NodeFactory = default_node_factory) -> Graph:
    """Build a graph from a serialisable representation

    An optional node factory function can be provided to create a node with the
    given name, outputs, payload and inputs. The default factory will create
    bare `Node` objects
    """
    deps = {
        name: [
            inp if isinstance(inp, str) else inp[0]
            for inp in node.get("inputs", {}).values()
        ]
        for name, node in data.items()
    }

    ts = graphlib.TopologicalSorter(deps)
    nodes: dict[str, Any] = {}
    sinks: list[Node] = []
    for name in ts.static_order():
        node_data = data[name]
        node_inputs = {}
        for iname, src in node_data.get("inputs", {}).items():
            if isinstance(src, str):
                node_inputs[iname] = nodes[src].get_output()
            else:
                parent, oname = src
                node_inputs[iname] = nodes[parent].get_output(oname)
        nodes[name] = _deserialise_node(
            name, node_data, node_factory=node_factory, **node_inputs
        )
        if (sink := nodes[name]).is_sink():
            sinks.append(sink)
    return Graph(sinks)


def from_json(data: str) -> Graph:
    """Build a graph from JSON

    See also `deserialise`
    """
    return deserialise(json.loads(data))
