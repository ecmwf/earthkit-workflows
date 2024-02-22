import graphlib
import json
from typing import Any, Protocol

from .graph import Graph
from .nodes import Node, Processor, Sink, Source


class NodeFactory(Protocol):
    def __call__(
        self, name: str, outputs: list[str], payload: Any, **inputs: Node.Output
    ) -> Node:
        pass


def default_node_factory(
    name: str, outputs: list[str], payload: Any, **inputs: Node.Output
) -> Node:
    if inputs and outputs:
        return Processor(name, outputs, payload, **inputs)
    if not outputs:
        return Sink(name, payload, **inputs)
    return Source(name, outputs, payload)


def _deserialise_node(
    name: str,
    data: dict,
    node_factory: NodeFactory = default_node_factory,
    **inputs: Node.Output
) -> "Node":
    payload = data.get("payload", None)
    outputs = data.get("outputs", [])
    return node_factory(name, outputs, payload, **inputs)


def serialise(graph: Graph) -> dict:
    data = {}
    for node in graph.nodes():
        assert node.name not in data
        data[node.name] = node.serialise()
    return data


def to_json(graph: Graph) -> str:
    return json.dumps(serialise(graph))


def deserialise(data: dict, node_factory: NodeFactory = default_node_factory) -> Graph:
    deps = {}
    for name, node in data.items():
        deps[name] = []
        for inp in node.get("inputs", {}).values():
            if isinstance(inp, str):
                deps[name].append(inp)
            else:
                deps[name].append(inp[0])
    ts = graphlib.TopologicalSorter(deps)
    nodes = {}
    sinks = []
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
        if isinstance(nodes[name], Sink):
            sinks.append(nodes[name])
    return Graph(sinks)


def from_json(data: str) -> Graph:
    return deserialise(json.loads(data))
