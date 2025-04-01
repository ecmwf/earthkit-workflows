from earthkit.workflows.graph.graph import Graph
from earthkit.workflows.graph.nodes import Node, Output
from earthkit.workflows.graph.transform import Transformer


class _Copier(Transformer):
    def node(self, node: Node, **inputs: Output) -> Node:
        newnode = node.copy()
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return Graph(sinks)


def copy_graph(g: Graph) -> Graph:
    """Create a shallow copy of a whole graph (payloads are not copied)"""
    return _Copier().transform(g)
