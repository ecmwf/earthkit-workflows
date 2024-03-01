from .graph import Graph
from .nodes import Node, Sink
from .transform import Transformer


class _Copier(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Node:
        newnode = node.copy()
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> Graph:
        return Graph(sinks)


def copy_graph(g: Graph) -> Graph:
    """Create a shallow copy of a whole graph (payloads are not copied)"""
    return _Copier().transform(g)
