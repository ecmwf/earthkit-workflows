from .graph import Graph
from .nodes import Node, Processor, Sink, Source


class Visitor:
    """Graph visitor base class

    When `visit` is called on a graph, the graph will be visited in arbitrary
    order. A callback method will be called on each node, depending on its type.
    The following callbacks can be defined:
    - ``source(self, n: Node)``
    - ``sink(self, n: Node, **inputs: OutputLike)``
    - ``processor(self, n: Node, **inputs: OutputLike)``
    - ``node(self, n: Node, **inputs: OutputLike)``

    The more specific methods are tried first, then ``node`` is called if no
    specific method was available.
    """

    def visit(self, graph: Graph):
        """Visit the given graph

        See `Visitor` for details.
        """
        for node in graph.nodes():
            self.__visit(node)

    def __visit(self, node: Node):
        if isinstance(node, Source) and hasattr(self, "source"):
            self.source(node)
        elif isinstance(node, Sink) and hasattr(self, "sink"):
            self.sink(node)
        elif isinstance(node, Processor) and hasattr(self, "processor"):
            self.processor(node)
        elif isinstance(node, Node) and hasattr(self, "node"):
            self.node(node)
        assert isinstance(node, (Node, Sink, Processor, Source))
