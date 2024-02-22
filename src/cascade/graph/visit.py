from .graph import Graph
from .nodes import Node, Source, Processor, Sink


class Visitor:
    def visit(self, graph: Graph):
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
