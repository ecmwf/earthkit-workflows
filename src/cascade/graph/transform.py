from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Graph

from .nodes import Node, Source, Processor, Sink


class Transformer:
    def transform(self, graph: "Graph") -> Any:
        done: dict[Node, dict[str, Any]] = {}
        todo: list[Node] = [sink for sink in graph.sinks]

        while todo:
            node = todo[-1]
            if node in done:
                todo.pop()
                continue

            inputs = {}
            complete = True
            for iname, isrc in node.inputs.items():
                inode = isrc.parent
                if inode not in done:
                    todo.append(inode)
                    complete = False
                    break

                inputs[iname] = self.__transform_output(done[inode], isrc)

            if not complete:
                continue

            transformed = self.__transform(node, inputs)
            done[node] = transformed
            todo.pop()

        return self.__transform_graph(graph, [done[onode] for onode in graph.sinks])

    def __transform(self, node: Node, inputs: dict[str, Any]) -> Any:
        if isinstance(node, Source) and hasattr(self, "source"):
            return self.source(node)
        if isinstance(node, Sink) and hasattr(self, "sink"):
            return self.sink(node, **inputs)
        if isinstance(node, Processor) and hasattr(self, "processor"):
            return self.processor(node, **inputs)
        if isinstance(node, Node) and hasattr(self, "node"):
            return self.node(node, **inputs)
        assert isinstance(node, (Node, Sink, Processor, Source))
        return node

    def __transform_output(self, node: Any, output: Node.Output) -> Any:
        if hasattr(self, "output"):
            return self.output(node, output.name)
        if isinstance(node, dict):
            return node[output.name]
        try:
            return getattr(node, output.name)
        except AttributeError:
            return (node, output)

    def __transform_graph(self, graph: "Graph", sinks: list) -> Any:
        if hasattr(self, "graph"):
            return self.graph(graph, sinks)
        return sinks
