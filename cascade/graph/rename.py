from functools import reduce
from operator import add
from typing import Callable

from .graph import Graph
from .nodes import Node
from .transform import Transformer


RenamerFunc = Callable[[str], str]


class _Renamer(Transformer):
    func: RenamerFunc

    def __init__(self, func: RenamerFunc):
        self.func = func

    def node(self, n: Node, **inputs: Node.Output) -> Node:
        n.name = self.func(n.name)  # XXX: should we create a copy of n?
        n.inputs = inputs
        return n

    def graph(self, g: Graph, sinks: list[Node]) -> Graph:
        return Graph(sinks)


def rename_nodes(func: RenamerFunc, graph: Graph) -> Graph:
    r = _Renamer(func)
    return r.transform(graph)


def join_namespaced(**graphs: Graph) -> Graph:
    return reduce(
        add,
        (
            rename_nodes((lambda n: f"{namespace}.{n}"), graph)
            for namespace, graph in graphs.items()
        ),
    )
