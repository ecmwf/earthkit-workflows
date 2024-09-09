from typing import Callable, cast

from .graph import Graph, Node, Sink
from .transform import Transformer

PredicateType = Callable[[Node, Node], bool]


def _cmp_nodes(a: Node, b: Node) -> bool:
    if a.outputs != b.outputs:
        return False
    if set(a.inputs.keys()) != set(b.inputs.keys()):
        return False
    for iname in a.inputs:
        ai = a.inputs[iname]
        bi = b.inputs[iname]
        if ai.name != bi.name or ai.parent is not bi.parent:
            return False
    return True


class _DedupTransformer(Transformer):
    pred: PredicateType
    nodes: set[Node]

    def __init__(self, pred: PredicateType):
        self.pred = pred
        self.nodes = set()

    def __find_node(self, node: Node) -> Node | None:
        for other in self.nodes:
            if not _cmp_nodes(node, other):
                continue
            if self.pred(node, other):
                return other
        return None

    def node(self, node: Node, **inputs: Node.Output) -> Node:
        node.inputs = inputs  # XXX: should we create a copy of node?
        other = self.__find_node(node)
        if other is not None:
            return other
        self.nodes.add(node)
        return node

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        new_sinks = set()
        for sink in sinks:
            ref = self.__find_node(sink)
            assert ref is not None
            new_sinks.add(ref)
        return Graph(cast(list[Sink], list(new_sinks)))


def same_payload(a: Node, b: Node):
    return a.payload == b.payload


def deduplicate_nodes(graph: Graph, pred: PredicateType = same_payload) -> Graph:
    """Deduplicate graph nodes

    Two nodes are considered identical if:
    - They have the same outputs
    - They have the same inputs
    - The predicate matches

    Parameters
    ----------
    graph: Graph
        Input graph
    pred: (Node, Node) -> bool, optional
        If set, use this predicate to compare nodes for equality.
        If not set, compare node payloads.

    Returns
    -------
    Graph
        Deduplicated graph"""
    mt = _DedupTransformer(pred)
    return mt.transform(graph)
