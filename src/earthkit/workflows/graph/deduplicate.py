# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Callable, cast

from earthkit.workflows.graph.graph import Graph
from earthkit.workflows.graph.nodes import Node, Output
from earthkit.workflows.graph.transform import Transformer

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

    def node(self, node: Node, **inputs: Output) -> Node:
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
        return Graph(cast(list[Node], list(new_sinks)))


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
        Deduplicated graph
    """
    mt = _DedupTransformer(pred)
    return mt.transform(graph)
