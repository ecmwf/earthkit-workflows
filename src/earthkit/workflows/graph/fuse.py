# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import Counter
from typing import Callable

from .graph import Graph
from .nodes import Node, Output
from .transform import Transformer

FuseCallback = Callable[[Node, str, Node, str], Node | None]


class _FuseTransformer(Transformer):
    func: FuseCallback
    counter: Counter[Node] | None

    def __init__(self, func: FuseCallback):
        self.func = func
        self.counter = None

    def node(self, node: Node, **inputs: Output) -> Node:
        assert self.counter is not None
        result = node
        any_fused = False
        for iname, isrc in inputs.items():
            if self.counter[isrc.parent] > 1:
                continue
            fused = self.func(isrc.parent, isrc.name, result, iname)
            if fused is None:
                continue
            any_fused = True
            result = fused
        if any_fused:
            self.counter[result] = self.counter[node]
        else:
            result.inputs = inputs  # XXX: should we create a copy of result/node?
        return result

    def graph(self, g: Graph, sinks: list[Node]) -> Graph:
        return Graph(sinks)

    def transform(self, graph: Graph) -> Graph:
        self.counter = Counter(
            isrc.parent for node in graph.nodes() for isrc in node.inputs.values()
        )
        return super().transform(graph)


def fuse_nodes(func: FuseCallback, g: Graph) -> Graph:
    """Fuse compatible nodes of a graph

    Candidates for fusion are 4-tuples: (parent node, parent output, current
    node, current input). One such tuple is considered only if the parent node
    has no other child. The current node may have multiple inputs to consider.
    In that case, they will be considered in order (after a successful fusion,
    the callback is passed the fused node).

    Parameters
    ----------
    func: (Node, str, Node, str) -> (Node | None)
        Fusion callback. If ``func(parent, parent_out, current, current_in)``
        returns a node, use it to replace the current node and its parent. If it
        returns None, do not replace the current node.
    g: Graph
        Input graph

    Returns
    -------
    Graph
        Fused graph
    """
    return _FuseTransformer(func).transform(g)
