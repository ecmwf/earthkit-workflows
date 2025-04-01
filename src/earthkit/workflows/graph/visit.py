# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.workflows.graph.graph import Graph
from earthkit.workflows.graph.nodes import Node


def node_visit(impl: Any, node: Node, inputs: dict[str, Any]):
    """Helper method shared by Visitor and Transformer"""
    if node.is_source() and hasattr(impl, "source"):
        return impl.source(node)
    elif node.is_sink() and hasattr(impl, "sink"):
        return impl.sink(node, **inputs)
    elif node.is_processor() and hasattr(impl, "processor"):
        return impl.processor(node, **inputs)
    elif hasattr(impl, "node"):
        return impl.node(node, **inputs)
    return node


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
            node_visit(self, node, {})
