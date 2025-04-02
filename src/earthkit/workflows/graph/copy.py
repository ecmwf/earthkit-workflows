# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
