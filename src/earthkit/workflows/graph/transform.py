# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.workflows.graph.graph import Graph
from earthkit.workflows.graph.nodes import Node, Output
from earthkit.workflows.graph.visit import node_visit


class Transformer:
    """Graph transformer base class

    When `transform` is called on a graph, the graph will be visited in
    topological order. A callback method will be called on each node, and the
    result will be used as a replacement. The following callbacks can be
    defined:
    - ``source(self, n: Node) -> NodeLike``
    - ``sink(self, n: Node, **inputs: OutputLike) -> NodeLike``
    - ``processor(self, n: Node, **inputs: OutputLike) -> NodeLike``
    - ``node(self, n: Node, **inputs: OutputLike) -> NodeLike``
    - ``output(self, n: NodeLike, name: str) -> OutputLike``
    - ``graph(self, g: Graph, sinks: list[NodeLike]) -> GraphLike``

    The nodes are transformed as follows, trying each option in order until one
    succeeds:
    - call ``source``, if the node is a source,
    - call ``sink``, if the node is a sink,
    - call ``processor``, if the node is a processor,
    - call ``node``,
    - leave untouched.
    The inputs are connected to the transformed outputs of the parent nodes.

    The outputs from a transformed node are constructed as follows, trying each option
    in order until one succeeds:
    - call ``output``,
    - if the transformed node (``NodeLike``) is a ``dict``, look up the output name in it,
    - if not, try getting an attribute named after the output,
    - build a (transformed node, output name) tuple.

    Once every node has been transformed, the graph object is transformed as follows:
    - call ``graph``, if defined,
    - return the list of transformed sinks.
    The return value of this step is the return value of ``transform``.
    """

    def transform(self, graph: Graph) -> Any:
        """Apply the transformation to the given graph

        See ``Transformer`` for details.
        """
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
        return node_visit(self, node, inputs)

    def __transform_output(self, node: Any, output: Output) -> Any:
        if hasattr(self, "output"):
            return self.output(node, output.name)
        if isinstance(node, dict):
            return node[output.name]
        try:
            return getattr(node, output.name)
        except AttributeError:
            return (node, output)

    def __transform_graph(self, graph: Graph, sinks: list[dict[str, Any]]) -> Any:
        if hasattr(self, "graph"):
            return self.graph(graph, sinks)
        return sinks
