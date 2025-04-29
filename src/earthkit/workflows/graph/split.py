# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Hashable
from dataclasses import dataclass
from struct import pack
from typing import Callable, Generic, TypeVar

from .graph import Graph
from .nodes import Node, Output
from .transform import Transformer

K = TypeVar("K", bound=Hashable)
KeyFunc = Callable[[Node], K]


@dataclass(frozen=True)
class CutEdge(Generic[K]):
    source_key: K
    source_node: str
    source_output: str
    dest_key: K
    dest_node: str
    dest_input: str

    @property
    def name(self) -> str:
        h = pack("n", hash(self)).hex()
        return f"__cut_{h}__"

    @property
    def source(self) -> str | tuple[str, str]:
        if self.source_output == Node.DEFAULT_OUTPUT:
            return self.source_node
        return (self.source_node, self.source_output)


class Splitter(Transformer, Generic[K]):
    """Transformer to perform graph splitting

    Subclasses can override the `cut_edge` method to control how the cut edges
    are replaced by a sink and a source on either side.
    """

    key: KeyFunc[K]
    cuts: list[CutEdge[K]]
    sinks: dict[K, list[Node]]

    def __init__(self, key: KeyFunc[K]):
        self.key = key
        self.cuts = []
        self.sinks = {}

    def node(self, node: Node, **inputs: tuple[K, Output]) -> tuple[K, Node]:
        k = self.key(node)
        new_inputs = {}
        for iname, (ik, ival) in inputs.items():
            if ik == k:
                new_inputs[iname] = ival
                continue
            cut = CutEdge(ik, ival.parent.name, ival.name, k, node.name, iname)
            self.cuts.append(cut)
            sink, source = self.cut_edge(cut, ival)
            self.sinks.setdefault(ik, []).append(sink)
            new_inputs[iname] = source.get_output()
        node.inputs = new_inputs  # XXX: should we create a copy of node?
        return (k, node)

    def output(self, tnode: tuple[K, Node], output: str) -> tuple[K, Output]:
        k, node = tnode
        return (k, node.get_output(output))

    def graph(
        self, graph: Graph, sinks: list[tuple[K, Node]]
    ) -> tuple[dict[K, Graph], list[CutEdge[K]]]:
        for k, sink in sinks:
            self.sinks.setdefault(k, []).append(sink)
        return {k: Graph(s) for k, s in self.sinks.items()}, self.cuts

    def cut_edge(self, cut: CutEdge[K], sink_in: Output) -> tuple[Node, Node]:
        """Create nodes to replace a cut edge

        Parameters
        ----------
        cut: CutEdge
            Edge that has been cut
        sink_in: Output
            Input to be connected to the sink replacing the start of the edge

        Returns
        -------
        Node
            Sink to replace the start of the edge
        Node
            Source to replace the end of the edge
        """
        return Node(cut.name, outputs=[], input=sink_in), Node(cut.name)


SplitterType = Callable[[KeyFunc[K]], Splitter[K]]


def split_graph(
    key: KeyFunc[K], graph: Graph, splitter: SplitterType[K] = Splitter
) -> tuple[dict[K, Graph], list[CutEdge[K]]]:
    """Split a graph according to some key

    Each sub-graph in the split will consist of nodes with the same key. The key
    type ``K`` must be hashable and support the ``==`` operator.

    Sources and sinks are created by the `Splitter` class to replace cut edges.
    The names are generated using a hash (see `CutEdge.name`). See the
    `Splitter` class for information on how to create custom nodes instead.

    Parameters
    ----------
    key: Node -> K
        Callback to get the key for a given node
    graph: Graph
        Input graph
    splitter: (Node -> K) -> Splitter[K]
        Splitter class

    Returns
    -------
    dict[K, Graph]
        Sub-graphs corresponding to each key
    list[CutEdge[K]]
        List of cut edges
    """
    return splitter(key).transform(graph)
