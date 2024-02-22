from collections.abc import Hashable
from dataclasses import dataclass
from struct import pack
from typing import Callable, Generic, TypeVar

from .graph import Graph
from .nodes import Node, Sink, Source
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
    key: KeyFunc[K]
    cuts: list[CutEdge[K]]
    sinks: dict[K, list[Sink]]

    def __init__(self, key: KeyFunc[K]):
        self.key = key
        self.cuts = []
        self.sinks = {}

    def node(self, node: Node, **inputs: tuple[K, Node.Output]) -> tuple[K, Node]:
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

    def output(self, tnode: tuple[K, Node], output: str) -> tuple[K, Node.Output]:
        k, node = tnode
        return (k, node.get_output(output))

    def graph(
        self, graph: Graph, sinks: list[tuple[K, Sink]]
    ) -> tuple[dict[K, Graph], list[CutEdge[K]]]:
        for k, sink in sinks:
            self.sinks.setdefault(k, []).append(sink)
        return {k: Graph(s) for k, s in self.sinks.items()}, self.cuts

    def cut_edge(self, cut: CutEdge[K], sink_in: Node.Output) -> tuple[Sink, Source]:
        return Sink(cut.name, input=sink_in), Source(cut.name)


SplitterType = Callable[[KeyFunc[K]], Splitter[K]]


def split_graph(
    key: KeyFunc[K], graph: Graph, splitter: SplitterType[K] = Splitter
) -> tuple[dict[K, Graph], list[CutEdge[K]]]:
    return splitter(key).transform(graph)
