from typing import Callable

from .graph import Graph
from .nodes import Node, Processor, Source, Sink
from .transform import Transformer


class _Subgraph:
    name: str
    leaves: dict[str, Node]
    output_map: dict[str, str]
    inner_sinks: list[Sink]

    def __init__(
        self,
        name: str,
        leaves: dict[str, Node],
        output_map: dict[str, str],
        inner_sinks: list[Sink],
    ):
        self.name = name
        self.leaves = leaves
        self.output_map = output_map
        self.inner_sinks = inner_sinks

    def __getattr__(self, name: str) -> Node.Output:
        return self.get_output(name)

    def get_output(self, name: str | None = None) -> Node.Output:
        if name is None:
            name = Node.DEFAULT_OUTPUT
        if name in self.output_map:
            lname = self.output_map[name]
            if lname in self.leaves:
                return self.leaves[lname].get_output()
        if name == Node.DEFAULT_OUTPUT:
            raise AttributeError(
                f"No default output node found in sub-graph {self.name!r}"
            )
        raise AttributeError(
            f"No output node named {name!r} in sub-graph {self.name!r}"
        )


class Splicer(Transformer):
    name: str
    inputs: dict[str, Node.Output]
    outputs: dict[str, str]

    def __init__(
        self,
        name: str,
        inputs: dict[str, Node.Output],
        input_map: dict[str, str] | None,
        outputs: list[str],
        output_map: dict[str, str] | None,
    ):
        self.name = name
        self.inputs = (
            inputs
            if input_map is None
            else {iname: inputs[mname] for iname, mname in input_map.items()}
        )
        if output_map is None:
            self.outputs = {oname: oname for oname in outputs}
        else:
            self.outputs = {oname: output_map.get(oname, oname) for oname in outputs}

    def source(self, s: Source) -> Node:
        if s.name not in self.inputs:
            s.name = f"{self.name}.{s.name}"  # XXX: should we create a copy of s?
            return s
        return self.splice_source(f"{self.name}.{s.name}", s, self.inputs[s.name])

    def processor(self, p: Processor, **inputs: Node.Output) -> Processor:
        p.name = f"{self.name}.{p.name}"  # XXX: should we create a copy of p?
        p.inputs = inputs
        return p

    def sink(self, s: Sink, **inputs: Node.Output) -> Node:
        if s.name not in self.outputs.values():
            s.name = f"{self.name}.{s.name}"  # XXX: should we create a copy of s?
            s.inputs = inputs
            return s
        return self.splice_sink(f"{self.name}.{s.name}", s, **inputs)

    def graph(self, g: Graph, sinks: list[Node]) -> _Subgraph:
        leaves = {}
        inner_sinks = []
        for s in sinks:
            sname = s.name.lstrip(f"{self.name}.")
            if sname in self.outputs.values():
                leaves[sname] = s
            else:
                inner_sinks.append(s)
        return _Subgraph(self.name, leaves, self.outputs, inner_sinks)

    def splice_source(self, name: str, s: Source, input: Node.Output) -> Node:
        return Processor(name, s.outputs, s.payload, input=input)

    def splice_sink(self, name: str, s: Sink, **inputs: Node.Output) -> Node:
        return Processor(name, payload=s.payload, **inputs)


ExpanderType = Callable[
    [Node], Graph | tuple[Graph, dict[str, str] | None, dict[str, str | None]] | None
]
SplicerType = Callable[
    [
        str,
        dict[str, Node.Output],
        dict[str, str] | None,
        list[str],
        dict[str, str] | None,
    ],
    Transformer,
]


class _Expander(Transformer):
    expand: ExpanderType
    splicer: SplicerType

    def __init__(self, expand: ExpanderType, splicer: SplicerType = Splicer):
        self.expand = expand
        self.splicer = splicer

    def node(self, n: Node, **inputs: Node.Output) -> Node | _Subgraph:
        expanded = self.expand(n)
        if expanded is None:
            n.inputs = inputs  # XXX: should we create a copy of n?
            return n
        if isinstance(expanded, Graph):
            input_map = None
            output_map = None
        else:
            expanded, input_map, output_map = expanded
        sp = self.splicer(n.name, inputs, input_map, n.outputs, output_map)
        return sp.transform(expanded)

    def graph(self, graph: Graph, sinks: list[Node | _Subgraph]) -> Graph:
        new_sinks = []
        for sink in sinks:
            if isinstance(sink, Node):
                new_sinks.append(sink)
            else:
                new_sinks.extend(sink.inner_sinks)
        return Graph(new_sinks)


def expand_graph(
    expand: ExpanderType, graph: Graph, splicer: SplicerType = Splicer
) -> Graph:
    ex = _Expander(expand, splicer)
    return ex.transform(graph)
