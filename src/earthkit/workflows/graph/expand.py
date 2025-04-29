# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Callable

from .graph import Graph
from .nodes import Node, Output
from .transform import Transformer


class _Subgraph:
    name: str
    leaves: dict[str, Node]
    output_map: dict[str, str]
    inner_sinks: list[Node]

    def __init__(
        self,
        name: str,
        leaves: dict[str, Node],
        output_map: dict[str, str],
        inner_sinks: list[Node],
    ):
        self.name = name
        self.leaves = leaves
        self.output_map = output_map
        self.inner_sinks = inner_sinks

    def __getattr__(self, name: str) -> Output:
        return self.get_output(name)

    def get_output(self, name: str | None = None) -> Output:
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
    """Transformer to connect a sub-graph in place of an expanded node

    Subclasses can override the `splice_source` and `splice_sink` methods to
    control how sources and sinks of the sub-graph are converted to processors
    connected to the parent graph.

    Parameters
    ----------
    name: str
        Name of the expanded node
    inputs: dict[str, Output]
        Inputs of the expanded node
    input_map: dict[str, str] | None
        Mapping from source names to node input names
    outputs: list[str]
        Outputs of the expanded node
    output_map: dict[str, str] | None
        Mapping from node output names to sink names
    """

    name: str
    inputs: dict[str, Output]
    outputs: dict[str, str]

    def __init__(
        self,
        name: str,
        inputs: dict[str, Output],
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

    def source(self, s: Node) -> Node:
        if s.name not in self.inputs:
            s.name = f"{self.name}.{s.name}"  # XXX: should we create a copy of s?
            return s
        return self.splice_source(f"{self.name}.{s.name}", s, self.inputs[s.name])

    def processor(self, p: Node, **inputs: Output) -> Node:
        p.name = f"{self.name}.{p.name}"  # XXX: should we create a copy of p?
        p.inputs = inputs
        return p

    def sink(self, s: Node, **inputs: Output) -> Node:
        if s.name not in self.outputs.values():
            s.name = f"{self.name}.{s.name}"  # XXX: should we create a copy of s?
            s.inputs = inputs
            return s
        return self.splice_sink(f"{self.name}.{s.name}", s, **inputs)

    def graph(self, g: Graph, sinks: list[Node]) -> _Subgraph:
        leaves = {}
        inner_sinks: list[Node] = []
        for s in sinks:
            sname = s.name.lstrip(f"{self.name}.")
            if sname in self.outputs.values():
                leaves[sname] = s
            else:
                inner_sinks.append(s)
        return _Subgraph(self.name, leaves, self.outputs, inner_sinks)

    def splice_source(self, name: str, s: Node, input: Output) -> Node:
        """Create a processor node to replace a source in the sub-graph

        The default implementation creates a bare Node with the original
        source's name, payload and outputs.

        Parameters
        ----------
        name: str
            Name of the processor to create
        s: Node
            Source to replace
        input: Output
            Output to which to connect the processor

        Returns
        -------
        Node
            Replacement for the source
        """
        return Node(name, s.outputs, s.payload, input=input)

    def splice_sink(self, name: str, s: Node, **inputs: Output) -> Node:
        """Create a processor node to replace a sink in the sub-graph

        The default implementation creates a bare Node with the original
        sink's name, payload and inputs.

        Parameters
        ----------
        name: str
            Name of the processor to create
        s: Node
            Sink to replace
        **inputs: dict[str, Output]
            Inputs for the processor

        Returns
        -------
        Node
            Replacement for the sink
        """
        return Node(name, outputs=None, payload=s.payload, **inputs)


ExpanderType = Callable[
    [Node], Graph | tuple[Graph, dict[str, str] | None, dict[str, str | None]] | None
]
SplicerType = Callable[
    [
        str,
        dict[str, Output],
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

    def node(self, n: Node, **inputs: Output) -> Node | _Subgraph:
        expanded = self.expand(n)
        output_map: dict[str, str] | None
        if expanded is None:
            n.inputs = inputs  # XXX: should we create a copy of n?
            return n
        if isinstance(expanded, Graph):
            input_map = None
            output_map = None
        else:
            expanded, input_map, output_map = expanded  # type: ignore # expanded[2] is dict[str, str|None]
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
    """Expand a graph by replacing nodes with sub-graphs

    The expansion is controlled by the ``expand`` callback, called for every
    node in topological order. For each node:
    - If the return value is None, do not replace it
    - If it is a graph, splice it in
    - If it is a 3-tuple, splice it in, using the second and third elements
      as input and output mappings, respectively.

    The splicing behaviour can be adapted by providing a factory function. When
    a node needs to be expanded, a new splicer (transformer) is created and
    applied to the sub-graph. The splicer is responsible for renaming the
    sub-graph nodes and connecting the subgraphs sources and sinks to the node
    inputs and outputs. See `Splicer` for details.

    The default splicing behaves as follows:
    - if not provided, the input mapping will map any source to the node input
      with the same name, if it exists,
    - if not provided, the output mapping will map any node output to the sink
      with the same name,
    - node names are prefixed by ``parent_name + "."``, where ``parent_name`` is
      the name of the node being expanded,
    - sources whose name is in the input mapping are replaced by a processor
      connected to the corresponding input,
    - sinks whose name is a value in the output mapping are replaced by a processor,
    - outputs of the node being expanded are connected to the corresponding
      transformed sink, based on the output map,
    - sources and sinks not found in the input and output maps are left as is.

    Parameters
    ----------
    expand: Node -> (None | Graph | tuple[Graph, dict[str, str] | None, dict[str, str]])
        Expand callback
    graph: Graph
        Input graph
    splicer: (str, dict[str, Output], dict[str, str] | None, list[str], dict[str, str] | None) -> Transformer
        Splicer factory, see `Splicer` for details
    """
    ex = _Expander(expand, splicer)
    return ex.transform(graph)
