# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Sequence

import networkx as nx

from .graph import Graph
from .nodes import Node


def to_networkx(graph: Graph, serialise=False) -> nx.MultiDiGraph:
    """Convert a graph to a NetworkX graph

    If ``serialise`` is true, the node payloads in the output graph are the
    serialised values of the corresponding nodes of the input graph. Otherwise,
    the node objects of the input graph are passed as is.
    """
    graph_s = None if serialise else graph
    g = nx.MultiDiGraph(graph=graph_s, sinks=[s.name for s in graph.sinks])
    for node in graph.nodes():
        node_s = node.serialise() if serialise else node
        g.add_node(node.name, node=node_s)
        g.add_edges_from(
            (isrc.parent.name, node.name, {"source_out": isrc.name, "dest_in": iname})
            for iname, isrc in node.inputs.items()
        )
    return g


def topological_layout(g: nx.MultiDiGraph):
    pos = {}
    for i, gen in enumerate(nx.topological_generations(g)):
        for j, node in enumerate(sorted(gen)):
            pos[node] = [float(j), -float(i)]
    return pos


def draw_graph(
    graph: Graph | nx.MultiDiGraph,
    pos: dict[str, Sequence[float]] | None = None,
    with_edge_labels: bool = False,
):
    """Draw a graph using NetworkX"""
    g = to_networkx(graph) if isinstance(graph, Graph) else graph
    pos = topological_layout(g) if pos is None else pos
    nx.draw(g, pos, with_labels=True)
    if with_edge_labels:
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels={
                e: (a["source_out"] if a["source_out"] != Node.DEFAULT_OUTPUT else "")
                + "->"
                + a["dest_in"]
                for e, a in g.edges.items()
            },
        )
