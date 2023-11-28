from collections.abc import Sequence

import networkx as nx

from .graph import Graph
from .nodes import Node


def to_networkx(graph: Graph, serialise=False) -> nx.MultiDiGraph:
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
