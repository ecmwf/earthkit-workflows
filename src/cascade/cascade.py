from .graph import Graph
from .graph import deduplicate_nodes
from .contextgraph import ContextGraph
from .schedulers.depthfirst import DepthFirstScheduler
from .schedulers.schedule import Schedule
from .executors.dask import DaskLocalExecutor

GRAPHS: dict = {}


class Cascade:
    def graph(product, *args, **kwargs) -> Graph:
        if product not in GRAPHS:
            raise Exception(f"No graph for '{product}' registered")
        return GRAPHS[product](*args, **kwargs)

    def merge(*graphs) -> Graph:
        sinks = []
        for graph in graphs:
            sinks.extend(graph.sinks)
        total_graph = Graph(sinks)
        return deduplicate_nodes(total_graph)

    def schedule(graph: Graph, context: ContextGraph) -> Schedule:
        return DepthFirstScheduler().schedule(graph, context)

    def execute(schedule: Schedule):
        return DaskLocalExecutor().execute(schedule)


def register_graph(name: str, func):
    assert name not in GRAPHS
    GRAPHS[name] = func
