from .scheduler import Schedule, DepthFirstScheduler
from .graph import Graph
from .graphs import ContextGraph, TaskGraph
from .executor import ExecutionReport, BasicExecutor
from .transformers import to_task_graph

GRAPHS = []


class Cascade:
    def graph(product, *args, **kwargs):
        if product not in GRAPHS:
            raise Exception(f"No graph for '{product}' registered")
        return getattr(Cascade, product)(*args, **kwargs)

    def schedule(
        taskgraph: Graph,
        contextgraph: ContextGraph
    ) -> Schedule:
        if not isinstance(taskgraph, TaskGraph):
            taskgraph = to_task_graph(taskgraph)

        return DepthFirstScheduler(taskgraph, contextgraph).create_schedule()

    def execute(schedule: Schedule) -> ExecutionReport:
        return BasicExecutor(schedule).execute()

    def simulate(
        schedule: Schedule, with_communication: bool = True
    ) -> ExecutionReport:
        return BasicExecutor(schedule, with_communication).simulate()


def register_graph(name: str, func):
    assert name not in GRAPHS
    GRAPHS.append(name)
    setattr(Cascade, name, func)
