from .scheduler import Schedule, DepthFirstScheduler
from .graphs import ContextGraph, TaskGraph
from .executor import ExecutionReport, BasicExecutor
from .graph_config import Config

from ppgraph import Graph
from . import graph_templates


class Cascade:

    def create_schedule(taskgraph: TaskGraph, contextgraph: ContextGraph) -> Schedule:
        return DepthFirstScheduler(taskgraph, contextgraph).create_schedule()
    
    def execute(schedule: Schedule) -> ExecutionReport:
        return BasicExecutor(schedule).execute()
    
    def simulate(schedule: Schedule) -> ExecutionReport:
        return BasicExecutor(schedule).simulate()

    def graph(product: str, config: Config):
        total_graph = Graph([])
        for _, param_config in config.parameters.items():
            total_graph += getattr(graph_templates, product)(param_config)

        return total_graph



            



