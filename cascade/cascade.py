from .scheduler import Schedule, DepthFirstScheduler
from .graphs import ContextGraph, TaskGraph
from .executor import ExecutionReport, BasicExecutor
from .graph_config import Config
from .transformers import to_task_graph

from ppgraph import Graph, deduplicate_nodes
from . import graph_templates


class Cascade:
    def graph(config: Config):
        total_graph = Graph([])
        for _, param_config in config.parameters.items():
            total_graph += getattr(graph_templates, config.product)(param_config)

        return deduplicate_nodes(total_graph)

    def schedule(
        taskgraph: Graph, contextgraph: ContextGraph, determine_resources: bool = True
    ) -> Schedule:
        if not isinstance(taskgraph, TaskGraph):
            taskgraph = to_task_graph(taskgraph)
            if determine_resources:
                # Need to execute and assign resources
                test_context = ContextGraph()
                test_context.add_node("cpu1", "cpu", 100, 100)
                test_schedule = DepthFirstScheduler(
                    taskgraph, test_context
                ).create_schedule()
                BasicExecutor(test_schedule).determine_resources()
                taskgraph = test_schedule.task_graph

        return DepthFirstScheduler(taskgraph, contextgraph).create_schedule()

    def execute(schedule: Schedule) -> ExecutionReport:
        return BasicExecutor(schedule).execute()

    def simulate(
        schedule: Schedule, with_communication: bool = True
    ) -> ExecutionReport:
        return BasicExecutor(schedule, with_communication).simulate()
