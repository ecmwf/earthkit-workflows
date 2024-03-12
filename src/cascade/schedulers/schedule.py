import randomname
import datetime

from cascade.contextgraph import ContextGraph
from cascade.graph import Graph, Node
from cascade.graph.copy import copy_graph


class Schedule:
    def __init__(
        self, task_graph: Graph, context_graph: ContextGraph, task_allocation: dict
    ):
        self.task_graph = task_graph
        self.context_graph = context_graph
        self.task_allocation = task_allocation
        self.name = randomname.get_name()
        self.created_at = datetime.datetime.utcnow()

        assert self.valid_allocations(self.task_graph, self.task_allocation)

    def __repr__(self) -> str:
        str = f"============= Schedule: {self.name} =============\n"
        str += f"Created at: {self.created_at} UTC\n"
        for name, tasks in self.task_allocation.items():
            str += f"Processor {name}:\n"
            str += " → ".join(task for task in tasks) + "\n"
        str += "================================================\n"

        return str

    def get_processor(self, task_name) -> str:
        for processor, tasks in self.task_allocation.items():
            if task_name in tasks:
                return processor
        raise RuntimeError(f"Task {task_name} not in schedule {self.task_allocation}")

    @classmethod
    def valid_allocations(
        cls, task_graph: Graph, task_allocation: dict[str, list]
    ) -> bool:
        dependency_graph = copy_graph(task_graph)

        for _, p_tasks in task_allocation.items():
            for i_task, task in enumerate(p_tasks):
                if i_task < len(p_tasks) - 1:
                    current_task = dependency_graph.get_node(task)
                    next_task = dependency_graph.get_node(p_tasks[i_task + 1])
                    key = f"inputs{len(next_task.inputs)}"
                    assert key not in next_task.inputs
                    if current_task in dependency_graph.sinks:
                        dependency_graph.sinks.remove(current_task)
                        current_task.outputs = [Node.DEFAULT_OUTPUT]
                    next_task.inputs[key] = current_task.get_output()
                    # Check number of nodes as disconnected components may have been introduced
                    if dependency_graph.has_cycle() or len(
                        list(dependency_graph.nodes())
                    ) != len(list(task_graph.nodes())):
                        return False
        return True
