from typing import Iterator, cast

from cascade.contextgraph import ContextGraph
from cascade.graph import Graph, Node, Source, copy_graph


class Schedule(Graph):
    def __init__(
        self, task_graph: Graph, context_graph: ContextGraph, task_allocation: dict
    ):
        super().__init__(task_graph.sinks)
        self.context_graph = context_graph
        self.task_allocation = task_allocation
        if not self.has_cycle():
            raise ValueError(
                "Task graph and task allocation combination results in dependency cycle"
            )

    def __repr__(self) -> str:
        str = f"============= Schedule =============\n"
        for name, tasks in self.task_allocation.items():
            str += f"Processor {name}:\n"
            str += " → ".join(task for task in tasks) + "\n"
        str += "====================================\n"

        return str

    def processor(self, task_name) -> str:
        for processor, tasks in self.task_allocation.items():
            if task_name in tasks:
                return processor
        raise RuntimeError(f"Task {task_name} not in schedule {self.task_allocation}")

    def processors(self) -> Iterator[str]:
        return self.task_allocation.keys()

    def has_cycle(self) -> bool:
        new_graph = copy_graph(self)
        for _, p_tasks in self.task_allocation.items():
            for i_task, task in enumerate(p_tasks):
                if i_task < len(p_tasks) - 1:
                    current_task = new_graph.get_node(task)
                    next_task = new_graph.get_node(p_tasks[i_task + 1])
                    index = len(next_task.inputs)
                    while f"input{index}" in next_task.inputs:
                        index += 1
                    if current_task in new_graph.sinks:
                        new_graph.sinks.remove(current_task)
                        current_task.outputs = [Node.DEFAULT_OUTPUT]
                    next_task.inputs[f"input{index}"] = current_task.get_output()
        # Check number of nodes as disconnected components may have been introduced
        return not new_graph.has_cycle() and len(list(new_graph.nodes())) == len(
            list(self.nodes())
        )

    def sources(self) -> Iterator[Source]:
        for tasks in self.task_allocation.values():
            yield cast(Source, self.get_node(tasks[0]))

    def get_successors(self, node: Node) -> list[Node]:
        """
        Determines the next tasks to be executed after the given task,
        taking into account direct dependences in the task graph and
        also the task allocation.
        """
        successors = super().get_successors(node)
        task_worker = self.processor(node.name)
        tasks = self.task_allocation[task_worker]
        next = tasks.index(node.name) + 1
        if next < len(tasks):
            successors.setdefault(Node.DEFAULT_OUTPUT, []).append(
                (self.get_node(tasks[next]), "allocation")
            )
        return successors

    def get_predecessors(self, node: Node) -> dict[str, Node | tuple[Node, str]]:
        predecessors = super().get_predecessors(node)
        task_worker = self.processor(node.name)
        tasks = self.task_allocation[task_worker]
        previous = tasks.index(node.name) - 1
        if previous > 0:
            predecessors["allocation"].append(self.get_node(tasks[next]))
        return predecessors
