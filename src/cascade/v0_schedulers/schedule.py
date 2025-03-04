from typing import Iterable, Iterator

from cascade.v0_contextgraph import ContextGraph
from cascade.graph import Graph, Node, copy_graph


class Schedule(Graph):
    def __init__(
        self, task_graph: Graph, context_graph: ContextGraph, task_allocation: dict
    ):
        if not Schedule.valid_allocations(task_graph, task_allocation):
            raise ValueError(
                "Task graph and task allocation combination results in dependency cycle,"
                + "or not all tasks have been allocated"
            )
        super().__init__(task_graph.sinks)
        self.context_graph = context_graph
        self.task_allocation = task_allocation

    def __repr__(self) -> str:
        str = "============= Schedule =============\n"
        for name, tasks in self.task_allocation.items():
            str += f"Processor {name}:\n"
            str += " → ".join(task for task in tasks) + "\n"
        str += "====================================\n"

        return str

    def processor(self, task_name: str) -> str:
        """Get processor name that task is allocated to

        Params
        ------
        task_name: str, name of task

        Returns
        -------
        str, name of processor
        """
        for processor, tasks in self.task_allocation.items():
            if task_name in tasks:
                return processor
        raise RuntimeError(f"Task {task_name} not in schedule {self.task_allocation}")

    def processors(self) -> Iterable[str]:
        """Iterator over all processor names in the schedule

        Returns
        -------
        Iterable[str]
        """
        return self.task_allocation.keys()

    def valid_allocations(task_graph: Graph, task_allocation: dict) -> bool:
        """Checks if the task allocation is valid, i.e. all tasks are allocated and there are no
        dependency cycles.

        Params
        ------
        task_graph: Graph, task graph
        task_allocation: dict, task allocation

        Returns
        -------
        bool, True if valid, False otherwise
        """
        # Check that all tasks are allocated
        all_allocated_tasks = set()
        for allocations in task_allocation.values():
            all_allocated_tasks.update(allocations)
        if len(list(task_graph.nodes())) != len(all_allocated_tasks):
            print(
                "Not all tasks are allocated. Task graph has",
                len(list(task_graph.nodes())),
                "tasks, but only",
                len(all_allocated_tasks),
                "are allocated.",
            )
            return False

        new_graph = copy_graph(task_graph)
        for _, p_tasks in task_allocation.items():
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
                    if new_graph.has_cycle():
                        print(
                            f"Schedule has dependency cycle with task {current_task.name}."
                            + f"Task allocation: \n {task_allocation}"
                        )
                        return False
                    if len(list(new_graph.nodes())) != len(list(task_graph.nodes())):
                        print("Schedule has disconnected components.")
                        return False
        return True

    def has_cycle(self) -> bool:
        return Schedule.valid_allocations(self, self.task_allocation)

    def sources(self) -> Iterator[Node]:
        return (self.get_node(tasks[0]) for tasks in self.task_allocation.values())

    def get_successors(self, node: Node) -> dict[str, list[tuple[Node, str]]]:
        """Determines the children of node, taking into account direct dependences
        in the task graph and also the task allocation.

        Params
        ------
        node: Node, node to determine children of

        Returns
        -------
        dict, output names mapped to list of (child, input_name) tuples
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
        """Determines parents of node, taking into account direct dependences in the task graph and
        also the task allocation.

        Params
        ------
        node: Node, node to determine parents of

        Returns
        -------
        dict, input names mapped to parent or (parent, output_name) tuple
        """
        predecessors = super().get_predecessors(node)
        task_worker = self.processor(node.name)
        tasks = self.task_allocation[task_worker]
        previous = tasks.index(node.name) - 1
        if previous > 0:
            predecessors["allocation"] = self.get_node(tasks[previous])
        return predecessors
