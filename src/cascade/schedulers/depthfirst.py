import numpy as np

from cascade.utility import EventLoop, successors, predecessors
from cascade.taskgraph import Task, TaskGraph
from cascade.graph import Graph
from cascade.contextgraph import ContextGraph
from cascade.transformers import to_task_graph
from .schedule import Schedule


class MemoryUsage:
    def __init__(self):
        self.tasks_in_memory: list[Task] = []

    @property
    def memory(self) -> float:
        return np.sum([t.memory for t in self.tasks_in_memory])

    def remove_task(self, task: Task):
        self.tasks_in_memory.remove(task)

    def add_task(self, task: Task):
        if task not in self.tasks_in_memory:
            self.tasks_in_memory.append(task)

    def current_tasks(self) -> list[Task]:
        return self.tasks_in_memory[:]

    def __repr__(self) -> str:
        return f"Memory:{self.memory},Tasks:{self.tasks_in_memory}"


class DepthFirstScheduler:
    class State:
        def __init__(self, task_graph: Graph | TaskGraph, context_graph: ContextGraph):
            if not isinstance(task_graph, TaskGraph):
                task_graph = to_task_graph(task_graph, None)
            self.task_graph = task_graph
            self.context_graph = context_graph
            self.completed_tasks = set()
            self.task_allocation = {p.name: [] for p in self.context_graph}
            self.memory_usage = {p.name: MemoryUsage() for p in self.context_graph}

            # Sort sinks by total compute cost
            sinks = self.task_graph.sinks[:]
            sinks.sort(key=lambda x: self.task_graph.accumulated_cost(x))
            self.eligible = []
            for sink in sinks:
                tmp_graph = Graph([sink])
                for src in tmp_graph.sources():
                    if src not in self.eligible:
                        self.eligible.append(src)
            self.eligible.reverse()
            self.sim = EventLoop()

    def __init__(self):
        self.state = None

    def assign_task_to_processor(self, task, processor, start_time):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        self.state.task_allocation[processor.name].append(task.name)
        end_time = start_time + task.cost / processor.speed
        self.state.memory_usage[processor.name].add_task(task)
        self.state.sim.add_event(end_time, self.on_task_complete, task)

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time} by processor {processor.name}")
        self.state.completed_tasks.add(task.name)
        children = successors(self.state.task_graph, task)

        for dependent in children:
            parents = predecessors(self.state.task_graph, dependent)
            if all(t.name in self.state.completed_tasks for t in parents):
                self.state.eligible.append(dependent)

        self.assign_idle_processors(time)

    def update_memory_usage(self, processor) -> float:
        for task in self.state.memory_usage[processor.name].current_tasks():
            if task.name in self.state.completed_tasks:
                self.state.memory_usage[processor.name].remove_task(task)

        return self.state.memory_usage[processor.name].memory

    def assign_idle_processors(self, time):
        # Assign idle processors
        for processor in self.state.context_graph:
            new_mem_usage = self.update_memory_usage(processor)
            if (
                len(self.state.task_allocation[processor.name]) == 0
                or self.state.task_allocation[processor.name][-1]
                in self.state.completed_tasks
            ):
                pop_index = None
                # Take from back so newly added dependents get picked off first
                for index in range(len(self.state.eligible) - 1, -1, -1):
                    if (
                        new_mem_usage + self.state.eligible[index].memory
                    ) < processor.memory:
                        pop_index = index
                        break

                if pop_index is not None:
                    self.assign_task_to_processor(
                        self.state.eligible.pop(pop_index), processor, time
                    )

    def schedule(
        self, task_graph: Graph | TaskGraph, context_graph: ContextGraph
    ) -> Schedule:
        """
        Schedule tasks in task graph to workers in context graph.

        Params
        ------
        task_graph: Graph or TaskGraph, if Graph then will perform execution of graph
        using thread pool to determine resources in the transformation to a TaskGraph
        context_graph: ContextGraph, containers nodes to which tasks should be assigned

        Returns
        -------
        Schedule
        """

        self.state = DepthFirstScheduler.State(task_graph, context_graph)
        self.assign_idle_processors(time=0)
        self.state.sim.run()
        print(
            f"Finished {len(self.state.completed_tasks)} tasks out of {len(list(self.state.task_graph.nodes()))}"
        )
        if len(self.state.completed_tasks) != len(list(self.state.task_graph.nodes())):
            raise RuntimeError("Scheduler failed to complete all tasks.")

        return Schedule(
            self.state.task_graph, self.state.context_graph, self.state.task_allocation
        )
