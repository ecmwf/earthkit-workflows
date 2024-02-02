import numpy as np

from cascade.utility import EventLoop
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
        def __init__(self, task_graph: Graph, context_graph: ContextGraph):
            if not isinstance(task_graph, TaskGraph):
                task_graph = to_task_graph(task_graph)
            self.task_graph = task_graph
            self.context_graph = context_graph
            self.completed_tasks = set()
            self.task_allocation = {p.name: [] for p in self.context_graph}
            self.memory_usage = {p.name: MemoryUsage() for p in self.context_graph}
            self.eligible = list(self.task_graph.sources())
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
        successors = self.state.task_graph.successors(task)

        for dependent in successors:
            predecessors = self.state.task_graph.predecessors(dependent)
            if all(t.name in self.state.completed_tasks for t in predecessors):
                self.state.eligible.append(dependent)

        self.assign_idle_processors(time)

    def update_memory_usage(self, processor) -> float:
        for task in self.state.memory_usage[processor.name].current_tasks():
            successors = self.state.task_graph.successors(task)
            if all(
                [
                    x in self.state.completed_tasks or x in self.state.eligible
                    for x in successors
                ]
            ):
                self.state.memory_usage[processor.name].remove_task(task)

        return self.state.memory_usage[processor.name].memory

    def assign_idle_processors(self, time):
        # Sort next tasks
        self.state.eligible.sort(key=lambda x: (x.cost, len(x.inputs)))

        # Assign idle processors
        for processor in self.state.context_graph:
            new_mem_usage = self.update_memory_usage(processor)
            if (
                len(self.state.task_allocation[processor.name]) == 0
                or self.state.task_allocation[processor.name][-1]
                in self.state.completed_tasks
            ):
                if (
                    len(self.state.eligible) > 0
                    and (new_mem_usage + self.state.eligible[0].memory)
                    < processor.memory
                ):
                    self.assign_task_to_processor(
                        self.state.eligible.pop(), processor, time
                    )

    def schedule(self, task_graph: Graph, context_graph: ContextGraph):
        self.state = DepthFirstScheduler.State(task_graph, context_graph)
        self.assign_idle_processors(time=0)
        self.state.sim.run()
        print(
            f"Finished {len(self.state.completed_tasks)} tasks out of {len(list(self.state.task_graph.nodes()))}"
        )
        assert len(self.state.completed_tasks) == len(
            list(self.state.task_graph.nodes())
        )

        return Schedule(
            self.state.task_graph, self.state.context_graph, self.state.task_allocation
        )
