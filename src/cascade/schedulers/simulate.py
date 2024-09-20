import copy
import functools
from dataclasses import dataclass
from typing import Callable, cast

import numpy as np

from cascade.contextgraph import Communicator, ContextGraph, Processor
from cascade.graph import Graph, Node
from cascade.taskgraph import Communication, Task, TaskGraph
from cascade.transformers import to_execution_graph
from cascade.utility import EventLoop, predecessors, successors

from .schedule import Schedule


@dataclass
class TaskState:
    finished: bool = False
    start_time: float = 0
    end_time: float = 0

    def duration(self) -> float:
        return self.end_time - self.start_time


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


@dataclass
class ProcessorState:
    current_task: Task | None = None
    end_time: float = 0
    idle_time: float = 0
    memory_usage: MemoryUsage = MemoryUsage()


@dataclass
class CommunicatorState:
    current_task: Communication | None = None
    end_time: float = 0
    idle_time: float = 0


class ContextState:
    def __init__(self, context_graph: ContextGraph, with_communication: bool = False):
        self.context_graph = copy.deepcopy(context_graph)
        for ctx in self.context_graph:
            ctx.state = ProcessorState()
        for communicator in self.context_graph.communicators():
            communicator.state = CommunicatorState()  # type: ignore # TODO declare state in communicator
        self.sim = EventLoop()
        self.with_communication = with_communication

    def processor(self, name: str):
        return self.context_graph.node_dict[name]

    def communicator(self, name: str):
        for comm in self.context_graph.communicators():
            if comm.name == name:
                return comm
        raise ValueError(f"Communicator {name} not found in context graph")

    def assign_task_to_processor(self, task: Task, processor: Processor, start_time: float, callback: Callable):
        """Assign task to processor and schedule callback for task completion.

        Params
        ------
        task: Task, task to assign
        processor: Processor, processor to assign task to
        start_time: float, time to start the task
        callback: callable, callback to trigger on task completion
        """
        if not hasattr(processor, "state"):
            raise TypeError
        processor.state.idle_time += start_time - processor.state.end_time
        processor.state.current_task = task
        if not task.state:
            raise TypeError
        task.state.start_time = start_time
        if isinstance(task, Task) and isinstance(processor, Processor):
            task.state.end_time = start_time + task.duration
        elif isinstance(task, Communication) and isinstance(processor, Communicator):
            task.state.end_time = start_time + task.size / processor.bandwidth + processor.latency
        else:
            raise ValueError(f"Invalid task {type(task)} and processor {type(processor)} combination")

        processor.state.end_time = task.state.end_time
        if isinstance(processor, Processor):
            processor.state.memory_usage.add_task(task)
        self.sim.add_event(task.state.end_time, callback)

    def update(self, completed_tasks: list[str]):
        """Update memory usage of processors and communicators based on completed tasks.

        Params
        ------
        completed_tasks: list[str], list of completed task names
        """
        for processor in self.context_graph:
            for task in processor.state.memory_usage.current_tasks():
                if task.name in completed_tasks:
                    processor.state.memory_usage.remove_task(task)

    def idle_processors(self) -> list[Processor | Communicator]:
        """List of idle processors and communicators.

        Returns
        -------
        list[Processor | Communicator]
        """
        ret = [p for p in self.context_graph if p.state.current_task is None]
        if self.with_communication:
            for communicator in self.context_graph.communicators():
                if not hasattr(communicator, "state"):
                    raise TypeError
                if communicator.state.current_task is None:
                    ret.append(communicator)
        return ret

    def run(self):
        self.sim.run()


class ExecutionState:
    def __init__(self, graph: Graph | Schedule, with_communication: bool = False):
        self.task_graph = to_execution_graph(graph, TaskState)
        self.communication_tasks: dict[str, str] = {}

        if with_communication:
            if not isinstance(graph, Schedule):
                raise ValueError("Communication tasks can only be enabled for schedules")
            for start, end in self.task_graph.edges():
                start_processor = graph.context_graph.node_dict[graph.processor(start.name)]
                end_processor = graph.context_graph.node_dict[graph.processor(end.name)]
                if start_processor != end_processor:
                    t = self.task_graph._make_communication_task(start, end, TaskState)
                    # Find the communicator which can handle this communication
                    communicator = graph.context_graph.communicator(start_processor, end_processor)
                    self.communication_tasks[t.name] = communicator.name

        self.total_tasks = len(list(self.task_graph.nodes()))


class Simulator:
    """Simulator for task execution for graph or schedule. If communication tasks are
    enable, the  simulatordoes not dictate the order of communications, it just begins
    communications as soon as data becomes available.
    """

    DEFAULT_PROCESSOR = "__defaultprocessor__"

    def __init__(self):
        self.completed_tasks = set()
        self.eligible = {}

    def reset_state(self):
        self.completed_tasks = set()
        self.eligible = {}

    def on_task_complete(
        self,
        execution_state: ExecutionState,
        context_state: ContextState,
        task: Task,
        processor: Processor,
        time: float,
    ):
        """Callback when task is completed. Updates the state of the task and processor,
        and triggers next round of task assignments to idle processors.

        Params
        ------
        execution_state: ExecutionState, state of the execution
        context_state: ContextState, state of the context
        task: Task, task that has been completed
        processor: Processor, processor that completed the task
        time: float, time at which the task was completed
        """
        if task.state is None:
            raise TypeError
        task.state.finished = True
        if not hasattr(processor, "state"):
            raise TypeError
        processor.state.current_task = None
        self.completed_tasks.add(task.name)
        self.update_eligible_tasks(task, execution_state)
        self.assign_eligible_tasks(time, execution_state, context_state)

    def is_task_eligible(self, task: Node, schedule: Graph) -> bool:
        """Check if a task is eligible for execution based on its predecessors.

        Params
        ------
        task: Task, task to check eligibility
        schedule: Graph or Schedule, which specifies the task dependencies

        Returns
        -------
        bool, True if task is eligible for execution
        """
        return all(t.name in self.completed_tasks for t in predecessors(schedule, task))

    def update_eligible_tasks(self, task: Task, execution: ExecutionState):
        """Update the eligible tasks for execution based on the completion of a task.

        Params
        ------
        task: Task, task that has been completed
        execution: ExecutionState, state of the execution
        """
        next_tasks = successors(execution.task_graph, task)
        for next_task in next_tasks:
            if isinstance(next_task, Communication):
                comm = execution.communication_tasks[next_task.name]
                self.eligible.setdefault(comm, []).append(next_task)
            else:
                if self.is_task_eligible(next_task, execution.task_graph):
                    processor = (
                        execution.task_graph.get_processor(next_task.name).name  # type: ignore
                        if isinstance(execution.task_graph, Schedule)
                        else Simulator.DEFAULT_PROCESSOR
                    )
                    if processor == Simulator.DEFAULT_PROCESSOR:
                        eligible_tasks = self.eligible.setdefault(Simulator.DEFAULT_PROCESSOR, [])
                        if next_task not in eligible_tasks:
                            eligible_tasks.append(next_task)
                    else:
                        if processor in self.eligible:
                            raise RuntimeError(f"Processor {processor} already has an eligible task")
                        self.eligible[processor] = next_task

    def initialise_eligible_tasks(self, graph: TaskGraph | Schedule):
        """Initialise the eligible tasks for execution based on the graph or
        schedule.

        Params
        ------
        graph: TaskGraph or Schedule, graph or schedule obtain tasks from

        Returns
        -------
        nothing, the list of tasks that are eligible for execution are set to self
        """
        sources = graph.sources()
        if isinstance(graph, Schedule):
            for source in sources:
                processor = graph.processor(source.name)
                self.eligible[processor] = source
        else:
            self.eligible[Simulator.DEFAULT_PROCESSOR] = list(sources)

    def next_task(self, processor: Processor | Communicator):
        """Get the next task to be executed by the processor.

        Params
        ------
        processor: Processor or Communicator, processor on which to
        execute next task
        """
        tasks = self.eligible.get(processor.name, None)
        if isinstance(tasks, list) and len(tasks) > 0:
            return self.eligible[processor.name].pop(0)
        if isinstance(tasks, Task):
            return self.eligible.pop(processor.name)
        if isinstance(processor, Processor) and len(self.eligible.get(Simulator.DEFAULT_PROCESSOR, [])) > 0:
            return self.eligible[Simulator.DEFAULT_PROCESSOR].pop(0)
        return None

    def assign_eligible_tasks(self, time: float, execution_state: ExecutionState, context_state: ContextState):
        """Assign eligible tasks to idle processors.

        Params
        ------
        time: float, current time
        execution_state: ExecutionState, state of the execution
        context_state: ContextState, state of the context
        """
        context_state.update(self.completed_tasks)
        for processor in context_state.idle_processors():
            next_task = self.next_task(processor)
            if next_task is not None:
                context_state.assign_task_to_processor(
                    next_task,
                    cast(Processor, processor),
                    time,
                    functools.partial(
                        self.on_task_complete,
                        execution_state,
                        context_state,
                        next_task,
                        cast(Processor, processor),
                    ),
                )

    def _execute_graph(
        self,
        graph: Graph,
        *,
        context_graph: ContextGraph,
        with_communication: bool = False,
    ) -> tuple[ExecutionState, ContextState]:
        """Simulate execution graph with context graph.

        Params
        ------
        graph: Graph, graph to execute
        context_graph: ContextGraph, context graph to execute on
        with_communication: bool, whether to include communication tasks

        Returns
        -------
        tuple[ExecutionState, ContextState], final state of the execution and context
        """
        context_state = ContextState(context_graph, with_communication)
        execution_state = ExecutionState(graph, with_communication)
        self.reset_state()
        self.initialise_eligible_tasks(execution_state.task_graph)
        self.assign_eligible_tasks(time=0, execution_state=execution_state, context_state=context_state)
        context_state.run()

        if len(self.completed_tasks) != execution_state.total_tasks:
            raise RuntimeError(
                f"Failed to complete all tasks. Completed {len(self.completed_tasks)}"
                + f"out of {execution_state.total_tasks}."
            )
        return execution_state, context_state

    def execute(self, graph: Graph | Schedule, **kwargs) -> tuple[ExecutionState, ContextState]:
        """Execute graph or schedule with context graph.

        Params
        ------
        graph: Graph or Schedule, graph or schedule to execute

        Returns
        -------
        tuple[ExecutionState, ContextState], final state of the execution and context
        """
        if isinstance(graph, Schedule):
            kwargs["context_graph"] = graph.context_graph
        return self._execute_graph(graph, **kwargs)
