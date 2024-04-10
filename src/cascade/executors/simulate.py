import copy
from dataclasses import dataclass, field
import randomname
import datetime

from cascade.contextgraph import Communicator, Processor
from cascade.taskgraph import Task, Communication
from cascade.utility import EventLoop
from cascade.transformers import to_execution_graph
from cascade.schedulers.schedule import Schedule


####################################################################################################


@dataclass
class TaskState:
    finished: bool = False
    start_time: float = 0
    end_time: float = 0

    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class CommunicationState:
    communicator: Communicator = None
    finished: bool = False
    start_time: float = 0
    end_time: float = 0


@dataclass
class ProcessorState:
    next_task_index: int = 0
    current_task: Task = None
    end_time: float = 0
    idle_time: float = 0


@dataclass
class CommunicatorState:
    tasks: list[Communication] = field(default_factory=list)
    next_task_index: int = 0
    current_task: Communication = None
    end_time: float = 0
    idle_time: float = 0


####################################################################################################


class ExecutionReport:
    def __init__(self, schedule, task_graph, context_graph):
        self.schedule = schedule
        self.task_graph = task_graph
        self.context_graph = context_graph
        self.name = randomname.get_name()
        self.created_at = datetime.datetime.utcnow()

    def __repr__(self) -> str:
        str = f"============= Execution Report: {self.name} =============\n"
        str += f"Using schedule: {self.schedule.name}\n"
        str += f"Created at: {self.created_at} UTC\n"
        for processor in self.context_graph:
            str += f"Processor {processor.name} completes at {processor.state.end_time:.2f} ({processor.state.idle_time:.2f} idle):\n"
            str += (
                " → ".join(
                    f"{task} (end: {self.task_graph.get_node(task).state.end_time:.2f})"
                    for task in self.schedule.task_allocation[processor.name]
                )
                + "\n"
            )
        str += "================================================\n"
        str += "Note: Idle times currently incorrect.\n"

        return str

    def cost(self) -> float:
        return sum([processor.state.idle_time for processor in self.context_graph])


class Simulator:
    """
    Simulator for task execution according to order given by the schedule.
    It does not dictate the order of communications, it just begins communications
    as soon as data becomes available.
    """

    class State:
        def __init__(self, schedule: Schedule):
            self.schedule = schedule
            self.task_graph = to_execution_graph(schedule.task_graph)
            self.context_graph = copy.deepcopy(schedule.context_graph)
            self.total_tasks = len(list(self.task_graph.nodes()))
            self.ntasks_complete = 0
            self.sim = EventLoop()

            for task in self.task_graph.nodes():
                task.state = TaskState()
            for ctx in self.context_graph:
                ctx.state = ProcessorState()
            for _, _, ctx in self.context_graph.edges(data=True):
                ctx["obj"].state = CommunicatorState()

    def _assign_task_to_processor(
        state: "Simulator.State", task: Task, processor: Processor, start_time: float
    ):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        processor.state.idle_time += start_time - processor.state.end_time
        processor.state.current_task = task
        task.state.start_time = start_time
        if isinstance(task, Task):
            task.state.end_time = start_time + task.cost / processor.speed
        else:
            task.state.end_time = (
                start_time + task.size / processor.bandwidth + processor.latency
            )
        processor.state.end_time = task.state.end_time
        processor.state.next_task_index += 1
        state.sim.add_event(
            task.state.end_time, Simulator._on_task_complete, state, task
        )
        # print(f"Task {task.name} will finish at time {task.state.end_time}")

    def _on_task_complete(time: float, state: "Simulator.State", task: Task):
        # print(f"Task {task.name} completed at time {time}")
        task.state.finished = True
        state.ntasks_complete += 1

        if isinstance(task, Task):
            processor = state.schedule.get_processor(task.name)
            state.context_graph.node_dict[processor].state.current_task = None
        else:
            communicator = task.state.communicator
            communicator.state.current_task = None

        Simulator._assign_idle_processors_and_communicators(state, time)

    def _is_task_eligible(state: "Simulator.State", task: Task):
        return all(t.state.finished for t in state.task_graph.predecessors(task))

    def _assign_idle_processors_and_communicators(
        state: "Simulator.State", time: float
    ):
        for processor in state.context_graph:
            if processor.state.current_task is None:
                if processor.state.next_task_index >= len(
                    state.schedule.task_allocation[processor.name]
                ):
                    continue
                next_task_name = state.schedule.task_allocation[processor.name][
                    processor.state.next_task_index
                ]
                next_task = state.task_graph.get_node(next_task_name)
                if Simulator._is_task_eligible(state, next_task):
                    Simulator._assign_task_to_processor(
                        state, next_task, processor, time
                    )

        for _, _, communicator in state.context_graph.edges(data=True):
            communicator = communicator["obj"]
            if communicator.state.current_task is None:
                # Check all tasks, because communications are not ordered
                for task in communicator.state.tasks:
                    if communicator.state.current_task is not None:
                        break
                    if task.state.finished:
                        continue
                    if Simulator._is_task_eligible(state, task):
                        Simulator._assign_task_to_processor(
                            state, task, communicator, time
                        )

    def execute(
        schedule: Schedule, *, with_communication: bool = False
    ) -> ExecutionReport:
        state = Simulator.State(schedule)

        if with_communication:
            for start, end in state.task_graph.edges():
                start_processor = state.context_graph.node_dict[
                    schedule.get_processor(start.name)
                ]
                end_processor = state.context_graph.node_dict[
                    schedule.get_processor(end.name)
                ]
                if start_processor != end_processor:
                    t = state.task_graph._make_communication_task(start, end)
                    t.state = CommunicationState()
                    # find the communicator which can handle this communication
                    ctx = state.context_graph.get_edge_data(
                        start_processor, end_processor
                    )["obj"]
                    t.state.communicator = ctx
                    ctx.state.tasks.append(t)
                    state.total_tasks += 1

        Simulator._assign_idle_processors_and_communicators(state, time=0)
        state.sim.run()

        # print(f"Finished {self.ntasks_complete} tasks out of {self.total_tasks}")
        assert state.ntasks_complete == state.total_tasks

        return ExecutionReport(state.schedule, state.task_graph, state.context_graph)
