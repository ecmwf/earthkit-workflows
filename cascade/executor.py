import copy
from dataclasses import dataclass, field
import randomname
import datetime

from pproc.common.resources import ResourceMeter

from .graphs import Task, Communication, Communicator, to_execution_graph
from .utility import EventLoop


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


class Executor:
    def __init__(self, schedule, with_communication: bool = True):
        self.schedule = schedule

        self.task_graph = to_execution_graph(schedule.task_graph)
        self.context_graph = copy.deepcopy(schedule.context_graph)
        self.with_communication = with_communication

    def reset_state(self):
        for task in self.task_graph.nodes():
            task.state = TaskState()
        for ctx in self.context_graph:
            ctx.state = ProcessorState()
        for _, _, ctx in self.context_graph.edges(data=True):
            ctx["obj"].state = CommunicatorState()

    def execute(self) -> ExecutionReport:
        raise NotImplementedError()

    def simulate(self) -> ExecutionReport:
        raise NotImplementedError()


class BasicExecutor(Executor):
    """The basic executor executes tasks in the order they are given by the schedule.
    It does not dictate the order of communications, it just begins communications
    as soon as data becomes available."""

    def assign_task_to_processor(self, task, processor, start_time):
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
        self.sim.add_event(task.state.end_time, self.on_task_complete, task)
        # print(f"Task {task.name} will finish at time {task.state.end_time}")

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time}")
        task.state.finished = True
        self.ntasks_complete += 1

        if isinstance(task, Task):
            processor = self.schedule.get_processor(task.name)
            self.context_graph.node_dict[processor].state.current_task = None
        else:
            communicator = task.state.communicator
            communicator.state.current_task = None

        self.assign_idle_processors_and_communicators(time)

    def is_task_eligible(self, task):
        return all(t.state.finished for t in self.task_graph.predecessors(task))

    def assign_idle_processors_and_communicators(self, time):
        for processor in self.context_graph:
            if processor.state.current_task is None:
                if processor.state.next_task_index >= len(
                    self.schedule.task_allocation[processor.name]
                ):
                    continue
                next_task_name = self.schedule.task_allocation[processor.name][
                    processor.state.next_task_index
                ]
                next_task = self.task_graph.get_node(next_task_name)
                if self.is_task_eligible(next_task):
                    self.assign_task_to_processor(next_task, processor, time)

        for _, _, communicator in self.context_graph.edges(data=True):
            communicator = communicator["obj"]
            if communicator.state.current_task is None:
                # Check all tasks, because communications are not ordered
                for task in communicator.state.tasks:
                    if communicator.state.current_task is not None:
                        break
                    if task.state.finished:
                        continue
                    if self.is_task_eligible(task):
                        self.assign_task_to_processor(task, communicator, time)

    def simulate(self) -> ExecutionReport:
        self.reset_state()

        self.total_tasks = len(list(self.task_graph.nodes()))

        if self.with_communication:
            for start, end in self.task_graph.edges():
                start_processor = self.context_graph.node_dict[
                    self.schedule.get_processor(start.name)
                ]
                end_processor = self.context_graph.node_dict[
                    self.schedule.get_processor(end.name)
                ]
                if start_processor != end_processor:
                    t = self.task_graph._make_communication_task(start, end)
                    t.state = CommunicationState()
                    # find the communicator which can handle this communication
                    ctx = self.context_graph.get_edge_data(
                        start_processor, end_processor
                    )["obj"]
                    t.state.communicator = ctx
                    ctx.state.tasks.append(t)
                    self.total_tasks += 1

        self.ntasks_complete = 0
        self.sim = EventLoop()
        self.assign_idle_processors_and_communicators(time=0)
        self.sim.run()

        # print(f"Finished {self.ntasks_complete} tasks out of {self.total_tasks}")
        assert self.ntasks_complete == self.total_tasks

        return ExecutionReport(self.schedule, self.task_graph, self.context_graph)

    def determine_resources(self) -> ExecutionReport:
        assert len(self.schedule.task_allocation) == 1

        for task_name in sum(self.schedule.task_allocation.values(), []):
            task = self.task_graph.get_node(task_name)
            schedule_task = self.schedule.task_graph.get_node(task.name)
            payload = task.payload
            with ResourceMeter() as rm:
                arguments = payload[1:]
            print("NODE", task_name, "PAYLOAD", payload, "ARGS", len(arguments))
            schedule_task.in_memory = rm.mem
            with ResourceMeter() as rm:
                output = payload[0](*arguments)

            print("OUTPUT", output)
            schedule_task.cost = rm.elapsed_cpu
            schedule_task.out_memory = rm.mem

            # Pass output to arguments in payloads requiring it
            successors = self.task_graph.successors(task)
            for successor in successors:
                assert not isinstance(
                    successor.payload, str
                ), f"Payload can not be str. Got {successor.payload}"
                if not hasattr(successor.payload, "__iter__"):
                    successor.payload = tuple(
                        [successor.payload] + list(successor.inputs.keys())
                    )
                for iname, input in successor.inputs.items():
                    if input.parent == task:
                        successor.payload = tuple(
                            output if (isinstance(x, str) and x == iname) else x
                            for x in successor.payload
                        )
                        break
            task.payload = None
