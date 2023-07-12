import randomname
import datetime

from .graphs import Task
from .scheduler import CommunicationState
from .utility import EventLoop


class ExecutionReport:
    def __init__(self, schedule):
        self.schedule = schedule
        self.task_graph = schedule.task_graph
        self.context_graph = schedule.context_graph
        self.name = randomname.get_name()
        self.created_at = datetime.datetime.utcnow()
    
    def __repr__(self) -> str:
        str = f"============= Execution Report: {self.name} =============\n"
        str += f"Using schedule: {self.schedule.name}\n"
        str += f"Created at: {self.created_at} UTC\n"
        for processor in self.context_graph:
            str += f"Processor {processor.name} completes at {processor.state.end_time:.2f} ({processor.state.idle_time:.2f} idle):\n"
            str += " → ".join(task.name for task in processor.state.tasks) + "\n"
        str += "================================================\n"
        str += "Note: Idle times currently incorrect.\n"

        return str


class Executor:
    def __init__(self, schedule):
        self.schedule = schedule

        # TODO: need to do a deepcopy here but keeping task assignments to processors
        #       the links break if we try this at the moment
        #       schedule should be const
        self.task_graph = schedule.task_graph
        self.context_graph = schedule.context_graph

    def reset_state(self):
        for task in self.task_graph:
            task.state.finished = False
            task.state.start_time = 0
            task.state.end_time = 0
        for _,_,communication in self.task_graph.edges(data=True):
            communication["obj"].state.finished = False
            communication["obj"].state.start_time = 0
            communication["obj"].state.end_time = 0
        for processor in self.context_graph:
            processor.state.idle_time = 0
            processor.state.current_task = None
            processor.state.next_task_index = 0
        for _,_,communicator in self.context_graph.edges(data=True):
            communicator["obj"].state.idle_time = 0
            communicator["obj"].state.current_task = None
            communicator["obj"].state.next_task_index = 0
        
    def execute(self) -> ExecutionReport:
        raise NotImplementedError()
    
    def simulate(self) -> ExecutionReport:
        raise NotImplementedError()


class BasicExecutor(Executor):
    """ The basic executor executes tasks in the order they are given by the schedule.
        It does not dictate the order of communications, it just begins communications
        as soon as data becomes available."""

    def assign_task_to_processor(self, task, processor, start_time):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        processor.state.idle_time += processor.state.end_time - start_time
        processor.state.current_task = task
        task.state.processor = processor
        task.state.start_time = start_time
        if isinstance(task, Task):
            task.state.end_time = start_time + task.cost / processor.speed
        else:
            task.state.end_time = start_time + task.size / processor.bandwidth + processor.latency
        processor.state.end_time = task.state.end_time
        processor.state.next_task_index += 1
        self.sim.add_event(task.state.end_time, self.on_task_complete, task)
        # print(f"Task {task.name} will finish at time {task.state.end_time}")

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time} by processor {task.state.processor.name}")
        task.state.finished = True
        self.ntasks_complete += 1

        processor = task.state.processor
        processor.state.current_task = None

        self.assign_idle_processors_and_communicators(time)

    def is_task_eligible(self, task):
        if all(t.state.finished for t in self.task_graph.predecessors(task)):
            return True
        return False

    def assign_idle_processors_and_communicators(self, time):
        for processor in self.context_graph:
            if processor.state.current_task is None:
                if processor.state.next_task_index >= len(processor.state.tasks):
                    continue
                next_task = processor.state.tasks[processor.state.next_task_index]
                if self.is_task_eligible(next_task):
                    self.assign_task_to_processor(next_task, processor, time)
        
        for _,_,communicator in self.context_graph.edges(data=True):
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

        self.total_tasks = len(self.task_graph)

        for start, end, edge in list(self.task_graph.edges(data=True)):
            if start.state.processor == end.state.processor:
                continue
            else:
                t = self.task_graph._make_communication_task(start, end, edge["obj"])
                t.state = CommunicationState()
                # find the communicator which can handle this communication
                ctx = self.context_graph.get_edge_data(start.state.processor, end.state.processor)["obj"]
                t.state.communicator = ctx
                ctx.state.tasks.append(t)
                self.total_tasks += 1

        # self.task_graph.draw("with_communications.png")

        self.ntasks_complete = 0
        self.sim = EventLoop()
        self.assign_idle_processors_and_communicators(time=0)
        self.sim.run()

        # for processor in self.context_graph:
        #     processor.state.idle_time += processor.state.end_time - self.sim.time
        #     processor.state.end_time = self.sim.time

        # print(f"Finished {self.ntasks_complete} tasks out of {self.total_tasks}")
        assert self.ntasks_complete == self.total_tasks

        return ExecutionReport(self.schedule)
        
    def execute(self) -> ExecutionReport:
        raise NotImplementedError()