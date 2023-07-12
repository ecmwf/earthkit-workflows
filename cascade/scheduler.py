import copy
from dataclasses import dataclass, field
import randomname
import datetime

from .graphs import Task, Communication, Processor, Communicator
from .utility import EventLoop


####################################################################################################


@dataclass
class TaskState:
    processor: Processor = None
    finished: bool = False
    start_time: float = 0
    end_time: float = 0


@dataclass
class CommunicationState:
    communicator: Communicator = None
    finished: bool = False
    start_time: float = 0
    end_time: float = 0


@dataclass
class ProcessorState:
    tasks: list[Task] = field(default_factory=list)
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


class Schedule:
    def __init__(self, task_graph, context_graph):
        self.task_graph = task_graph
        self.context_graph = context_graph
        self.name = randomname.get_name()
        self.created_at = datetime.datetime.utcnow()
    
    def __repr__(self) -> str:
        str = f"============= Schedule: {self.name} =============\n"
        str += f"Created at: {self.created_at} UTC\n"
        for processor in self.context_graph:
            str += f"Processor {processor.name} completes at {processor.state.end_time:.2f} ({processor.state.idle_time:.2f} idle):\n"
            str += " → ".join(task.name for task in processor.state.tasks) + "\n"
        str += "================================================\n"

        return str


####################################################################################################


class Scheduler:
    def __init__(self, task_graph, context_graph):
        self.task_graph = task_graph  # copy.deepcopy(task_graph)
        self.context_graph = context_graph  # copy.deepcopy(context_graph)

    def reset_state(self):
        for ctx in self.task_graph:
            ctx.state = TaskState()
        for _,_,ctx in self.task_graph.edges(data=True):
            ctx["obj"].state = CommunicationState()
        for ctx in self.context_graph:
            ctx.state = ProcessorState()
        for _,_,ctx in self.context_graph.edges(data=True):
            ctx["obj"].state = CommunicatorState()

    def create_schedule(self) -> Schedule:
        raise NotImplementedError()


####################################################################################################
        

class DepthFirstScheduler(Scheduler):
    def __init__(self, task_graph, context_graph):
        super().__init__(task_graph, context_graph)

    def assign_task_to_processor(self, task, processor, start_time):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        processor.state.idle_time += processor.state.end_time - start_time
        processor.state.tasks.append(task)
        processor.state.current_task = task
        task.state.processor = processor
        task.state.start_time = start_time
        task.state.end_time = start_time + task.cost / processor.speed
        processor.state.end_time = task.state.end_time
        self.sim.add_event(task.state.end_time, self.on_task_complete, task)

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time} by processor {task.state.processor.name}")
        task.state.finished = True
        self.ntasks_complete += 1
        for dependent in self.task_graph.successors(task):
            if all(t.state.finished for t in self.task_graph.predecessors(dependent)):
                self.eligible.append(dependent)

        processor = task.state.processor
        processor.state.current_task = None

        self.assign_idle_processors(time)

    def assign_idle_processors(self, time):
        # Sort next tasks
        self.eligible.sort(key=lambda x: (x.cost, self.task_graph.out_degree[x]))

        # Assign idle processors
        for processor in self.context_graph:
            if processor.state.current_task is None:
                if len(self.eligible) > 0:
                    self.assign_task_to_processor(self.eligible.pop(), processor, time)

    def create_schedule(self):
        self.reset_state()
        self.eligible = self.task_graph.get_roots()
        self.ntasks_complete = 0
        self.sim = EventLoop()
        self.assign_idle_processors(time=0)
        self.sim.run()
        # print(f"Finished {self.ntasks_complete} tasks out of {len(self.task_graph)}")
        assert self.ntasks_complete == len(self.task_graph)

        s = Schedule(self.task_graph, self.context_graph)
        return s


class AnnealingScheduler(Scheduler):
    def __init__(self, task_graph, context_graph):
        super().__init__(task_graph, context_graph)

    def create_schedule(self):
        self.reset_state()
        
        # will use a dumb scheduler internally as a start point
        schedule = DepthFirstScheduler(self.task_graph, self.context_graph).create_schedule()
        self.task_graph = schedule.task_graph
        self.context_graph = schedule.context_graph

        # will use an executor internally as a cost function





####################################################################################################

# class TaskGraph
# class HardwareGraph

# class Scheduler(graph, hardwaregraph) -> Schedule
#   DumbScheduler
#   AnnealingScheduler(costfunction)

# class Executor(schedule, hardwaregraph) -> Execution/Cost
#   SimulatedExecutor


# class ExecutionOptimizer:
#     def __init__(self, task_graph, context_graph):
#         self.task_graph = copy.deepcopy(task_graph)
#         self.context_graph = copy.deepcopy(context_graph)
#         assert self.task_graph.is_directed()

#         self.ideal_execution_time = float("NaN")

#     def first_guess(self):

#         for ctx in self.context_graph:
#             ctx.reset()

#         for task in self.task_graph:
#             task.reset()

#         ntasks_processed = 0
        
#         # Get all the nodes which have no incoming edges
#         eligible = [n for n, d in self.task_graph.in_degree() if d == 0]
#         current_time = 0

#         while ntasks_processed < len(self.task_graph):

#             # Find the context with the earliest available time
#             ctx = min(self.context_graph, key=lambda x: x.update_time)

#             current_time = ctx.update_time

#             if len(ctx.task_list) > 0:
#                 task = ctx.task_list[-1]
#                 task.end_time = current_time
#                 task.finished = True

#                 # Check dependents of this just-completed task, and see if any of them are eligible to run
#                 # All their inputs must be finished
#                 for dependent in self.task_graph.successors(task):
#                     if all(t.finished for t in self.task_graph.predecessors(dependent)):
#                         # print(f"Task {dependent.name} is now eligible")
#                         eligible.append(dependent)

            
#             # It can be the case that nothing is eligible, and the task has to wait
#             # Set the wait time to the next highest available time
#             if len(eligible) == 0:
#                 ctx.update_time = min([c.update_time if c.update_time > current_time else float("inf") for c in self.context_graph])
#                 continue


#             # Find the eligible node with the highest cost and most dependents
#             eligible.sort(key=lambda x: (x.cost, self.task_graph.out_degree[x]))

#             # Assign the node to the context
#             task = eligible.pop()
#             task.start_time = current_time
#             ctx.task_list.append(task)
#             task.context = ctx
#             ctx.update_time += task.cost / ctx.speed
#             ntasks_processed += 1

#             # print(f"Context {ctx.name} has task {ctx.task_list[-1].name}, next available time is {ctx.update_time:.2f}s")

#         for ctx in self.context_graph:
#             print(f"{Fore.RED}{ctx.name}{Fore.RESET} has total execution time of {Fore.RED}{ctx.update_time:.2f}s{Fore.RESET}")
#             print(" 🡆 ".join([task.name for task in ctx.task_list]))

#         for t in self.task_graph:
#             assert t.context is not None
#         assert ntasks_processed == len(self.task_graph)

#         self.ideal_execution_time = max([ctx.update_time for ctx in self.context_graph])

    # def cost(self):


    #     # We should break down the edges into nodes with a cost
    #     # Or the communication context should be a worker

    #     simulation_timesteps = SortedList()
    #     simulation_timesteps.add(0)

    #     # Reset all the contexts and tasks
    #     for ctx in self.context_graph:
    #         ctx.reset(keep_tasks=True)
    #     for task in self.task_graph:
    #         task.reset()

    #     current_time = 0
    #     ntasks_processed = 0
    #     ntasks_total = len(self.task_graph)

    #     # For each context, seed the initial eligible tasks (no dependencies)
    #     for ctx in self.context_graph:
    #         ctx.eligible_tasks = [n for n, d in self.task_graph.in_degree(ctx.task_list) if d == 0]

    #     while len(simulation_timesteps) > 0:

    #         current_time = simulation_timesteps[0]

    #         # Loops all contexts which need to be updated at this timestep
    #         for ctx in self.context_graph:
    #             if ctx.update_time == current_time:

    #                 # The current task is now complete
    #                 if ctx.current_task is not None:
    #                     task = ctx.current_task
    #                     if task.start_time + task.cost / ctx.speed < current_time:
    #                         continue  # task is not finished yet
    #                     task.end_time = current_time
    #                     task.finished = True
    #                     ntasks_processed += 1
    #                     ctx.current_task = None
    #                     print(f"Task {task.name} finished at {current_time:.2f}s ({ntasks_processed}/{ntasks_total})")
    #                     # if ntasks_processed == ntasks_total:
    #                     #     break

    #                     # Check dependents of this just-completed task, and see if any of them are eligible to run
    #                     # All their inputs must be finished
    #                     for dependent in self.task_graph.successors(task):
    #                         if all(t.finished for t in self.task_graph.predecessors(dependent)):
    #                             # We need to add it to the eligibile list for the context where this task will run
    #                             dependent.context.eligible_tasks.append(dependent)
            
            
    #         for ctx in self.context_graph:
    #             if ctx.update_time == current_time:

    #                 # Now check for new tasks that we can run, taking into account communication times
    #                 communication_complete = []
    #                 next_comm_completion = float("inf")
    #                 for task in ctx.eligible_tasks:

    #                     # Check when all dependencies would have finished communication, or when they will finish
    #                     task_ready_time = float("-inf")
    #                     for dep in self.task_graph.predecessors(task):
                            
    #                         # if tasks are on the same context, there is no communication
    #                         if (dep.context == task.context):
    #                             completion_time = dep.end_time
    #                             print(f"Task {task.name} depends on {dep.name} which will finish at {dep.end_time:.2f}s, communication time is 0s, current time is {current_time:.2f}s")
    #                         # else find the communication context and find the communication time
    #                         else:
    #                             comm_size = self.task_graph[dep][task]["obj"].size
    #                             bandwidth = self.context_graph[task.context][dep.context]["obj"].bandwidth
    #                             latency = self.context_graph[task.context][dep.context]["obj"].latency
    #                             comm_time = comm_size / bandwidth + latency
    #                             completion_time = dep.end_time + comm_time
    #                             print(f"Task {task.name} depends on {dep.name} which will finish at {dep.end_time:.2f}s, communication time is {comm_time:.2f}s, current time is {current_time:.2f}s")

    #                         print(f"Task {task.name} completion time is {completion_time:.2f}s, task ready time is {task_ready_time:.2f}s")
                            
    #                         task_ready_time = max(task_ready_time, completion_time)
                        
    #                     # If ready, add to list
    #                     if task_ready_time <= current_time:
    #                         communication_complete.append(task)
                        
    #                     # Find the time of the soonest next-ready task
    #                     next_comm_completion = min(next_comm_completion, task_ready_time)

    #                 # print(f"Ready time is {next_comm_completion:.2f}s")
    #                 # print(f"number of eligible tasks is {len(communication_complete)}")

    #                 # We can process something
    #                 if len(communication_complete) > 0:
    #                     communication_complete.sort(key=lambda x: (x.cost, self.task_graph.out_degree[x]))
    #                     task = communication_complete.pop()
    #                     ctx.eligible_tasks.remove(task)
    #                     ctx.current_task = task
    #                     task.start_time = current_time
    #                     ctx.update_time += task.cost / ctx.speed
    #                     print(f"Context {ctx.name} has task {task.name}, next available time is {ctx.update_time:.2f}s")
                    
    #                 # We need to wait for communication
    #                 elif next_comm_completion < float("inf"):
    #                     ctx.update_time = next_comm_completion

    #                 # We need to wait for tasks to complete
    #                 else:
    #                     ctx.update_time = simulation_timesteps[1]  # contexts not processed yet still be on step=0
    #                     continue

    #                 if ctx.update_time not in simulation_timesteps:
    #                     simulation_timesteps.add(ctx.update_time)

    #         simulation_timesteps.pop(0)

    #     for ctx in self.context_graph:
    #         print(f"{Fore.RED}{ctx.name}{Fore.RESET} has total execution time of {Fore.RED}{ctx.update_time:.2f}s{Fore.RESET}")
    #         print(" 🡆 ".join([task.name for task in ctx.task_list]))
        
    #     print(f"ideal execution time is {self.ideal_execution_time:.2f}s")
    #     # NB if a task on another context finishes and has very fast communication, we don't check if it might jump in while waiting


                    

            
            

