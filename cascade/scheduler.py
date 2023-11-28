import copy
import randomname
import datetime
import random
import numpy as np

from .utility import EventLoop
from .executor import BasicExecutor
from .graph.copy import copy_graph
from .graphs import Task, ContextGraph, TaskGraph


class Schedule:
    def __init__(
        self, task_graph: TaskGraph, context_graph: ContextGraph, task_allocation: dict
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
        cls, task_graph: TaskGraph, task_allocation: dict[str, list]
    ) -> bool:
        dependency_graph = copy_graph(task_graph)

        for _, p_tasks in task_allocation.items():
            for i_task, task in enumerate(p_tasks):
                if i_task < len(p_tasks) - 1:
                    current_task = dependency_graph.get_node(task)
                    next_task = dependency_graph.get_node(p_tasks[i_task + 1])
                    key = f"inputs{len(next_task.inputs)}"
                    assert key not in next_task.inputs
                    next_task.inputs[key] = current_task.get_output()
                    if current_task in dependency_graph.sinks:
                        dependency_graph.sinks.remove(current_task)
                    # Check number of nodes as disconnected components may have been introduced
                    if dependency_graph.has_cycle() or len(
                        list(dependency_graph.nodes())
                    ) != len(list(task_graph.nodes())):
                        return False
        return True


####################################################################################################


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


####################################################################################################


class Scheduler:
    def __init__(self, task_graph, context_graph):
        self.task_graph = task_graph
        self.context_graph = context_graph

    def create_schedule(self) -> Schedule:
        raise NotImplementedError()


####################################################################################################


class DepthFirstScheduler(Scheduler):
    def __init__(self, task_graph, context_graph):
        super().__init__(task_graph, context_graph)
        self.task_allocation = {p.name: [] for p in context_graph}
        self.memory_usage = {p.name: MemoryUsage() for p in self.context_graph}
        self.completed_tasks = set()

    def assign_task_to_processor(self, task, processor, start_time):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        self.task_allocation[processor.name].append(task.name)
        end_time = start_time + task.cost / processor.speed
        self.memory_usage[processor.name].add_task(task)
        self.sim.add_event(end_time, self.on_task_complete, task)

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time} by processor {processor.name}")
        self.completed_tasks.add(task.name)
        successors = self.task_graph.successors(task)

        for dependent in successors:
            predecessors = self.task_graph.predecessors(dependent)
            if all(t.name in self.completed_tasks for t in predecessors):
                self.eligible.append(dependent)

        self.assign_idle_processors(time)

    def update_memory_usage(self, processor) -> float:
        for task in self.memory_usage[processor.name].current_tasks():
            successors = self.task_graph.successors(task)
            if all(
                [x in self.completed_tasks or x in self.eligible for x in successors]
            ):
                self.memory_usage[processor.name].remove_task(task)

        return self.memory_usage[processor.name].memory

    def assign_idle_processors(self, time):
        # Sort next tasks
        self.eligible.sort(key=lambda x: (x.cost, len(x.inputs)))

        # Assign idle processors
        for processor in self.context_graph:
            new_mem_usage = self.update_memory_usage(processor)
            if (
                len(self.task_allocation[processor.name]) == 0
                or self.task_allocation[processor.name][-1] in self.completed_tasks
            ):
                if (
                    len(self.eligible) > 0
                    and (new_mem_usage + self.eligible[0].memory) < processor.memory
                ):
                    self.assign_task_to_processor(self.eligible.pop(), processor, time)

    def create_schedule(self):
        self.completed_tasks = set()
        self.task_allocation = {p.name: [] for p in self.context_graph}
        self.eligible = list(self.task_graph.sources())
        self.sim = EventLoop()
        self.assign_idle_processors(time=0)
        self.sim.run()
        print(
            f"Finished {len(self.completed_tasks)} tasks out of {len(list(self.task_graph.nodes()))}"
        )
        assert len(self.completed_tasks) == len(list(self.task_graph.nodes()))

        s = Schedule(self.task_graph, self.context_graph, self.task_allocation)
        return s


class AnnealingScheduler(Scheduler):
    def __init__(self, task_graph, context_graph):
        super().__init__(task_graph, context_graph)

    def create_schedule(
        self,
        num_temp_levels: int = 100,
        num_tries: int = 100,
        num_success_cutoff: int = 10,
        init_temp: float = 0.5,
        temp_factor: float = 0.9,
    ) -> Schedule:
        """
        Create schedule using simulated annealing to minimise execution cost, determined by simulating the
        schedule using BasicExecutor. Uses depth first schedule as initial starting point. Moves currently
        consist only of randomly picking two processors and one task in each and swapping them. Uses the
        normalised exponential form for the acceptance function.

        :param num_temp_levels: number of temperature levels to iterate over
        :param num_tries: number of iterations to perform per temperature level
        :param num_success_cutoff: number of successful iterations at each level after which
        to proceed onto the next temperature level
        :param init_temp: initial temperature between 0 and 1
        :param temp_factor: factor for determining temperature of next level in the formula
            T_{i+1} = T_{i} * temp_factor, where i is the iteration index
        :return: schedule output from annealing algorithm
        """

        # Determine initial conditions
        schedule = DepthFirstScheduler(
            self.task_graph, self.context_graph
        ).create_schedule()
        executor = BasicExecutor(schedule)
        report = executor.simulate()
        print("Initial Report\n", report)
        initial_cost = report.cost()
        previous_cost = report.cost()

        temp = init_temp
        num_processes = len(self.context_graph)
        processors = list(schedule.task_allocation.keys())
        for i_temp in range(num_temp_levels):
            num_success = 0
            for _ in range(num_tries):
                valid_schedule = False
                while not valid_schedule:
                    new_task_allocations = copy.deepcopy(schedule.task_allocation)

                    # Randomly pick two tasks to permute
                    i_p1 = processors[random.randrange(0, num_processes)]
                    i_task1 = random.randrange(0, len(new_task_allocations[i_p1]))
                    task1 = new_task_allocations[i_p1][i_task1]

                    i_p2 = processors[random.randrange(0, num_processes)]
                    i_task2 = random.randrange(0, len(new_task_allocations[i_p2]))

                    new_task_allocations[i_p1][i_task1] = new_task_allocations[i_p2][
                        i_task2
                    ]
                    new_task_allocations[i_p2][i_task2] = task1

                    valid_schedule = Schedule.valid_allocations(
                        self.task_graph, new_task_allocations
                    )

                all_allocated_tasks = set()
                for allocations in new_task_allocations.values():
                    all_allocated_tasks.update(allocations)
                assert len(list(self.task_graph.nodes())) == len(all_allocated_tasks)

                # Check to see if new schedule is improvement
                new_schedule = Schedule(
                    self.task_graph, self.context_graph, new_task_allocations
                )
                executor = BasicExecutor(new_schedule)
                new_report = executor.simulate()
                new_cost = new_report.cost()

                rand = random.random()
                delta_cost = new_cost - previous_cost
                prob = np.exp(-delta_cost / (initial_cost * temp))
                if delta_cost < 0 or rand < prob:
                    # print(f"Temp {i_temp}, Success {n_success}: Updated schedule cost change: {delta_cost}, rand: {rand}, prob: {prob}")
                    previous_cost = new_cost
                    schedule = new_schedule
                    report = new_report
                    num_success += 1

                if num_success > num_success_cutoff:
                    break

            print(
                f"Temperature iteration {i_temp}: temp {temp}, new cost {previous_cost}"
            )
            temp = temp * temp_factor
            if num_success == 0:
                break

        print("Annealed Report\n", report)
        print(f"Initial Cost: {initial_cost}, Annealed Cost: {new_cost}")
        return schedule
