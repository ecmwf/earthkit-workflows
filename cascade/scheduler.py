import copy
import randomname
import datetime
import random
import numpy as np
import networkx as nx
import math

from .utility import EventLoop
from .executor import BasicExecutor


class Schedule:
    def __init__(self, task_graph, context_graph, task_allocation):
        self.task_graph = task_graph
        self.context_graph = context_graph
        self.task_allocation = task_allocation
        self.name = randomname.get_name()
        self.created_at = datetime.datetime.utcnow()

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

    def valid_allocations(self) -> bool:
        dependency_graph = copy.deepcopy(self.task_graph)

        for _, p_tasks in self.task_allocation.items():
            for i_task, task in enumerate(p_tasks):
                if i_task < len(p_tasks) - 1:
                    current_task = self.task_graph.node_dict[task]
                    next_task = self.task_graph.node_dict[p_tasks[i_task + 1]]
                    dependency_graph.add_edge(current_task, next_task)

        return not nx.dag.has_cycle(dependency_graph)


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
        self.completed_tasks = set()

    def assign_task_to_processor(self, task, processor, start_time):
        # print(f"Task {task.name} assigned to processor {processor.name} at time {start_time}")
        self.task_allocation[processor.name].append(task.name)
        end_time = start_time + task.cost / processor.speed
        self.sim.add_event(end_time, self.on_task_complete, task)

    def on_task_complete(self, time, task):
        # print(f"Task {task.name} completed at time {time} by processor {processor.name}")
        self.completed_tasks.add(task.name)
        for dependent in self.task_graph.successors(task):
            if all(
                t.name in self.completed_tasks
                for t in self.task_graph.predecessors(dependent)
            ):
                self.eligible.append(dependent)

        self.assign_idle_processors(time)

    def assign_idle_processors(self, time):
        # Sort next tasks
        self.eligible.sort(key=lambda x: (x.cost, self.task_graph.out_degree[x]))

        # Assign idle processors
        for processor in self.context_graph:
            if (
                len(self.task_allocation[processor.name]) == 0
                or self.task_allocation[processor.name][-1] in self.completed_tasks
            ):
                if len(self.eligible) > 0:
                    self.assign_task_to_processor(self.eligible.pop(), processor, time)

    def create_schedule(self):
        self.completed_tasks = set()
        self.task_allocation = {p.name: [] for p in self.context_graph}
        self.eligible = self.task_graph.get_roots()
        self.sim = EventLoop()
        self.assign_idle_processors(time=0)
        self.sim.run()
        # print(f"Finished {self.ntasks_complete} tasks out of {len(self.task_graph)}")
        assert len(self.completed_tasks) == len(self.task_graph)

        s = Schedule(self.task_graph, self.context_graph, self.task_allocation)
        return s


class AnnealingScheduler(Scheduler):
    def __init__(self, task_graph, context_graph):
        super().__init__(task_graph, context_graph)

    def create_schedule(self):
        # will use a dumb scheduler internally as a start point
        schedule = DepthFirstScheduler(
            self.task_graph, self.context_graph
        ).create_schedule()
        assert schedule.valid_allocations()

        # will use an executor internally as a cost function
        return self.anneal(schedule)

    def update_temperature(
        iteration: int, n_tries: int, init_temp: float, geometric_factor: float
    ):
        return init_temp * geometric_factor * math.floor(iteration / n_tries)

    def anneal(self, schedule):
        n_temp = 100
        n_tries = 100
        n_cont = 10
        temp_factor = 0.9
        temp = 0.5

        executor = BasicExecutor(schedule)
        report = executor.simulate()
        print("Initial Report\n", report)
        initial_cost = report.cost()
        previous_cost = report.cost()

        num_processes = len(self.context_graph)
        processors = list(schedule.task_allocation.keys())
        for i_temp in range(n_temp):
            n_success = 0
            for _ in range(n_tries):
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

                    new_schedule = Schedule(
                        self.task_graph, self.context_graph, new_task_allocations
                    )
                    valid_schedule = new_schedule.valid_allocations()

                all_allocated_tasks = set()
                for allocations in new_task_allocations.values():
                    all_allocated_tasks.update(allocations)
                assert len(self.task_graph) == len(all_allocated_tasks)

                # Check to see if new schedule is improvement
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
                    n_success += 1

                if n_success > n_cont:
                    break

            temp = temp * temp_factor
            if n_success == 0:
                break

        print("Annealed Report\n", report)
        print(f"Initial Cost: {initial_cost}, Annealed Cost: {new_cost}")
        return schedule
