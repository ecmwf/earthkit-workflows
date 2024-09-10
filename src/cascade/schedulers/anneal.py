import copy
import random
from typing import Callable, cast

import numpy as np

from cascade.contextgraph import ContextGraph
from cascade.graph import Graph
from cascade.taskgraph import TaskGraph

from .depthfirst import DepthFirstScheduler
from .schedule import Schedule
from .simulate import Simulator


class AnnealingScheduler:
    """
    Simulated annealing scheduler to determine optimal schedule for a given task graph and context graph.
    Configurable options for each instance are:
        cost_function: callable[[Schedule], float], function to determine cost of schedule. Cost should be
        positive with minimum at 0.
        num_temp_levels: int, number of temperature levels to iterate over
        num_tries: int, number of iterations to perform per temperature level
        num_success_cutoff: int, number of successful iterations at each level after which
        to proceed onto the next temperature level
        init_temp: float, initial temperature between 0 and 1
        temp_factor: float, factor for determining temperature of next level in the formula
            T_{i+1} = T_{i} * temp_factor, where i is the iteration index
    """

    def __init__(
        self,
        cost_function: Callable,
        num_temp_levels: int = 100,
        num_tries: int = 100,
        num_success_cutoff: int = 10,
        init_temp: float = 0.5,
        temp_factor: float = 0.9,
    ):
        self.cost_function = cost_function
        self.num_temp_levels = num_temp_levels
        self.num_tries = num_tries
        self.num_success_cutoff = num_success_cutoff
        self.init_temp = init_temp
        self.temp_factor = temp_factor

    @staticmethod
    def total_idle_time(schedule: Schedule) -> float:
        """
        Cost associated to simulated execution of the schedule in terms of the total idle time
        of all processors in the context graph

        Params
        ------
        schedule: Schedule, schedule to be evaluated

        Returns
        -------
        float
        """
        _, context_state = Simulator().execute(schedule, with_communication=True)
        return sum(
            [processor.state.idle_time for processor in context_state.context_graph]
        )

    @staticmethod
    def total_execution_time(schedule: Schedule) -> float:
        """
        Cost associated to simulated execution of the schedule in terms of the total execution
        time

        Params
        ------
        schedule: Schedule, schedule to be evaluated

        Returns
        -------
        float
        """
        start: float | None = None
        end = 0.0
        execution_state, _ = Simulator().execute(schedule, with_communication=True)
        for task in execution_state.task_graph.sources():
            if not hasattr(
                task.state, "start_time"
            ):  # runtime patched somewhere I guess
                raise TypeError
            p_start = cast(float, task.state.start_time)
            start = min(start, p_start) if start is not None else p_start

        for sink in execution_state.task_graph.sinks:
            if not hasattr(sink.state, "end_time"):  # runtime patched somewhere I guess
                raise TypeError
            end = max(end, sink.state.end_time)
        # we cast because it could be None... which should crash runtime, but somehow in tests doesnt
        return end - cast(float, start)

    @staticmethod
    def permute(schedule: Schedule) -> Schedule:
        """
        Generate new task allocation by randomly selecting two processors and swapping one task
        between them. Ensures that the new task allocation is valid.

        Params
        ------
        schedule: Schedule, schedule containing task allocation to be permuted

        Returns
        -------
        Schedule, new schedule with permuted task allocation
        """
        valid_schedule = False
        processors = list(schedule.task_allocation.keys())
        while not valid_schedule:
            new_task_allocations = copy.deepcopy(schedule.task_allocation)

            # Randomly pick two tasks to permute
            i_p1 = processors[random.randrange(0, len(processors))]
            i_task1 = random.randrange(0, len(new_task_allocations[i_p1]))
            task1 = new_task_allocations[i_p1][i_task1]

            i_p2 = processors[random.randrange(0, len(processors))]
            i_task2 = random.randrange(0, len(new_task_allocations[i_p2]))

            new_task_allocations[i_p1][i_task1] = new_task_allocations[i_p2][i_task2]
            new_task_allocations[i_p2][i_task2] = task1

            valid_schedule = Schedule.valid_allocations(schedule, new_task_allocations)
        return Schedule(schedule, schedule.context_graph, new_task_allocations)

    def schedule(
        self,
        task_graph: Graph | TaskGraph,
        context_graph: ContextGraph,
    ) -> Schedule:
        """
        Create schedule using simulated annealing to minimise execution cost, determined by simulating the
        execution of the schedule. Uses depth first schedule as initial starting point. Moves currently
        consist only of randomly picking two processors and one task in each and swapping them. Uses the
        normalised exponential form for the acceptance function.

        Params
        ------
        task_graph: Graph or TaskGraph, if Graph then will perform execution of graph
        using thread pool to determine resources in the transformation to a TaskGraph
        context_graph: ContextGraph, containers nodes to which tasks should be assigned

        Returns
        -------
        Schedule, output from annealing algorithm
        """
        # Determine initial conditions
        scheduler = DepthFirstScheduler()
        schedule = scheduler.schedule(task_graph, context_graph)
        initial_cost = self.cost_function(schedule)
        if initial_cost == 0:
            return schedule

        previous_cost = initial_cost

        temp = self.init_temp
        for i_temp in range(self.num_temp_levels):
            num_success = 0
            for _ in range(self.num_tries):
                new_schedule = AnnealingScheduler.permute(schedule)

                # Check to see if new schedule is an acceptable improvement
                new_cost = self.cost_function(new_schedule)
                rand = random.random()
                delta_cost = new_cost - previous_cost
                prob = np.exp(-delta_cost / (initial_cost * temp))
                if delta_cost < 0 or rand < prob:
                    previous_cost = new_cost
                    schedule = new_schedule
                    num_success += 1

                if num_success > self.num_success_cutoff:
                    break

            print(
                f"Temperature iteration {i_temp}: temp {temp}, new cost {previous_cost}"
            )
            temp = temp * self.temp_factor
            if num_success == 0:
                break
        print(f"Initial Cost: {initial_cost}, Annealed Cost: {new_cost}")
        return schedule
