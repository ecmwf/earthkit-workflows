import copy
import random
import numpy as np

from cascade.executors.simulate import Simulator
from cascade.graph import Graph
from cascade.contextgraph import ContextGraph
from .base import Scheduler
from .schedule import Schedule
from .depthfirst import DepthFirstScheduler


class AnnealingScheduler(Scheduler):
    def schedule(
        self,
        task_graph: Graph,
        context_graph: ContextGraph,
        *,
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
        scheduler = DepthFirstScheduler()
        schedule = scheduler.schedule(task_graph, context_graph)
        executor = Simulator()
        report = executor.execute(schedule, with_communication=True)
        print("Initial Report\n", report)
        initial_cost = report.cost()
        previous_cost = report.cost()

        temp = init_temp
        num_processes = len(context_graph)
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
                        task_graph, new_task_allocations
                    )

                all_allocated_tasks = set()
                for allocations in new_task_allocations.values():
                    all_allocated_tasks.update(allocations)
                assert len(list(task_graph.nodes())) == len(all_allocated_tasks)

                # Check to see if new schedule is improvement
                new_schedule = Schedule(task_graph, context_graph, new_task_allocations)
                new_report = executor.execute(new_schedule, with_communication=True)
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
