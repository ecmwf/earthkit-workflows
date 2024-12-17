"""
Implementations of various assignment strategies (ie, part of planning), utilizing
smaller functions from `controller.dynamic` and `controller.views`
"""

# NOTE there are a few suboptimalities here, we traverse the graph
# all over at every planning stage, instead of retaining some information
# computed. To be improved if planning becomes bottleneck

from cascade.low.core import Environment, WorkerId, TaskId, DatasetId, JobInstance, JobExecutionRecord
from cascade.scheduler.core import Schedule
from cascade.controller.core import State, DatasetStatus
from cascade.controller.dynamic import get_task_values, taskAvailability, taskFitness
from cascade.controller.views import get_host2datasets
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def naive_assignment(environment: Environment, availableWorkers: set[WorkerId], state: State, taskInputs: dict[TaskId, set[DatasetId]], job: JobInstance, schedule: Schedule) -> dict[WorkerId, list[TaskId]]:
    """The algorithm: just round robin naive assignment"""
    available_datasets = set(d for d, w2s in state.ds2worker.items() if DatasetStatus.available in set(w2s.values()))
    tasks = schedule.computable
    logger.debug(f"trying to assign given {available_datasets=} and {tasks=}")
    return {
        worker_id: [task_id]
        for worker_id, task_id in zip(availableWorkers, tasks)
    }

def fitness_assignment(environment: Environment, availableWorkers: set[WorkerId], state: State, taskInputs: dict[TaskId, set[DatasetId]], job: JobInstance, schedule: Schedule) -> dict[WorkerId, list[TaskId]]:
    """The algorithm:
    - consider first tasks with minimum availability (globally) and maximum fitness (among hosts): if any, find maximum valued and repeat
    - if none, then choose a task with maximum availability (globally) which has a maximum value
      - NOTE: this choice means we prefer future transmits over present transmits. The other option would be to choose here by fitness solely
    """
    assignment: dict[WorkerId, list[TaskId]] = defaultdict(list)
    availableGlobally = set(d for d, w2s in state.ds2worker.items() if DatasetStatus.available in set(w2s.values()))
    # the schedule.computable represents "assuming everything ongoing has been computed", so may not necessarily be
    # computable immediately. We thus filter for computable-given-current-state. TODO this should be avoidable by
    # merging State and Schedule
    computable = {e for e in schedule.computable if taskInputs.get(e, set()) <= availableGlobally}
    remaining = {task for layer in schedule.layers for task in layer}.union(computable)
    host2datasets = get_host2datasets(state, environment)
    hosts = [colocation[0].split(":", 1)[0] for colocation in environment.colocations]
        
    while availableWorkers and computable:
        logger.debug(f"assignment round with {availableWorkers=} and {len(computable)=}:")
        hostsWithWorker = {w.split(":", 1)[0] for w in availableWorkers}
        taskValues = get_task_values(schedule)
        task2hostAvailability = {
            task: {
                host: taskAvailability(task, taskInputs, host2datasets[host], schedule.record).datasets
                for host in hostsWithWorker
            }
            for task in computable
        }
        bestAvailability = min((task2hostAvailability[task][host] for task in computable for host in hostsWithWorker))
        task2hostFitness = {
            task: {
                host: taskFitness(task, job, taskInputs, host2datasets[host], remaining).improvedComputability
                for host in hosts
            }
            for task in computable
        }
        task2maxFit = {task: max(task2hostFitness[task].values()) for task in computable}

        choice: tuple[WorkerId, TaskId]|None = None
        choice_value = -1
        for host in hostsWithWorker:
            for task in computable:
                isMaxFit = task2maxFit[task] == task2hostFitness[task][host] 
                isBestAvail = task2hostAvailability[task][host] == bestAvailability
                isMaxValue = taskValues[task] > choice_value
                if isMaxFit and isBestAvail and isMaxValue:
                    any((workerId := e) for e in availableWorkers if e.split(":", 1)[0] == host)
                    choice = (workerId, task)
                    choice_value = taskValues[task]
                    break
        if not choice:
            for host in hostsWithWorker:
                for task in computable:
                    isBestAvail = task2hostAvailability[task][host] == bestAvailability
                    isMaxValue = taskValues[task] > choice_value
                    if isBestAvail and isMaxValue:
                        any((workerId := e) for e in availableWorkers if e.split(":", 1)[0] == host)
                        choice = (workerId, task)
                        choice_value = taskValues[task]
                        break
        if not choice:
            # best availability is computed for hosts with worker only, so we must always find a candidate
            raise ValueError("unexpected state")

        assignment[choice[0]].append(choice[1])
        availableWorkers = availableWorkers - {choice[0]}
        computable = computable - {choice[1]}
        remaining = remaining - {choice[1]}
        host2datasets[choice[0].split(":", 1)[0]].update(job.outputs_of(choice[1]))
        logger.debug(f"assignment {choice=}")

    return assignment
