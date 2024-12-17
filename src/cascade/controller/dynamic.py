"""
Utilities for dynamic scheduling and assignments during planning
"""

import logging
from dataclasses import dataclass
from cascade.low.core import DatasetId, TaskId, JobExecutionRecord, JobInstance
from cascade.scheduler.core import Schedule
from cascade.controller.core import State

logger = logging.getLogger(__name__)

@dataclass
class DataTransmitCost:
    datasets: int
    # total_mb_volume: int

def taskAvailability(task: TaskId, taskInputs: dict[TaskId, set[DatasetId]], datasetsAvailable: set[DatasetId], record: JobExecutionRecord) -> DataTransmitCost:
    """How much data must be transfered for this task to be run at given worker"""
    # TODO consider some extension of JobExecutionRecord -- we *know* at runtime how large exactly the computed datasets are
    # Afterwards, make the total_mb_volume available, and use it in decision making
    datasets = 0
    total_mb_volume = 0

    for ds in taskInputs.get(task, set()):
        if ds not in datasetsAvailable:
            datasets += 1
            total_mb_volume += record.datasets_mb[ds]

    return DataTransmitCost(datasets=datasets) # , total_mb_volume=total_mb_volume)

@dataclass
class TaskFitness:
    # just number of tasks for whom the number of available inputs at the worker increases from non-zero
    # ext: possibly consider LCA instead of direct computability
    # ext: possibly consider weighing by the tasks made available (their relative importance, memory ratio, how close to fully computable, computability elsewhere...)
    improvedComputability: int 

def taskFitness(task: TaskId, job: JobInstance, taskInputs: dict[TaskId, set[DatasetId]], workerAvailable: set[DatasetId], remainingTasks: set[TaskId]) -> TaskFitness:
    """How useful it would be, in terms of computable datasets, to have this task's outputs at the worker"""

    partiallyComputable = (t for t in remainingTasks - {task} if taskInputs.get(t, set()).intersection(workerAvailable))
    improvedComputability = sum((1 for t in partiallyComputable if taskInputs.get(t, set()).intersection(job.outputs_of(task))))
    return TaskFitness(improvedComputability = improvedComputability)

def assume_computed(schedule: Schedule, available: set[DatasetId], toBeComputed: list[TaskId], taskInputs: dict[TaskId, set[DatasetId]], job: JobInstance) -> None:
    """Preparation for the next planning round. Mutates the schedule as if the toBeComputed became available."""
    # drain computed
    s_toBeComputed = set(toBeComputed)
    schedule.computable = [e for e in schedule.computable if e not in s_toBeComputed]
    # extend available
    for task in toBeComputed:
        available = available.union(job.outputs_of(task))
    # drain schedule and extend computable
    i_layer = 0
    while i_layer < len(schedule.layers):
        i_task = 0
        while i_task < len(schedule.layers[i_layer]):
            task = schedule.layers[i_layer][i_task]
            if taskInputs[task] <= available:
                schedule.layers[i_layer].pop(i_task)
                # toBeComputed may have contained elements not in `computable` bcs of fusing
                if task not in s_toBeComputed:
                    schedule.computable.append(task)
            else:
                i_task += 1
        if not schedule.layers[i_layer]:
            schedule.layers.pop(i_layer)
        else:
            i_layer += 1
    logger.debug(f"computable is now {schedule.computable}")

def get_task_values(schedule: Schedule) -> dict[TaskId, int]:
    """This computes how valuable it is to compute the task at this moment.
    Currently it is simply by considering depth in the schedule.
    Ideally, would be 'actual and potenial memory pressure relief' or 'sink distance', etc.""" 
    values: dict[TaskId, int] = {task: 0 for task in schedule.computable}
    for l, layer in enumerate(schedule.layers):
        for task in layer:
            values[task] = l + 1
    return values
