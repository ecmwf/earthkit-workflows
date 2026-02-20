# ) (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Scheduling-related checkpoint utilities, namely:
    - trim_with_persisted -- given a graph and a set of nodes which are persisted,
      discard everything that doesnt need to be computed
    - virtual_update_schedule -- as we start executing, we virtually publish all
      checkpointed datasets. That on its own would break some invariants in
      the schedule, so we do all the corrections in this function
"""

import logging
from collections import defaultdict

from cascade.low.core import DatasetId, JobInstance, JobInstanceRich, TaskId
from cascade.low.execution_context import JobExecutionContext
from cascade.scheduler.core import ComponentCore, Preschedule, Schedule

logger = logging.getLogger(__name__)

def _filter_invalid(job: JobInstance, persisted: set[DatasetId]) -> tuple[set[TaskId], set[DatasetId]]:
    """Filters out those persisted datasets that overlap partially with a gang or provide only partial outputs
    of a task"""
    persisted_per_task = defaultdict(set)
    for ds in persisted:
        persisted_per_task[ds.task].add(ds)
    persisted_tasks = {
        task
        for task, dss in persisted_per_task.items()
        if job.outputs_of(task) == dss
    }
    rem = set(persisted_per_task.keys()) - persisted_tasks
    if rem:
        logger.warning(f"ignoring persisted outputs of tasks {rem} because of output partiality")
    for gang in job.constraints:
        gang_set = set(gang)
        rem = gang_set - persisted_tasks
        if rem != gang_set:
            persisted_tasks = persisted_tasks - rem
            logger.warning(f"ignoring persisted outputs of tasks {rem} because of gang partiality")
    persisted_datasets = {
        dataset
        for task in persisted_tasks
        for dataset in persisted_per_task[task]
    }
    return persisted_tasks, persisted_datasets

def _get_parents(component: ComponentCore, preschedule: Preschedule) -> dict[TaskId, set[TaskId]]:
    """Just convert edge_i into a task-task dict"""
    parents: dict[TaskId, set[TaskId]] = defaultdict(set)
    for node in component.nodes:
        for dataset in preschedule.edge_i[node]:
            parents[node].add(dataset.task)
    return parents

def _discover_component(component: ComponentCore, parents: dict[TaskId, set[TaskId]], job: JobInstance, preschedule: Preschedule, persisted_tasks: set[TaskId]) -> set[TaskId]:
    """Returns only the valid nodes, ie, those that are either persisted or should be computed"""

    visitable = defaultdict(set)
    for node in component.nodes:
        for d in job.outputs_of(node):
            visitable[d.task].update(preschedule.edge_o[d])
    sinks = [v for v, e in visitable.items() if not e]
    is_valid = {v for v in sinks}
    queue = [v for v in sinks if v not in persisted_tasks]
    while queue:
        head = queue.pop(0)
        are_parents_validable = (head not in persisted_tasks) and (head in is_valid)
        for parent in parents[head]:
            visitable[parent].remove(head)
            if are_parents_validable:
                is_valid.add(parent)
            if not visitable[parent]:
                queue.append(parent)
    return is_valid

def _trim_component(component: ComponentCore, parents: dict[TaskId, set[TaskId]], preserved_tasks: set[TaskId]) -> ComponentCore:
    """Restricts the component only to preserved tasks"""
    nodes = list(preserved_tasks)
    value={node: component.value[node] for node in nodes}
    return ComponentCore(
        nodes=nodes,
        sources=[node for node in nodes if not parents[node]],
        distance_matrix={
            a: {b: component.distance_matrix[a][b] for b in nodes} 
            for a in nodes
        },
        value=value,
        depth=max(value.values()),
        fusing_opportunities={
            # We simply drop the persisted tasks from the lists -- presuming still valid fusing.
            # However, the tasks which are persisted make no real sense as having anything fused in them,
            # as they will be computed instantly at the job start.
            node: [e for e in component.fusing_opportunities[node] if e in preserved_tasks]
            for node in component.fusing_opportunities.keys()
        },
        gpu_fused_distance={node: component.gpu_fused_distance[node] for node in nodes},
    )

def trim_with_persisted(jobRich: JobInstanceRich, preschedule: Preschedule, persisted: set[DatasetId]) -> tuple[JobInstance, Preschedule, set[DatasetId]]:
    """Removes from job and preschedule everything that does not need to be computed anymore,
    ie, lies in the subgraph of nodes for which every path to a sink goes through a persisted
    dataset

    Some persisted sets may be ignored, such as if they are only a part of a gang, or only
    a part of a task's output -- we thus return the modified persisted set"""
    job = jobRich.jobInstance
    persisted_tasks, persisted_datasets = _filter_invalid(job, persisted)

    components_new = []
    preserved_tasks_all = set()
    for component in preschedule.components:
        parents = _get_parents(component, preschedule)
        preserved_tasks = _discover_component(component, parents, job, preschedule, persisted_tasks)
        component_trimmed = _trim_component(component, parents, preserved_tasks)
        components_new.append(component_trimmed)
        preserved_tasks_all = preserved_tasks_all.union(preserved_tasks)
    components_new.sort(key=lambda c: c.weight(), reverse=True)
    preschedule_mod = Preschedule(
        components=components_new,
        edge_o={
            ds: {task for task in tasks if task in preserved_tasks_all}
            for ds, tasks in preschedule.edge_o.items()
            if ds.task in preserved_tasks_all
        },
        edge_i={
            task: {dataset for dataset in datasets if dataset.task in preserved_tasks_all}
            for task, datasets in preschedule.edge_i.items()
            if task in preserved_tasks_all
        },
    )

    job_mod = JobInstance(
        tasks={taskId: taskInstance for taskId, taskInstance in job.tasks.items() if taskId in preserved_tasks_all},
        edges=[edge for edge in job.edges if edge.source.task in preserved_tasks_all and edge.sink_task in preserved_tasks_all],
        serdes=job.serdes,
        ext_outputs=[dataset for dataset in job.ext_outputs if dataset.task in preserved_tasks_all],
        constraints=[constraint for constraint in job.constraints if set(constraint) <= preserved_tasks_all],
    )
    return job_mod, preschedule_mod, persisted_datasets

def virtual_update_schedule(datasets: set[DatasetId], schedule: Schedule, context: JobExecutionContext) -> None:
    """Include all tasks that are children of these datasets in the worker2task_values
    of the component, and remove the tasks corresponding to these datasets from there and from computable."""

    # TODO order by component and make this more efficient

    finished_tasks = set(ds.task for ds in datasets)
    for task in finished_tasks:
        # TODO unify this with scheduler/assignment/_postproc as a schedule's 'mark_assigned' method
        component = schedule.components[schedule.ts2component[task]]
        component.worker2task_values.remove(task)
        component.computable.pop(task)
        schedule.computable -= 1

    for ds in datasets:
        component = schedule.components[schedule.ts2component[ds.task]]
        for task in context.edge_o[ds]:
            component.worker2task_values.add(task)
