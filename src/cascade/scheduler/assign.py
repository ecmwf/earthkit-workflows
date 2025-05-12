# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utility functions for handling assignments -- invocation assumed from scheduler.api module,
for all other purposes this should be treated private
"""

import logging
from collections import defaultdict
from time import perf_counter_ns
from typing import Iterable, Iterator

from cascade.low.core import DatasetId, HostId, TaskId, WorkerId
from cascade.low.execution_context import DatasetStatus, JobExecutionContext
from cascade.low.tracing import Microtrace, trace
from cascade.scheduler.core import Assignment, ComponentId, Schedule

logger = logging.getLogger(__name__)


def build_assignment(
    worker: WorkerId, task: TaskId, context: JobExecutionContext
) -> Assignment:
    eligible_load = {DatasetStatus.preparing, DatasetStatus.available}
    eligible_transmit = {DatasetStatus.available}
    prep: list[tuple[DatasetId, HostId]] = []
    for dataset in context.edge_i[task]:
        at_worker = context.worker2ds[worker]
        if at_worker.get(dataset, DatasetStatus.missing) not in eligible_load:
            if (
                context.host2ds[worker.host].get(dataset, DatasetStatus.missing)
                in eligible_load
            ):
                # NOTE this currently leads to no-op, but with persistent workers would possibly allow an early fetch
                prep.append((dataset, worker.host))
            else:
                if any(
                    candidate := host
                    for host, status in context.ds2host[dataset].items()
                    if status in eligible_transmit
                ):
                    prep.append((dataset, candidate))
                    # NOTE this is a slight hack, to prevent issuing further transmit commands of this ds to this host
                    # in this phase. A proper state transition happens later in the `plan` phase. We may want to instead
                    # create a new `transmit_queue` state field to capture this, and consume it later during plan
                    context.host2ds[worker.host][dataset] = DatasetStatus.preparing
                    context.ds2host[dataset][worker.host] = DatasetStatus.preparing
                else:
                    raise ValueError(f"{dataset=} not found in any host, whoa whoa!")

    return Assignment(
        worker=worker,
        tasks=[
            task
        ],  # TODO eager fusing for outdeg=1? Or heuristic via ratio of outdeg vs workers@component?
        prep=prep,
        outputs={  # TODO trim for only the necessary ones
            ds for ds in context.task_o[task]
        },
    )


def _assignment_heuristic(
    schedule: Schedule,
    tasks: list[TaskId],
    workers: list[WorkerId],
    component_id: ComponentId,
    context: JobExecutionContext,
) -> Iterator[Assignment]:
    """Finds a reasonable assignment within a single component. Does not migrate hosts to a different component"""
    start = perf_counter_ns()
    component = schedule.components[component_id]

    # first, attempt optimum-distance assignment
    unassigned: list[TaskId] = []
    for task in tasks:
        opt_dist = component.computable[task]
        was_assigned = False
        for idx, worker in enumerate(workers):
            if component.worker2task_distance[worker][task] == opt_dist:
                end = perf_counter_ns()
                trace(Microtrace.ctrl_assign, end - start)
                yield build_assignment(worker, task, context)
                start = perf_counter_ns()
                workers.pop(idx)
                component.computable.pop(task)
                component.worker2task_values.remove(task)
                component.weight -= 1
                schedule.computable -= 1
                context.idle_workers.remove(worker)
                was_assigned = True
                break
        if not was_assigned:
            unassigned.append(task)

    # second, sort task-worker combination by first overhead, second value, and pick greedily
    remaining_t = set(unassigned)
    remaining_w = set(workers)
    candidates = [
        (schedule.worker2task_overhead[w][t], component.core.value[t], w, t)
        for w in workers
        for t in remaining_t
    ]
    candidates.sort(key=lambda e: (e[0], e[1]))
    for _, _, worker, task in candidates:
        if task in remaining_t and worker in remaining_w:
            end = perf_counter_ns()
            trace(Microtrace.ctrl_assign, end - start)
            yield build_assignment(worker, task, context)
            start = perf_counter_ns()
            component.computable.pop(task)
            component.worker2task_values.remove(task)
            remaining_t.remove(task)
            remaining_w.remove(worker)
            context.idle_workers.remove(worker)
            schedule.computable -= 1
            component.weight -= 1

    end = perf_counter_ns()
    trace(Microtrace.ctrl_assign, end - start)


def assign_within_component(
    schedule: Schedule,
    workers: list[WorkerId],
    component_id: ComponentId,
    context: JobExecutionContext,
) -> Iterator[Assignment]:
    """We first handle gpu things, second cpu things, using the same algorithm for either case"""
    # TODO employ a more systematic solution and handle all multicriterially at once -- ideally together with adding support for multi-gpu-groups
    cpu_t: list[TaskId] = []
    gpu_t: list[TaskId] = []
    gpu_w: list[WorkerId] = []
    cpu_w: list[WorkerId] = []
    for task in schedule.components[component_id].computable.keys():
        if context.job_instance.tasks[task].definition.needs_gpu:
            gpu_t.append(task)
        else:
            cpu_t.append(task)
    for worker in workers:
        if context.environment.workers[worker].gpu > 0:
            gpu_w.append(worker)
        else:
            cpu_w.append(worker)
    yield from _assignment_heuristic(schedule, gpu_t, gpu_w, component_id, context)
    for worker in gpu_w:
        if worker in context.idle_workers:
            cpu_w.append(worker)
    yield from _assignment_heuristic(schedule, cpu_t, cpu_w, component_id, context)


def update_worker2task_distance(
    tasks: Iterable[TaskId],
    worker: WorkerId,
    schedule: Schedule,
    context: JobExecutionContext,
):
    """For a given task and worker, consider all tasks at the worker and see if any attains a better distance to said
    task. If additionally the task is _already_ computable and the global minimum attained by `component.computable`
    is improved, set that too.
    """
    # TODO we don't currently consider other workers at the host, probably subopt! Ultimately,
    # we need the `assign_within_component` to take both overhead *and* distance into account
    # simultaneously
    eligible = {DatasetStatus.preparing, DatasetStatus.available}
    for task in tasks:
        component_id = schedule.ts2component[task]
        worker2task = schedule.components[component_id].worker2task_distance
        task2task = schedule.components[component_id].core.distance_matrix
        schedule.components[component_id].worker2task_values.add(task)
        computable = schedule.components[component_id].computable
        for ds_key, ds_status in context.worker2ds[worker].items():
            if ds_status not in eligible:
                continue
            if schedule.ts2component[ds_key.task] != component_id:
                continue
            # TODO we only consider min task distance, whereas weighing by volume/ratio would make more sense
            val = min(
                worker2task[worker][task],
                task2task[ds_key.task][task],
            )
            worker2task[worker][task] = val
            if ((current := computable.get(task, None)) is not None) and current > val:
                computable[task] = val


def set_worker2task_overhead(
    schedule: Schedule, context: JobExecutionContext, worker: WorkerId, task: TaskId
):
    # NOTE beware this is used in migrate host2component as well as twice in notify. We may
    # want to later distinguish between `calc_new` (for migrate and new computable) vs
    # `calc_update` (basicaly when host2host transmit finishes)
    # TODO replace the numerical heuristic here with some numbers based on transfer speeds
    # and dataset volumes
    overhead = 0
    for ds in context.edge_i[task]:
        workerState = context.worker2ds[worker].get(ds, DatasetStatus.missing)
        if workerState == DatasetStatus.available:
            continue
        if workerState == DatasetStatus.preparing:
            overhead += 1
            continue
        hostState = context.host2ds[worker.host].get(ds, DatasetStatus.missing)
        if hostState == DatasetStatus.available or hostState == DatasetStatus.preparing:
            overhead += 10
            continue
        overhead += 100
    schedule.worker2task_overhead[worker][task] = overhead


def migrate_to_component(
    host: HostId,
    component_id: ComponentId,
    schedule: Schedule,
    context: JobExecutionContext,
):
    """Assuming original component assigned to the host didn't have enough tasks anymore,
    we invoke this function and update state to reflect it
    """
    schedule.host2component[host] = component_id
    component = schedule.components[component_id]
    logger.debug(
        f"migrate {host=} to {component_id=} => {component.worker2task_values=}"
    )
    for worker in context.host2workers[host]:
        component.worker2task_distance[worker] = defaultdict(
            lambda: component.core.depth
        )
        update_worker2task_distance(
            component.worker2task_values, worker, schedule, context
        )
        for task in component.worker2task_values:
            set_worker2task_overhead(schedule, context, worker, task)
