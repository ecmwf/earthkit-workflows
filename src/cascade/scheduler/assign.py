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
from cascade.scheduler.core import (
    Assignment,
    ComponentCore,
    ComponentId,
    ComponentSchedule,
    Schedule,
)

logger = logging.getLogger(__name__)


def build_assignment(
    worker: WorkerId, task: TaskId, context: JobExecutionContext, core: ComponentCore
) -> Assignment:
    eligible_load = {DatasetStatus.preparing, DatasetStatus.available}
    eligible_transmit = {DatasetStatus.available}
    prep: list[tuple[DatasetId, HostId]] = []
    if task in core.fusing_opportunities:
        tasks = core.fusing_opportunities.pop(task)
    else:
        tasks = [task]
    assigned = []
    exhausted = False
    at_worker = context.worker2ds[worker]
    at_host = context.host2ds[worker.host]
    worker_has_gpu = context.environment.workers[worker].gpu > 0
    while tasks and not exhausted:
        task = tasks[0]
        if context.job_instance.tasks[task].definition.needs_gpu and not worker_has_gpu:
            if not assigned:
                raise ValueError(f"tried to assign gpu {task=} to non-gpu {worker=}")
            else:
                break
        for dataset in context.edge_i[task]:
            if at_worker.get(dataset, DatasetStatus.missing) not in eligible_load:
                if at_host.get(dataset, DatasetStatus.missing) in eligible_load:
                    prep.append((dataset, worker.host))
                else:
                    if any(
                        candidate := host
                        for host, status in context.ds2host[dataset].items()
                        if status in eligible_transmit
                    ):
                        prep.append((dataset, candidate))
                        context.dataset_preparing(dataset, worker)
                    else:
                        # if we are dealing with the first task to assign, we don't expect to be here!
                        if not assigned:
                            raise ValueError(f"{dataset=} not found anywhere!")
                        # if we are already trying some fusing opportunities, it is legit to not find the dataset anywhere
                        else:
                            # TODO rollback preps done for this one task
                            exhausted = True
                            break
        if not exhausted:
            assigned.append(tasks.pop(0))
            for dataset in context.task_o[task]:
                context.dataset_preparing(dataset, worker)

    if len(tasks) > 1:
        head = tasks[0]
        if head in core.fusing_opportunities:
            raise ValueError(f"double assignment to {head} in fusing opportunities!")
        core.fusing_opportunities[head] = tasks

    # trim for only the necessary ones -- that is, having any edge outside of this current assignment
    all_outputs = {ds for task in assigned for ds in context.task_o[task]}
    assigned_tasks = set(assigned)
    trimmed_outputs = {
        ds
        for ds in all_outputs
        if (context.edge_o[ds] - assigned_tasks)
        or (ds in context.job_instance.ext_outputs)
    }

    return Assignment(
        worker=worker,
        tasks=assigned,
        prep=prep,
        outputs=trimmed_outputs,
        extra_env={},
    )


def _postproc_assignment(
    assignment: Assignment,
    component: ComponentSchedule,
    schedule: Schedule,
    context: JobExecutionContext,
) -> None:
    for assigned in assignment.tasks:
        if assigned in component.computable:
            component.computable.pop(assigned)
            component.worker2task_values.remove(assigned)
            schedule.computable -= 1
        else:
            # shortcut for fused-in tasks
            component.is_computable_tracker[assigned] = set()
    context.idle_workers.remove(assignment.worker)
    component.weight -= len(assignment.tasks)


# TODO this is not particularly systematic! We cant bind dynamically at the host as we send this
# in advance, so we need to hardcode. Ideally we centrallize all port opening into a single module,
# in particular unify this with the portBase from benchmarks/__main__ and then derived ports from
# executor/executor.py etc. As is, we have a single global variable that we increment, to ensure
# no port collision happens gang-wise -- we dont really expect many gangs per a workflow
gang_port = 12355


def _try_assign_gang(
    schedule: Schedule,
    gang: list[frozenset[TaskId]],
    workers: list[WorkerId],
    component_id: ComponentId,
    context: JobExecutionContext,
    fail_acc: list[frozenset[TaskId]],
) -> Iterator[Assignment]:
    """We greedily assign by descending worker-task distance"""
    global gang_port
    if len(gang) > len(workers):
        logger.debug(f"not enough workers ({len(workers)}) for {gang=}")
        fail_acc.append(gang)
        return
    start = perf_counter_ns()
    component = schedule.components[component_id]
    gpu_tasks: set[TaskId] = set()
    cpu_tasks: set[TaskId] = set()
    gpu_workers: set[WorkerId] = set()
    cpu_workers: set[WorkerId] = set()
    for task in gang:
        if context.job_instance.tasks[task].definition.needs_gpu:
            gpu_tasks.add(task)
        else:
            cpu_tasks.add(task)
    for worker in workers:
        if context.environment.workers[worker].gpu > 0:
            gpu_workers.add(worker)
        else:
            cpu_workers.add(worker)
    if len(gpu_tasks) > len(gpu_workers):
        logger.debug(f"not enough gpu workers ({len(workers)}) for {gang=}")
        fail_acc.append(gang)
        end = perf_counter_ns()
        trace(Microtrace.ctrl_assign, end - start)
        return

    world_size = len(gang)
    rank = 0
    coordinator = None

    # similarly to _assignment_heuristic, a greedy algorithm
    candidates = [
        (schedule.worker2task_overhead[w][t], component.core.value[t], w, t)
        for w in gpu_workers
        for t in gpu_tasks
    ]
    candidates.sort(key=lambda e: (e[0], e[1]))
    for _, _, worker, task in candidates:
        if task in gpu_tasks and worker in gpu_workers:
            if task not in component.computable:
                # it may be that some fusing for previous task already assigned this
                continue
            end = perf_counter_ns()
            trace(Microtrace.ctrl_assign, end - start)
            assignment = build_assignment(worker, task, context, component.core)
            if not coordinator:
                coordinator = (
                    f"{context.environment.host_url_base[worker.host]}:{gang_port}"
                )
            assignment.extra_env["CASCADE_GANG_WORLD_SIZE"] = str(world_size)
            assignment.extra_env["CASCADE_GANG_RANK"] = str(rank)
            assignment.extra_env["CASCADE_GANG_COORDINATOR"] = coordinator
            rank += 1
            yield assignment
            start = perf_counter_ns()
            _postproc_assignment(assignment, component, schedule, context)
            gpu_tasks.remove(task)
            gpu_workers.remove(worker)
    if gpu_tasks:
        raise ValueError(
            f"expected to assign all gang gpu tasks, yet {gpu_tasks} remain"
        )

    all_workers = cpu_workers.union(gpu_workers)
    candidates = [
        (schedule.worker2task_overhead[w][t], component.core.value[t], w, t)
        for w in all_workers
        for t in cpu_tasks
    ]
    candidates.sort(key=lambda e: (e[0], e[1]))
    for _, _, worker, task in candidates:
        if task in cpu_tasks and worker in all_workers:
            if task not in component.computable:
                # it may be that some fusing for previous task already assigned this
                continue
            end = perf_counter_ns()
            trace(Microtrace.ctrl_assign, end - start)
            assignment = build_assignment(worker, task, context, component.core)
            if not coordinator:
                coordinator = (
                    f"{context.environment.host_url_base[worker.host]}:{gang_port}"
                )
            assignment.extra_env["CASCADE_GANG_WORLD_SIZE"] = str(world_size)
            assignment.extra_env["CASCADE_GANG_RANK"] = str(rank)
            assignment.extra_env["CASCADE_GANG_COORDINATOR"] = coordinator
            rank += 1
            yield assignment
            start = perf_counter_ns()
            _postproc_assignment(assignment, component, schedule, context)
            cpu_tasks.remove(task)
            all_workers.remove(worker)
    if cpu_tasks:
        raise ValueError(
            f"expected to assign all gang cpu tasks, yet {cpu_tasks} remain"
        )

    end = perf_counter_ns()
    trace(Microtrace.ctrl_assign, end - start)
    gang_port += 1


def _assignment_heuristic(
    schedule: Schedule,
    tasks: list[TaskId],
    workers: list[WorkerId],
    component_id: ComponentId,
    context: JobExecutionContext,
) -> Iterator[Assignment]:
    """Finds a reasonable assignment within a single component. Does not migrate hosts to a different component."""
    start = perf_counter_ns()
    component = schedule.components[component_id]

    # first, attempt optimum-distance assignment
    unassigned: list[TaskId] = []
    for task in tasks:
        if task not in component.computable:
            # it may be that some fusing for previous task already assigned this
            continue
        opt_dist = component.computable[task]
        was_assigned = False
        for idx, worker in enumerate(workers):
            if component.worker2task_distance[worker][task] == opt_dist:
                end = perf_counter_ns()
                trace(Microtrace.ctrl_assign, end - start)
                assignment = build_assignment(worker, task, context, component.core)
                yield assignment
                start = perf_counter_ns()
                _postproc_assignment(assignment, component, schedule, context)
                workers.pop(idx)
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
            if task not in component.computable:
                # it may be that some fusing for previous task already assigned this
                continue
            end = perf_counter_ns()
            trace(Microtrace.ctrl_assign, end - start)
            assignment = build_assignment(worker, task, context, component.core)
            yield assignment
            start = perf_counter_ns()
            _postproc_assignment(assignment, component, schedule, context)
            remaining_t.remove(task)
            remaining_w.remove(worker)

    end = perf_counter_ns()
    trace(Microtrace.ctrl_assign, end - start)


def assign_within_component(
    schedule: Schedule,
    workers: list[WorkerId],
    component_id: ComponentId,
    context: JobExecutionContext,
) -> Iterator[Assignment]:
    """We hardcode order of handling task groups:
        1/ ready gangs,
        2/ tasks requiring a gpu,
        3/ tasks whose fusable child requires a gpu,
        4/ all other tasks,
    using the same algorithm for cases 2-4 and a naive for case 1
    """
    # TODO rework into a more systematic multicriterial opt solution that is able to consider all groups
    # at once, using a generic value/cost framework and matching algorithm. It should additionally be able
    # to issue a "strategic wait" command -- eg if we could assign a task to an idle worker with high cost,
    # or wait until a better-equipped busy worker finished, etc.
    component = schedule.components[component_id]

    # gangs
    fail_acc: list[frozenset[TaskId]] = []
    for gang in component.gang_preparation.ready:
        logger.debug(f"trying to assign a {gang=}")
        yield from _try_assign_gang(
            schedule, gang, list(context.idle_workers), component_id, context, fail_acc
        )
    component.gang_preparation.ready = fail_acc

    # the other cases: build them first
    cpu_t: list[TaskId] = []
    gpu_t: list[TaskId] = []
    opu_t: list[TaskId] = []
    for task in component.computable.keys():
        if component.gang_preparation.lookup[task]:
            # no gang participation in single-task scheduling
            continue
        elif context.job_instance.tasks[task].definition.needs_gpu:
            gpu_t.append(task)
        elif component.core.gpu_fused_distance[task] is not None:
            opu_t.append(task)
        else:
            cpu_t.append(task)

    # tasks immediately needing a gpu
    eligible_w = [
        worker
        for worker in workers
        if context.environment.workers[worker].gpu > 0
        and worker in context.idle_workers
    ]
    logger.debug(
        f"considering {len(gpu_t)}# gpu tasks, {len(opu_t)}# maybe-gpu tasks, {len(cpu_t)}# cpu tasks, with {len(workers)}# workers out of which {len(eligible_w)} have gpu"
    )
    yield from _assignment_heuristic(schedule, gpu_t, eligible_w, component_id, context)
    # tasks whose fusing opportunity needs a gpu
    eligible_w = [worker for worker in eligible_w if worker in context.idle_workers]
    yield from _assignment_heuristic(schedule, opu_t, eligible_w, component_id, context)
    # remaining tasks
    eligible_w = [worker for worker in workers if worker in context.idle_workers]
    u_opu_t = [task for task in opu_t if task in component.computable]
    yield from _assignment_heuristic(
        schedule, cpu_t + u_opu_t, eligible_w, component_id, context
    )


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
