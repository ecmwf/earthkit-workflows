# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict
from typing import Iterator

from cascade.low.core import TaskId, WorkerId
from cascade.low.execution_context import JobExecutionContext, TaskStatus
from cascade.low.tracing import Microtrace, timer
from cascade.scheduler.assign import (
    assign_within_component,
    migrate_to_component,
    update_worker2task_distance,
)
from cascade.scheduler.core import (
    Assignment,
    ComponentId,
    ComponentSchedule,
    GangPreparation,
    Preschedule,
    Schedule,
)

logger = logging.getLogger(__name__)


def gang_check_ready(task: TaskId, gang_prep: GangPreparation):
    """When a task becomes computable, mutate the gang_prep to possibly
    transition some gangs to `ready`
    """
    for gang in gang_prep.lookup[task]:
        if gang not in gang_prep.countdown:
            raise ValueError(
                f"after {task=} marked computable, {gang=} not found -- double compuptable mark?"
            )
        remaining = gang_prep.countdown[gang]
        if task not in remaining:
            raise ValueError(
                f"after {task=} marked computable, {gang=} does not have it in {remaining=}. Invalid gang?"
            )
        remaining.remove(task)
        if not remaining:
            logger.debug(f"gang just became ready {gang=}")
            gang_prep.ready.append(gang)
            gang_prep.countdown.pop(gang)


def init_schedule(preschedule: Preschedule, context: JobExecutionContext) -> Schedule:
    components: list[ComponentSchedule] = []
    ts2component: dict[TaskId, ComponentId] = {}

    gangs = [
        frozenset(constraint.gang) for constraint in context.job_instance.constraints
    ]

    computable = 0
    for componentId, precomponent in enumerate(preschedule.components):
        # gang preparation
        tasks = set(precomponent.nodes)
        lookup = defaultdict(list)
        countdown = {}
        i = 0
        while i < len(gangs):
            if not gangs[i].issubset(tasks):
                i += 1
                continue
            gang = gangs.pop(i)
            countdown[gang] = set(gang)
            for e in gang:
                lookup[e].append(gang)

        gang_preparation = GangPreparation(
            ready=[],
            lookup=lookup,
            countdown=countdown,
        )
        for source in precomponent.sources:
            gang_check_ready(source, gang_preparation)

        # component itself
        component = ComponentSchedule(
            core=precomponent,
            weight=precomponent.weight(),
            computable={task: 0 for task in precomponent.sources},
            worker2task_distance={},
            worker2task_values=set(precomponent.sources),
            is_computable_tracker={
                task: {inp for inp in context.edge_i[task]}
                for task in precomponent.nodes
            },
            gang_preparation=gang_preparation,
        )
        components.append(component)
        computable += len(precomponent.sources)
        for task in precomponent.nodes:
            ts2component[task] = componentId

    if gangs:
        for gang in gangs:
            logger.error(f"a gang not part of a component: {gang}")
        raise ValueError(f"a total of {len(gangs)} were not a subcomponent")

    return Schedule(
        components=components,
        ts2component=ts2component,
        host2component={host: None for host in context.host2workers.keys()},
        computable=computable,
        worker2task_overhead=defaultdict(dict),
    )


def assign(schedule: Schedule, context: JobExecutionContext) -> Iterator[Assignment]:
    """Given idle workers in `state`, assign actions to workers. Mutates the state:
     - pops from computable & idle workers,
     - decreases weight,
     - changes host2component.
    Yields, to allow for immediate async sending to workers.
    Performance critical section, we need to output an assignment asap. Steps taking longer
    should be deferred to `plan`
    """

    # step I: assign within existing components
    component2workers: dict[ComponentId, list[WorkerId]] = defaultdict(list)
    for worker in context.idle_workers:
        if (component := schedule.host2component[worker.host]) is not None:
            component2workers[component].append(worker)

    for component_id, local_workers in component2workers.items():
        if local_workers:
            yield from assign_within_component(
                schedule, local_workers, component_id, context
            )

    if not context.idle_workers:
        return

    # step II: assign remaining workers to new components
    components = [
        (component.weight, component_id)
        for component_id, component in enumerate(schedule.components)
        if component.weight > 0
    ]
    if not components:
        return

    components.sort(
        reverse=True
    )  # TODO consider number of currently assigned workers too
    migrants = defaultdict(list)
    for worker in context.idle_workers:
        # TODO we dont currently allow partial assignments, this is subopt!
        if (component := schedule.host2component[worker.host]) is None or (
            schedule.components[component].weight == 0
        ):
            migrants[worker.host].append(worker)
        # TODO we ultimately want to be able to have weight-and-capacity-aware m-n host2component
        # assignments, not just round robin of the whole host2component

    component_i = 0
    for host, workers in migrants.items():
        component_id = components[component_i][1]
        timer(migrate_to_component, Microtrace.ctrl_migrate)(
            host, component_id, schedule, context
        )
        yield from assign_within_component(schedule, workers, component_id, context)
        component_i = (component_i + 1) % len(components)


def plan(
    schedule: Schedule, context: JobExecutionContext, assignments: list[Assignment]
):
    """Given actions that were just sent to a worker, update state to reflect it, including preparation
    and planning for future assignments.
    Unlike `assign`, this is less performance critical, so slightly longer calculations can happen here.
    """

    # TODO when considering `children` below, filter for near-computable? Ie, either already in computable
    # or all inputs are already in preparing state? May not be worth it tho

    for assignment in assignments:
        for prep in assignment.prep:
            context.dataset_preparing(prep[0], assignment.worker)
            children = context.edge_o[prep[0]]
            update_worker2task_distance(children, assignment.worker, schedule, context)
        for task in assignment.tasks:
            for ds in assignment.outputs:
                children = context.edge_o[ds]
                # context.dataset_preparing(ds, assignment.worker) # happends during build already
                update_worker2task_distance(
                    children, assignment.worker, schedule, context
                )
            context.worker2ts[assignment.worker][task] = TaskStatus.enqueued
            context.ts2worker[task][assignment.worker] = TaskStatus.enqueued
            if task in context.ongoing[assignment.worker]:
                raise ValueError(f"double add of {task} to {assignment.worker}")
            context.ongoing[assignment.worker].add(task)
            context.ongoing_total += 1
