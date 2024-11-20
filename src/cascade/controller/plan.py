"""
Given a Schedule and current State, issue which Actions should happen next

This is effectively an adapter between the business logic of `scheduler.dynamic`
and the data structures of the `controller` -- it converts the State et cetera
and provides inputs for the `scheduler.dynamic` invocations, and then converts
the results back into `Action`s, taking care of identifying the transmits, finding
purges, and other minor details.
"""

# TODO many suboptimalities here, cf individual TODOs below

import logging

from cascade.scheduler.core import Schedule
from cascade.controller.dynamic import assume_computed
from cascade.controller.assignments import naive_assignment, fitness_assignment
from cascade.low.core import JobInstance, DatasetId, WorkerId, TaskId, Environment
from cascade.low.func import maybe_head
from cascade.controller.core import State, TaskStatus, Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, DatasetStatus
from cascade.controller.views import project_colocation

logger = logging.getLogger(__name__)

def get_available_workers(state: State, env: Environment) -> set[WorkerId]:
    """Returns workers that are available for more work"""
    # NOTE currently: have less running+enqueued than cpu -- not ideal
    return {
        worker_id
        for worker_id, worker in env.workers.items()
        if sum(
            1 for _, status in state.worker2ts[worker_id].items()
            if status in {TaskStatus.enqueued, TaskStatus.running}
        ) < worker.cpu
    }

def convert(assignment: dict[WorkerId, list[TaskId]], state: State, job: JobInstance, taskInputs: dict[TaskId, set[DatasetId]]) -> list[Action]:
    """Converts the assignment into actions -- which may be data transfers or subgraphs"""
    # TODO multiple optimizations required here:
    # - re transmits:
    #   - transmit from workers that are close
    #   - respect colocations: dont transmit to two workers in the same coloc at once
    #   - fuse transmits?
    #   - consider whether the transmit should purge from the source afterwards (if there are no other tasks requiring the input)
    # - fuse tasks (then trim outputs)
    #   - brainless fusing: if the current task has exactly one (remaining) dependant then fuse it (and continue checking to fuse more)
    actions: list[Action] = []
    for worker_id, tasks in assignment.items():
        for task_id in tasks:
            available_ds = set(state.worker2ds[worker_id])
            missing = taskInputs.get(task_id, set()) - available_ds
            outputs = job.outputs_of(task_id)
            for e in missing:
                fr = maybe_head(state.ds2worker[e].keys())
                if not fr:
                    raise ValueError(f"cannot find host with dataset {e}")
                actions.append(
                    ActionDatasetTransmit(
                        ds=[e],
                        fr=[fr],
                        to=[worker_id],
                    )
                )
            available_ds.update(missing)
            actions.append(
                ActionSubmit(
                    at=worker_id,
                    tasks=[task_id],
                    outputs = list(outputs),
                )
            )
            available_ds.update(outputs)
    return actions

def purges(schedule: Schedule, state: State) -> list[Action]:
    """Given remaining schedule, identify unnecessary datasets"""
    actions: list[Action] = [
        ActionDatasetPurge(
            ds=[e],
            at=project_colocation(list(state.ds2worker[e].keys()), state.worker_colocations),
        )
        for e in state.purging_queue
    ]
    state.purging_queue = []
    return actions

def plan(schedule: Schedule, state: State, environment: Environment, job: JobInstance, taskInputs: dict[TaskId, set[DatasetId]]) -> list[Action]:
    actions = purges(schedule, state)

    layer_idx: int|None = 0
    available_workers = get_available_workers(state, environment)
    logger.debug(f"about to start planning with {available_workers=} and {len(schedule.computable)=}")
    assignment = fitness_assignment(environment, available_workers, state, taskInputs, job, schedule)
    logger.debug(f"processing {assignment = }")
    actions += convert(assignment, state, job, taskInputs)
    assigned_tasks = [t for tl in assignment.values() for t in tl]
    available_datasets = set(d for d, w2s in state.ds2worker.items() if DatasetStatus.available in set(w2s.values()))
    assume_computed(schedule, available_datasets, assigned_tasks, taskInputs, job)

    logger.debug(f"planned {actions = }")
    return actions
