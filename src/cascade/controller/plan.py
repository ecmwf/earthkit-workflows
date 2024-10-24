"""
Given a Schedule and State, issue which Actions should happen next
"""

import logging

from cascade.scheduler.core import Schedule
from cascade.low.core import JobInstance, DatasetId, WorkerId, TaskId, Environment
from cascade.low.func import maybe_head
from cascade.controller.core import State, TaskStatus, Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit

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

def reasonable_assignment(workers: set[WorkerId], state: State, env: Environment, tasks: list[TaskId], taskInputs: dict[TaskId, set[DatasetId]]) -> dict[WorkerId, list[TaskId]]:
    """Returns which workers should compute which tasks"""
    # TODO consider required memory transfers etc
    available_somewhere = set(state.ds2worker.keys())
    computable = {task for task in tasks if taskInputs.get(task, set()) <= available_somewhere}
    return {
        worker_id: [task_id]
        for worker_id, task_id in zip(workers, computable)
    }

def convert(assignment: dict[WorkerId, list[TaskId]], state: State, job: JobInstance, taskInputs: dict[TaskId, set[DatasetId]]) -> list[Action]:
    """Converts the assignment into actions -- which may be data transfers or subgraphs"""
    # TODO multiple optimizations required here:
    # - fuse tasks, trim outputs
    # - transmit from workers that have available already
    # - fuse transmits
    actions: list[Action] = []
    for worker_id, tasks in assignment.items():
        available_ds = set(state.worker2ds[worker_id])
        for task_id in tasks:
            missing = taskInputs.get(task_id, set()) - available_ds
            outputs = job.outputs_of(task_id)
            if not missing:
                actions.append(
                    ActionSubmit(
                        at=worker_id,
                        tasks=[task_id],
                        outputs = outputs,
                    ),
                )
            else:
                for e in missing:
                    fr = maybe_head(state.ds2worker[e].keys())
                    if not fr:
                        raise ValueError(f"cannot find host with dataset {e}")
                    actions.append(
                        ActionDatasetTransmit(
                            ds={e},
                            fr={fr},
                            to={worker_id},
                        )
                    )
                available_ds.update(missing)
            available_ds.update(outputs)
    return actions

def purges(schedule: Schedule, state: State) -> list[Action]:
    """Given remaining schedule, identify unnecessary datasets"""
    actions: list[Action] = [
        ActionDatasetPurge(ds={e}, at=set(state.ds2worker[e].keys()))
        for e in state.purging_queue
    ]
    state.purging_queue = []
    return actions

def plan(schedule: Schedule, state: State, env: Environment, job: JobInstance, taskInputs: dict[TaskId, set[DatasetId]]) -> list[Action]:
    actions = purges(schedule, state)

    layer_idx: int|None = 0
    available_workers = get_available_workers(state, env)
    logger.debug(f"about to start planning with {available_workers=}")
    while layer_idx is not None and layer_idx < len(schedule.layers) and available_workers:
        current_layer = schedule.layers[layer_idx]
        logger.debug(f"planning with {current_layer=} of {layer_idx=} and {available_workers=}")
        next_assignment = reasonable_assignment(available_workers, state, env, current_layer, taskInputs)
        if not next_assignment:
            layer_idx += 1 # TODO suboptimal -- possibly we'd like to jump further, or assign in fusing mode instead
        logger.debug(f"processing {next_assignment = }")
        actions += convert(next_assignment, state, job, taskInputs)
        layer_idx = schedule.schedule_from_layer((t for ts in next_assignment.values() for t in ts), layer_idx)
        # TODO suboptimal -- we'd rather re-compute using get_available_workers and State+Actions
        available_workers -= set(next_assignment.keys())

    logger.debug(f"planned {actions = }")
    return actions
