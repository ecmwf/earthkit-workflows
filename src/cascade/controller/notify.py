"""
Implements the mutation of State after an Executor has reported some Events
"""

# NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
# Thus the caller always *must* use the return value and cease using the input.

import base64
import logging
from typing import Iterable

from cascade.controller.core import State, Event, TaskStatus, DatasetStatus, WorkerId
from cascade.low.core import TaskId, DatasetId
from cascade.low.tracing import mark, TaskLifecycle, TransmitLifecycle
from cascade.scheduler.assign import set_worker2task_overhead

logger = logging.getLogger(__name__)

def consider_purge(state: State, dataset: DatasetId) -> State:
    no_dependants = not state.purging_tracker[dataset]
    not_required_output = (dataset not in state.outputs) or (state.outputs[dataset] is not None)
    if no_dependants and not_required_output:
        state.purging_tracker.pop(dataset)
        state.purging_queue.append(dataset)
    return state

def consider_fetch(state: State, dataset: DatasetId, at: WorkerId) -> State:
    if dataset in state.outputs and state.outputs[dataset] is None and dataset not in state.fetching_queue:
        state.fetching_queue[dataset] = at
    return state

def consider_computable(state: State, dataset: DatasetId, workerId: WorkerId) -> State:
    # In case this is the first time this dataset was made available, we check
    # what tasks can now *in principle* be computed anywhere -- we ignore transfer
    # costs etc here, this is just about updating the `computable` part of `state`.
    # It may happen this is called after a transfer of an already computed dataset, in
    # which case this is a fast no-op
    component = state.components[state.ts2component[dataset.task]]
    for child_task in state.purging_tracker[dataset]:
        if child_task in component.computable:
            for worker in state.host2workers[workerId.host]:
                # NOTE since the child_task has already been computable, and the current
                # implementation of `overhead` assumes host2host being homogeneous, we can
                # afford to recalc overhead for the event's host only
                state = set_worker2task_overhead(state, worker, child_task)
        if child_task not in component.is_computable_tracker:
            continue
        if dataset in component.is_computable_tracker[child_task]:
            component.is_computable_tracker[child_task].remove(dataset)
            if not component.is_computable_tracker[child_task]:
                component.is_computable_tracker.pop(child_task)
                value = component.core.depth
                for distances in component.worker2task_distance.values():
                    if (new_opt := distances[child_task]) < value:
                        value = new_opt
                component.computable[child_task] = value
                state.computable += 1
                for worker in component.worker2task_distance.keys():
                    # NOTE this is a task newly made computable, so we need to calc
                    # `overhead` for all hosts/workers assigned to the component
                    state = set_worker2task_overhead(state, worker, child_task)

    return state
 
def notify(state: State, events: Iterable[Event]) -> State:
    for event in events:
        if event.failures:
            logger.debug(f"received {event.failures = }")
            raise ValueError(event.failures)
        logger.debug(f"received {event.ds_trans=}")
        for dataset_id, dataset_status in event.ds_trans:
            state.worker2ds[event.at][dataset_id] = dataset_status
            state.ds2worker[dataset_id][event.at] = dataset_status
            state.host2ds[event.at.host][dataset_id] = dataset_status
            state.ds2host[dataset_id][event.at.host] = dataset_status
            if dataset_status == DatasetStatus.available:
                state = consider_fetch(state, dataset_id, event.at)
                state = consider_computable(state, dataset_id, event.at)
            # TODO we don't necessarily know this corresponds exactly to transmits, thus this is imprecise
            if dataset_status == DatasetStatus.available and not event.ts_trans:
                mark({"dataset": dataset_id.task, "action": TransmitLifecycle.completed, "target": repr(event.at), "host": "controller"})
        logger.debug(f"received {event.ts_trans=}")
        for task_id, task_status in event.ts_trans:
            state.worker2ts[event.at][task_id] = task_status
            state.ts2worker[task_id][event.at] = task_status
            if task_status == TaskStatus.succeeded:
                mark({"task": task_id, "action": TaskLifecycle.completed, "worker": repr(event.at), "host": "controller"})
                for sourceDataset in state.edge_i.get(task_id, set()):
                    state.purging_tracker[sourceDataset].remove(task_id) 
                    state = consider_purge(state, sourceDataset)
                if task_id in state.ongoing[event.at]:
                    logger.debug(f"{task_id} succeeded, removing")
                    state.ongoing[event.at].remove(task_id)
                else:
                    logger.warning(f"{task_id} succeeded but removal from `ongoing` impossible")
                if not state.ongoing[event.at]:
                    state.idle_workers.add(event.at)
            elif task_status == TaskStatus.failed:
                raise ValueError(f"failure of {task_id}")
        for dataset, value in event.ds_fetch:
            logger.debug(f"received fetch of {dataset=} of raw len {len(value)}")
            state.outputs[dataset] = base64.b64decode(value)
            state = consider_purge(state, dataset)
    return state
