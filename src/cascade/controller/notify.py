"""
Implements the mutation of State after Executors have reported some Events
"""

# NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
# Thus the caller always *must* use the return value and cease using the input.

import base64
import logging
from typing import Iterable

from cascade.schedulre.core import State, TaskStatus, DatasetStatus
from cascade.executor.bridge import Event
from cascade.low.core import TaskId, DatasetId, WorkerId
from cascade.low.func import assert_never
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

def consider_fetch(state: State, dataset: DatasetId, at: HostId) -> State:
    if dataset in state.outputs and state.outputs[dataset] is None and dataset not in state.fetching_queue:
        state.fetching_queue[dataset] = at
    return state

def consider_computable(state: State, dataset: DatasetId, host: hostId) -> State:
    # In case this is the first time this dataset was made available, we check
    # what tasks can now *in principle* be computed anywhere -- we ignore transfer
    # costs etc here, this is just about updating the `computable` part of `state`.
    # It may happen this is called after a transfer of an already computed dataset, in
    # which case this is a fast no-op
    component = state.components[state.ts2component[dataset.task]]
    for child_task in state.purging_tracker.get(dataset, set()):
        if child_task in component.computable:
            for worker in state.host2workers[host]:
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
                logger.debug(f"{child_task} just became computable!")
                state.computable += 1
                for worker in component.worker2task_distance.keys():
                    # NOTE this is a task newly made computable, so we need to calc
                    # `overhead` for all hosts/workers assigned to the component
                    state = set_worker2task_overhead(state, worker, child_task)

    return state
 
def notify(state: State, events: Iterable[Event]) -> State:
    for event in events:
        if isinstance(event, DatasetPublished):
            logger.debug(f"received {event=}")
            state.host2ds[event.host][event.ds] = DatasetStatus.available
            state.ds2host[event.ds][event.host] = DatasetStatus.available
            state = consider_fetch(state, event.ds, event.host)
            state = consider_computable(state, event.ds, event.host)
            ! # TODO -- how to distinguish transits from task successes?
            mark({"dataset": event.ds, "action": TransmitLifecycle.completed, "target": event.host, "host": "controller"})
        elif isinstance(event, TaskSucceeded):
            logger.debug(f"received {event=}")
            state.worker2ts[event.worker][event.ts] = TaskStatus.succeeded
            state.ts2worker[event.ts][event.worker] = TaskStatus.succeeded
            mark({"task": event.ts, "action": TaskLifecycle.completed, "worker": repr(event.worker), "host": "controller"})
            for sourceDataset in state.edge_i.get(event.ts, set()):
                state.purging_tracker[sourceDataset].remove(event.ts) 
                state = consider_purge(state, sourceDataset)
            if event.ts in state.ongoing[event.worker]:
                logger.debug(f"{task_id} succeeded, removing")
                state.ongoing[event.worker].remove(event.ts)
                state.ongoing_total -= 1
            else:
                raise ValueError(f"{event.ts} succeeded but removal from `ongoing` impossible")
            if not state.ongoing[event.worker]:
                state.idle_workers.add(event.worker)
        elif isinstance(event, DatasetTransmitPayload):
            ! # TODO
        else:
            assert_never(event)
    return state
