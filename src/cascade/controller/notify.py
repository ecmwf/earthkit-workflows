"""
Implements the mutation of State after Executors have reported some Events
"""

# NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
# Thus the caller always *must* use the return value and cease using the input.

import logging
from typing import Iterable

import cascade.executor.serde as serde
from cascade.executor.bridge import Event
from cascade.executor.msg import DatasetPublished, DatasetTransmitPayload
from cascade.low.core import DatasetId, HostId, JobInstance, WorkerId
from cascade.low.func import assert_never
from cascade.low.tracing import TaskLifecycle, TransmitLifecycle, mark
from cascade.scheduler.assign import set_worker2task_overhead
from cascade.scheduler.core import DatasetStatus, State, TaskStatus
from cascade.controller.report import Reporter

logger = logging.getLogger(__name__)


def consider_purge(state: State, dataset: DatasetId) -> State:
    no_dependants = not state.purging_tracker[dataset]
    not_required_output = (dataset not in state.outputs) or (
        state.outputs[dataset] is not None
    )
    if no_dependants and not_required_output:
        state.purging_tracker.pop(dataset)
        state.purging_queue.append(dataset)
    return state


def consider_fetch(state: State, dataset: DatasetId, at: HostId) -> State:
    if (
        dataset in state.outputs
        and state.outputs[dataset] is None
        and dataset not in state.fetching_queue
    ):
        state.fetching_queue[dataset] = at
    return state


def consider_computable(state: State, dataset: DatasetId, host: HostId) -> State:
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


def is_last_output_of(dataset: DatasetId, job: JobInstance) -> bool:
    definition = job.tasks[dataset.task].definition
    # TODO change the definition to actually be the sorted list
    last = sorted(definition.output_schema.keys())[-1]  
    return last == dataset.output


def notify(state: State, job: JobInstance, events: Iterable[Event], reporter: Reporter) -> State:
    for event in events:
        if isinstance(event, DatasetPublished):
            logger.debug(f"received {event=}")
            # NOTE here we'll need to distinguish memory-only and host-wide (shm) publications, currently all events mean shm
            host = (
                event.origin if isinstance(event.origin, HostId) else event.origin.host
            )
            state.host2ds[host][event.ds] = DatasetStatus.available
            state.ds2host[event.ds][host] = DatasetStatus.available
            state = consider_fetch(state, event.ds, host)
            state = consider_computable(state, event.ds, host)
            if event.transmit_idx is not None:
                mark(
                    {
                        "dataset": repr(event.ds),
                        "action": TransmitLifecycle.completed,
                        "target": host,
                        "host": "controller",
                    }
                )
            elif is_last_output_of(event.ds, job):
                if not isinstance((worker := event.origin), WorkerId):
                    raise ValueError(
                        f"malformed event, expected origin to be WorkerId: {event}"
                    )
                logger.debug(
                    f"last output of {event.ds.task} published, assuming completion"
                )
                state.worker2ts[worker][event.ds.task] = TaskStatus.succeeded
                state.ts2worker[event.ds.task][worker] = TaskStatus.succeeded
                mark(
                    {
                        "task": event.ds.task,
                        "action": TaskLifecycle.completed,
                        "worker": repr(worker),
                        "host": "controller",
                    }
                )
                for sourceDataset in state.edge_i.get(event.ds.task, set()):
                    state.purging_tracker[sourceDataset].remove(event.ds.task)
                    state = consider_purge(state, sourceDataset)
                if event.ds.task in state.ongoing[worker]:
                    state.ongoing[worker].remove(event.ds.task)
                    state.ongoing_total -= 1
                    state.remaining -= 1
                    reporter.send_progress(state)
                else:
                    raise ValueError(
                        f"{event.ds.task} succeeded but removal from `ongoing` impossible"
                    )
                if not state.ongoing[worker]:
                    state.idle_workers.add(worker)
        elif isinstance(event, DatasetTransmitPayload):
            # TODO ifneedbe get annotation from job.tasks[event.ds.task].definition.output_schema[event.ds.output]
            state.outputs[event.header.ds] = serde.des_output(
                event.value, "Any", event.header.deser_fun
            )
            reporter.send_result(event.header.ds, event.value)
        else:
            assert_never(event)
    return state
