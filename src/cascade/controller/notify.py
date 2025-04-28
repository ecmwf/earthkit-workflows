# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Implements the mutation of State after Executors have reported some Events"""

# NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
# Thus the caller always *must* use the return value and cease using the input.

import logging
from typing import Iterable

import cascade.executor.serde as serde
from cascade.controller.core import State
from cascade.controller.report import Reporter
from cascade.executor.bridge import Event
from cascade.executor.msg import DatasetPublished, DatasetTransmitPayload
from cascade.low.core import DatasetId, HostId, WorkerId
from cascade.low.execution_context import DatasetStatus, JobExecutionContext, TaskStatus
from cascade.low.func import assert_never
from cascade.low.tracing import TaskLifecycle, TransmitLifecycle, mark
from cascade.scheduler.assign import set_worker2task_overhead
from cascade.scheduler.core import Schedule

logger = logging.getLogger(__name__)


def consider_purge(state: State, dataset: DatasetId):
    no_dependants = not state.purging_tracker.get(dataset, None)
    not_required_output = state.outputs.get(dataset, 1) is not None
    if no_dependants and not_required_output:
        logger.debug(f"adding {dataset=} to purging queue")
        if dataset in state.purging_tracker:
            state.purging_tracker.pop(dataset)
        state.purging_queue.append(dataset)


def consider_fetch(state: State, dataset: DatasetId, at: HostId):
    if (
        dataset in state.outputs
        and state.outputs[dataset] is None
        and dataset not in state.fetching_queue
    ):
        state.fetching_queue[dataset] = at


# TODO refac move to scheduler
def consider_computable(
    schedule: Schedule,
    state: State,
    context: JobExecutionContext,
    dataset: DatasetId,
    host: HostId,
):
    # In case this is the first time this dataset was made available, we check
    # what tasks can now *in principle* be computed anywhere -- we ignore transfer
    # costs etc here, this is just about updating the `computable` part of `state`.
    # It may happen this is called after a transfer of an already computed dataset, in
    # which case this is a fast no-op
    component = schedule.components[schedule.ts2component[dataset.task]]
    # TODO refac do we need purging_tracker here, or is edge_o enough?
    for child_task in state.purging_tracker.get(dataset, set()):
        if child_task in component.computable:
            for worker in context.host2workers[host]:
                # NOTE since the child_task has already been computable, and the current
                # implementation of `overhead` assumes host2host being homogeneous, we can
                # afford to recalc overhead for the event's host only
                set_worker2task_overhead(schedule, context, worker, child_task)
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
                schedule.computable += 1
                for worker in component.worker2task_distance.keys():
                    # NOTE this is a task newly made computable, so we need to calc
                    # `overhead` for all hosts/workers assigned to the component
                    set_worker2task_overhead(schedule, context, worker, child_task)


# TODO refac less explicit mutation of context, use class methods
def notify(
    state: State,
    schedule: Schedule,
    context: JobExecutionContext,
    events: Iterable[Event],
    reporter: Reporter,
):
    for event in events:
        if isinstance(event, DatasetPublished):
            logger.debug(f"received {event=}")
            # NOTE here we'll need to distinguish memory-only and host-wide (shm) publications, currently all events mean shm
            host = (
                event.origin if isinstance(event.origin, HostId) else event.origin.host
            )
            context.host2ds[host][event.ds] = DatasetStatus.available
            context.ds2host[event.ds][host] = DatasetStatus.available
            consider_fetch(state, event.ds, host)
            consider_computable(schedule, state, context, event.ds, host)
            if event.transmit_idx is not None:
                mark(
                    {
                        "dataset": repr(event.ds),
                        "action": TransmitLifecycle.completed,
                        "target": host,
                        "host": "controller",
                    }
                )
            elif context.is_last_output_of(event.ds):
                if not isinstance((worker := event.origin), WorkerId):
                    raise ValueError(
                        f"malformed event, expected origin to be WorkerId: {event}"
                    )
                logger.debug(
                    f"last output of {event.ds.task} published, assuming completion"
                )
                context.worker2ts[worker][event.ds.task] = TaskStatus.succeeded
                context.ts2worker[event.ds.task][worker] = TaskStatus.succeeded
                mark(
                    {
                        "task": event.ds.task,
                        "action": TaskLifecycle.completed,
                        "worker": repr(worker),
                        "host": "controller",
                    }
                )
                for sourceDataset in context.edge_i.get(event.ds.task, set()):
                    state.purging_tracker[sourceDataset].remove(event.ds.task)
                    consider_purge(state, sourceDataset)
                if event.ds.task in context.ongoing[worker]:
                    context.ongoing[worker].remove(event.ds.task)
                    context.ongoing_total -= 1
                    context.remaining -= 1
                    reporter.send_progress(context)
                else:
                    raise ValueError(
                        f"{event.ds.task} succeeded but removal from `ongoing` impossible"
                    )
                if not context.ongoing[worker]:
                    context.idle_workers.add(worker)
        elif isinstance(event, DatasetTransmitPayload):
            # TODO ifneedbe get annotation from job.tasks[event.ds.task].definition.output_schema[event.ds.output]
            state.outputs[event.header.ds] = serde.des_output(
                event.value, "Any", event.header.deser_fun
            )
            reporter.send_result(event.header.ds, event.value)
        else:
            assert_never(event)
    return state
