"""
Implements the invocation of Executor methods given a sequence of Actions
"""

import logging
from typing import Iterator

from cascade.controller.core import State, Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, DatasetStatus, TaskStatus
from cascade.controller.executor import Executor
from cascade.controller.notify import consider_purge
from cascade.low.func import assert_never
from cascade.low.tracing import mark, TaskLifecycle, TransmitLifecycle
from cascade.scheduler.core import Assignment

logger = logging.getLogger(__name__)

def act(executor: Executor, state: State, assignment: Assignment) -> Iterator[Action]:
    """Converts an assignment to one or more actions which are sent to the executor, and returned
    for tracing/updating purposes. Does *not* mutate State, but Executor *is* mutated."""

    for prep in assignment.prep:
        ds = prep[0] 
        source_host = prep[1]
        if assignment.worker == source_host:
            logger.debug(f"dataset {ds} should be locally available, doing no-op")
            continue
        action_transmit = ActionDatasetTransmit(
            ds=[ds],
            fr=[source_host],
            to=[assignment.worker],
        )
        logger.debug(f"sending {action_transmit} to executor")
        mark({"dataset": ds.task, "action": TransmitLifecycle.planned, "source": source_host, "target": repr(assignment.worker), "host": "controller"})
        executor.transmit(action_transmit)
        yield action_transmit

    action_submit = ActionSubmit(
        at=assignment.worker,
        tasks=assignment.tasks,
        outputs=list(assignment.outputs),
    )
    for task in assignment.tasks:
        mark({"task": task, "action": TaskLifecycle.planned, "worker": repr(assignment.worker), "host": "controller"})
    logger.debug(f"sending {action_submit} to executor")
    executor.submit(action_submit)
    yield action_submit


def flush_queues(executor: Executor, state: State) -> State:
    """Flushes elements in purging and fetching queues in State (and mutating it thus, as well as Executor).
    Returns the mutated State, as all tracing and updates are handled here."""

    # TODO handle this in some eg thread pool... may need lock on state, result queueing, handle purge tracking, etc
    fetchable = list(state.fetching_queue.keys())
    for dataset in fetchable:
        worker = state.fetching_queue.pop(dataset)
        if hasattr(executor, "backbone"):
            executor.lazyfetch_value(worker, dataset) # type: ignore
        else:
            state.outputs[dataset] = executor.fetch_as_value(dataset)
            state = consider_purge(state, dataset)

    for ds in state.purging_queue:
        # TODO finegraining, restrictions, checks for validity, etc. Do in concert with extension of `purging_queue`
        for host in state.ds2host[ds]:
            action_purge = ActionDatasetPurge(
                ds=[ds],
                workers=state.host2workers[host],
                at=host,
            )
            logger.debug(f"identified purge action {action_purge}")
            executor.purge(action_purge)
            state.host2ds[host].pop(ds)
            for worker in state.host2workers[host]:
                if ds in state.worker2ds[worker]:
                    state.worker2ds[worker].pop(ds)
                    state.ds2worker[ds].pop(worker)
        state.ds2host.pop(ds)
    state.purging_queue = []

    return state
