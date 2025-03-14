"""
Implements the invocation of Bridge/Executor methods given a sequence of Actions
"""

import logging
from typing import Iterator

from cascade.controller.notify import consider_purge
from cascade.executor.bridge import Bridge
from cascade.executor.msg import TaskSequence
from cascade.low.func import assert_never
from cascade.low.tracing import TaskLifecycle, TransmitLifecycle, mark
from cascade.scheduler.core import Assignment, DatasetStatus, State, TaskStatus

logger = logging.getLogger(__name__)


def act(bridge: Bridge, state: State, assignment: Assignment) -> None:
    """Converts an assignment to one or more actions which are sent to the bridge, and returned
    for tracing/updating purposes. Does *not* mutate State, but executors behind the Bridge *are* mutated.
    """

    for prep in assignment.prep:
        ds = prep[0]
        source_host = prep[1]
        if assignment.worker.host == source_host:
            logger.debug(
                f"dataset {ds} should be locally available at {assignment.worker.host}, doing no-op"
            )
            continue
        logger.debug(
            f"sending transmit ({ds}: {source_host}=>{assignment.worker.host}) to bridge"
        )
        mark(
            {
                "dataset": repr(ds),
                "action": TransmitLifecycle.planned,
                "source": source_host,
                "target": assignment.worker.host,
                "host": "controller",
            }
        )
        bridge.transmit(ds, source_host, assignment.worker.host)

    task_sequence = TaskSequence(
        worker=assignment.worker,
        tasks=assignment.tasks,
        publish=assignment.outputs,
    )

    for task in assignment.tasks:
        mark(
            {
                "task": task,
                "action": TaskLifecycle.planned,
                "worker": repr(assignment.worker),
                "host": "controller",
            }
        )
    logger.debug(f"sending {task_sequence} to bridge")
    bridge.task_sequence(task_sequence)


def flush_queues(bridge: Bridge, state: State) -> State:
    """Flushes elements in purging and fetching queues in State (and mutating it thus, as well as Executor).
    Returns the mutated State, as all tracing and updates are handled here."""

    # TODO handle this in some eg thread pool... may need lock on state, result queueing, handle purge tracking, etc
    fetchable = list(state.fetching_queue.keys())
    for dataset in fetchable:
        host = state.fetching_queue.pop(dataset)
        bridge.fetch(dataset, host)
        state = consider_purge(state, dataset)

    for ds in state.purging_queue:
        # TODO finegraining, restrictions, checks for validity, etc. Do in concert with extension of `purging_queue`
        for host in state.ds2host[ds]:
            logger.debug(f"identified purge of {ds} at {host}")
            bridge.purge(host, ds)
            state.host2ds[host].pop(ds)
            for worker in state.host2workers[host]:
                if ds in state.worker2ds[worker]:
                    state.worker2ds[worker].pop(ds)
                    state.ds2worker[ds].pop(worker)
        state.ds2host.pop(ds)
    state.purging_queue = []

    return state
