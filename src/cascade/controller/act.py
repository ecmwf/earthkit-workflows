# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Implements the invocation of Bridge/Executor methods given a sequence of Actions"""

import logging

from cascade.controller.core import State
from cascade.controller.notify import consider_purge
from cascade.executor.bridge import Bridge
from cascade.executor.msg import TaskSequence
from cascade.low.execution_context import JobExecutionContext
from cascade.low.tracing import TaskLifecycle, TransmitLifecycle, mark
from cascade.scheduler.core import Assignment

logger = logging.getLogger(__name__)


def act(bridge: Bridge, assignment: Assignment) -> None:
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


# TODO refac less explicit mutation of context, use class methods
def flush_queues(bridge: Bridge, state: State, context: JobExecutionContext):
    """Flushes elements in purging and fetching queues in State. Mutates State, JobExecutionContext,
    and via bridge the Executors.
    """

    # TODO handle this in some eg thread pool... may need lock on state, result queueing, handle purge tracking, etc
    fetchable = list(state.fetching_queue.keys())
    for dataset in fetchable:
        host = state.fetching_queue.pop(dataset)
        bridge.fetch(dataset, host)
        consider_purge(state, dataset)

    for ds in state.purging_queue:
        # TODO finegraining, restrictions, checks for validity, etc. Do in concert with extension of `purging_queue`
        for host in context.ds2host[ds]:
            logger.debug(f"identified purge of {ds} at {host}")
            bridge.purge(host, ds)
            context.host2ds[host].pop(ds)
            for worker in context.host2workers[host]:
                if ds in context.worker2ds[worker]:
                    context.worker2ds[worker].pop(ds)
                    context.ds2worker[ds].pop(worker)
        context.ds2host.pop(ds)
    state.purging_queue = []

    return state
