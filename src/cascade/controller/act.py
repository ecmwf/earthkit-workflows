# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Implements the invocation of Bridge/Executor methods given a sequence of Actions"""

import logging
from typing import Iterable, Iterator, cast

from cascade.controller.core import State
from cascade.executor.bridge import Bridge
from cascade.executor.checkpoints import build_retrieve_command, possible_repersist, retrieve_dataset
from cascade.executor.msg import DatasetPublished, TaskSequence
from cascade.low.core import DatasetId
from cascade.low.execution_context import JobExecutionContext, VirtualCheckpointHost
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
        extra_env=assignment.extra_env,
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


def flush_queues(bridge: Bridge, state: State, context: JobExecutionContext):
    """Flushes elements in purging and fetching queues in State. Marks the respective
    changes in Context, sends commands via Bridge. Mutates State, JobExecutionContext,
    and via bridge the Executors.
    """

    for dataset, host in state.drain_fetching_queue():
        if host != VirtualCheckpointHost:
            bridge.fetch(dataset, host)
        else:
            # NOTE we would rather not be here, but we dont generally expect
            # checkpointed datasets to be outputs. If needbe, send a command
            # to any worker, or spawn a thread with this
            logger.warning(f"execute checkpoint retrieve on controller")
            # NOTE the host is the virtual one so the message is not really valid, but no big deal
            virtual_command = build_retrieve_command(bridge.checkpoint_spec, dataset, host)
            buffer = retrieve_dataset(virtual_command)
            try:
                # the cast is wrong but ty is bit confused about memoryview anyway
                state.receive_payload(dataset, cast(bytes, buffer.view()), buffer.deser_fun)
            finally:
                buffer.close()

    for dataset, host in state.drain_persist_queue():
        if host != VirtualCheckpointHost:
            bridge.persist(dataset, host)
        else:
            possible_repersist(dataset, bridge.checkpoint_spec)
            state.acknowledge_persist(dataset)

    for ds in state.drain_purging_queue():
        for host in context.purge_dataset(ds):
            if host != VirtualCheckpointHost:
                logger.debug(f"issuing purge of {ds=} to {host=}")
                bridge.purge(host, ds)

    return state

def virtual_checkpoint_publish(datasets: Iterable[DatasetId]) -> Iterator[DatasetPublished]:
    """Virtual in the sense of not actually sending any message, but instead simulating
    a response so that controller.notify can bring the contexts into the right state.
    Invoked once, at the job start, after the checkpoint has been listed"""
    return (
        DatasetPublished(
            origin=VirtualCheckpointHost,
            ds=dataset,
            transmit_idx=None,
        )
        for dataset in datasets
    )
