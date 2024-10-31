"""
Implements the invocation of Executor methods given a sequence of Actions
"""

import logging
from cascade.controller.executor import Executor
from cascade.controller.core import State, Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, DatasetStatus, TaskStatus
from typing import Iterable
from cascade.controller.tracing import mark

logger = logging.getLogger(__name__)

def act(executor: Executor, state: State, actions: Iterable[Action]) -> State:
    # NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
    # Thus the caller always *must* use the return value and cease using the input.
    # The executor is obviously also mutated.

    for action in actions:
        logger.debug(f"sending {action = } to executor")
        if isinstance(action, ActionDatasetPurge):
            for dataset in action.ds:
                for worker in action.at:
                    state.worker2ds[worker].pop(dataset)
                    state.ds2worker[dataset].pop(worker)
            executor.purge(action)
        elif isinstance(action, ActionDatasetTransmit):
            for dataset in action.ds:
                for worker in action.to:
                    mark({"dataset": dataset.task, "action": "transmitPlanned", "worker": worker, "host": "controller"})
                    state.worker2ds[worker][dataset] = DatasetStatus.preparing
                    state.ds2worker[dataset][worker] = DatasetStatus.preparing
            executor.transmit(action)
        elif isinstance(action, ActionSubmit):
            for task in action.tasks:
                mark({"task": task, "action": "taskPlanned", "worker": action.at, "host": "controller"})
                state.worker2ts[action.at][task] = TaskStatus.enqueued
                state.ts2worker[task][action.at] = TaskStatus.enqueued
                state.remaining.add(task)
            for dataset in action.outputs:
                state.worker2ds[action.at][dataset] = DatasetStatus.preparing
                state.ds2worker[dataset][action.at] = DatasetStatus.preparing
            executor.submit(action)

    return state
