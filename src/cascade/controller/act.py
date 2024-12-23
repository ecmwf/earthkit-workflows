"""
Implements the invocation of Executor methods given a sequence of Actions
"""

import logging
from cascade.controller.executor import Executor
from cascade.controller.core import State, Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, DatasetStatus, TaskStatus
from cascade.low.func import assert_never
from cascade.controller.views import transition_dataset
from cascade.controller.notify import consider_purge
from typing import Iterable
from cascade.low.tracing import mark, TaskLifecycle, TransmitLifecycle

logger = logging.getLogger(__name__)

! update # TODO just switch the action -> assignment, and don't do some of the transitions as they are done in `plan` instead. However, make sure the `mark` and transition check etc are done *somewhere*. Microtrace here? Prolly no sense due to async nature

def act(executor: Executor, state: State, actions: Iterable[Action]) -> State:
    # NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
    # Thus the caller always *must* use the return value and cease using the input.
    # The executor is obviously also mutated.

    for action in actions:
        logger.debug(f"sending {action = } to executor")
        if isinstance(action, ActionDatasetPurge):
            for dataset in action.ds:
                for worker in action.at:
                    state = transition_dataset(state, worker, dataset, DatasetStatus.purged)
            executor.purge(action)
        elif isinstance(action, ActionDatasetTransmit):
            for dataset in action.ds:
                for target in action.to:
                    for source in action.fr:
                        mark({"dataset": dataset.task, "action": TransmitLifecycle.planned, "source": source, "target": target, "host": "controller"})
                        state = transition_dataset(state, target, dataset, DatasetStatus.preparing)
            executor.transmit(action)
        elif isinstance(action, ActionSubmit):
            for task in action.tasks:
                mark({"task": task, "action": TaskLifecycle.planned, "worker": action.at, "host": "controller"})
                state.worker2ts[action.at][task] = TaskStatus.enqueued
                state.ts2worker[task][action.at] = TaskStatus.enqueued
                state.remaining.add(task)
            for dataset in action.outputs:
                state = transition_dataset(state, action.at, dataset, DatasetStatus.preparing)
            executor.submit(action)
        else:
            assert_never(action)

    # TODO handle this in some thread pool etc... would need locks on state or result queueing etc
    fetchable = list(state.fetching_queue.keys())
    for dataset in fetchable:
        worker = state.fetching_queue.pop(dataset)
        if hasattr(executor, "backbone"):
            executor.lazyfetch_value(worker, dataset) # type: ignore
        else:
            state.outputs[dataset] = executor.fetch_as_value(worker, dataset)
            state = consider_purge(state, dataset)

    return state
