"""
Implements the mutation of State after an Executor has reported some Events
"""

import logging
from typing import Iterable
from cascade.controller.core import State, Event, TaskStatus, DatasetStatus
from cascade.controller.views import transition_dataset
from cascade.low.core import TaskId, DatasetId
from cascade.controller.tracing import mark, TaskLifecycle, TransmitLifecycle

logger = logging.getLogger(__name__)

def notify(state: State, events: Iterable[Event], taskInputs: dict[TaskId, set[DatasetId]]) -> State:
    # NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
    # Thus the caller always *must* use the return value and cease using the input.
    for event in events:
        logger.debug(f"received {event = }")
        if event.failures:
            raise ValueError(event.failures)
        for dataset_id, dataset_status in event.ds_trans:
            state = transition_dataset(state, event.at, dataset_id, dataset_status)
            # TODO this is risky -- we don't necessarily know this corresponds exactly to transmits
            if dataset_status == DatasetStatus.available and not event.ts_trans:
                mark({"dataset": dataset_id.task, "action": TransmitLifecycle.completed, "target": event.at, "host": "controller"})
        for task_id, task_status in event.ts_trans:
            state.worker2ts[event.at][task_id] = task_status
            state.ts2worker[task_id][event.at] = task_status
            if task_status == TaskStatus.succeeded:
                mark({"task": task_id, "action": TaskLifecycle.completed, "worker": event.at, "host": "controller"})
                for sourceDataset in taskInputs.get(task_id, set()):
                    state.purging_tracker[sourceDataset].remove(task_id) 
                    if not state.purging_tracker[sourceDataset]:
                        state.purging_tracker.pop(sourceDataset)
                        state.purging_queue.append(sourceDataset)
                if task_id in state.remaining:
                    logger.debug(f"{task_id} succeeded, removing")
                    state.remaining.remove(task_id)
                else:
                    logger.warning(f"{task_id} succeeded but removal from remaining impossible")
            elif task_status == TaskStatus.failed:
                raise ValueError(f"failure of {task_id}")
    return state
