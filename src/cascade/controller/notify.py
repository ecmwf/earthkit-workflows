"""
Implements the mutation of State after an Executor has reported some Events
"""

import logging
from typing import Iterable
from cascade.controller.core import State, Event, TaskStatus
from cascade.low.core import TaskId, DatasetId

logger = logging.getLogger(__name__)

def notify(state: State, events: Iterable[Event], taskInputs: dict[TaskId, set[DatasetId]]) -> State:
    # NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
    # Thus the caller always *must* use the return value and cease using the input.
    for event in events:
        logger.debug(f"received {event = }")
        for dataset_id, dataset_status in event.ds_trans:
            state.worker2ds[event.at][dataset_id] = dataset_status 
            state.ds2worker[dataset_id][event.at] = dataset_status
        for task_id, task_status in event.ts_trans:
            state.worker2ts[event.at][task_id] = task_status
            state.ts2worker[task_id][event.at] = task_status
            if task_status == TaskStatus.succeeded:
                for sourceDataset in taskInputs.get(task_id, set()):
                    state.purging_tracker[sourceDataset].remove(task_id) 
                    if not state.purging_tracker[sourceDataset]:
                        state.purging_tracker.pop(sourceDataset)
                        state.purging_queue.append(sourceDataset)
    return state
