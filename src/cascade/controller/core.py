"""
Core data structures: State, Event and Action
"""

from enum import Enum
from pydantic import BaseModel
from collections import defaultdict
from cascade.low.core import TaskInstance, Task2TaskEdge, TaskId, DatasetId, WorkerId

class DatasetStatus(int, Enum):
    preparing = 0 # set by controller
    available = 1 # set by executor

class TaskStatus(int, Enum):
    enqueued = 0 # set by controller
    running = 1 # set by executor
    succeeded = 2 # set by executor
    failed = 3 # set by executor

class State:
    """Captures what is where -- datasets, running tasks, ... Used for decision making and progress tracking"""

    def __init__(self, purging_tracker: dict[DatasetId, set[TaskId]]):
        self.worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]] = defaultdict(dict)
        self.ds2worker: dict[DatasetId, dict[WorkerId, DatasetStatus]] = defaultdict(dict)
        self.ts2worker: dict[TaskId, dict[WorkerId, TaskStatus]] = defaultdict(dict)
        self.worker2ts: dict[WorkerId, dict[TaskId, TaskStatus]] = defaultdict(dict)
        self.purging_tracker = purging_tracker
        self.purging_queue: list[DatasetId] = []

class Event(BaseModel):
    at: WorkerId
    ds_trans: list[tuple[DatasetId, DatasetStatus]]
    ts_trans: list[tuple[TaskId, TaskStatus]]

class ActionDatasetPurge(BaseModel):
    ds: set[DatasetId]
    at: set[WorkerId]

class ActionDatasetTransmit(BaseModel):
    ds: set[DatasetId]
    fr: set[WorkerId]
    to: set[WorkerId]

class ActionSubmit(BaseModel):
    at: WorkerId
    tasks: list[TaskId]
    outputs: set[DatasetId] # this is presumably subset of tasks -- some need not be published

Action = ActionDatasetPurge|ActionDatasetTransmit|ActionSubmit
