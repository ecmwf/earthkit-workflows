"""
Core data structures: State, Event and Action
"""

from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict
from cascade.low.core import TaskInstance, Task2TaskEdge, TaskId, DatasetId, WorkerId

class DatasetStatus(int, Enum):
    missing = -1 # virtual default status, never stored
    preparing = 0 # set by controller
    available = 1 # set by executor
    purged = 2 # temporal command status used as local comms between controller.act and controller.state

class TaskStatus(int, Enum):
    enqueued = 0 # set by controller
    running = 1 # set by executor
    succeeded = 2 # set by executor
    failed = 3 # set by executor

class State:
    """Captures what is where -- datasets, running tasks, ... Used for decision making and progress tracking"""

    def __init__(self, purging_tracker: dict[DatasetId, set[TaskId]], worker_colocations: dict[WorkerId, set[WorkerId]]):
        self.worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]] = defaultdict(dict)
        self.ds2worker: dict[DatasetId, dict[WorkerId, DatasetStatus]] = defaultdict(dict)
        self.ts2worker: dict[TaskId, dict[WorkerId, TaskStatus]] = defaultdict(dict)
        self.worker2ts: dict[WorkerId, dict[TaskId, TaskStatus]] = defaultdict(dict)
        self.remaining: set[TaskId] = set()
        self.purging_tracker = purging_tracker
        self.purging_queue: list[DatasetId] = []
        self.worker_colocations = worker_colocations

class Event(BaseModel):
    at: WorkerId
    ds_trans: list[tuple[DatasetId, DatasetStatus]] = Field(default_factory=list)
    ts_trans: list[tuple[TaskId, TaskStatus]] = Field(default_factory=list)
    # catch-all for when something irreparable goes wrong at the executor.
    # TODO replace with fine-grained retriable causes
    failures: list[str] = Field(default_factory=list)

class ActionDatasetPurge(BaseModel):
    ds: list[DatasetId]
    at: list[WorkerId]

class ActionDatasetTransmit(BaseModel):
    ds: list[DatasetId]
    fr: list[WorkerId]
    to: list[WorkerId]

class ActionSubmit(BaseModel):
    at: WorkerId
    tasks: list[TaskId]
    outputs: list[DatasetId] # this is presumably subset of tasks -- some need not be published

Action = ActionDatasetPurge|ActionDatasetTransmit|ActionSubmit

class TransmitPayload(BaseModel):
    # corresponds to ActionDatasetTransmit but used for remote transmits, to one of the sides
    other_url: str
    other_worker: str
    this_worker: str
    datasets: list[DatasetId]
    tracing_ctx_host: str
