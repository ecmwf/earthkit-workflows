"""
Core data structures: Event and Action
"""

# TODO consider merge with the other protocolar commands in serde / backbone

from collections import defaultdict
from enum import Enum

from cascade.low.core import TaskInstance, Task2TaskEdge, TaskId, DatasetId, WorkerId, HostId
from cascade.scheduler.core import DatasetStatus, TaskStatus, State
from pydantic import BaseModel, Field

class Event(BaseModel):
    at: WorkerId
    ds_trans: list[tuple[DatasetId, DatasetStatus]] = Field(default_factory=list)
    ts_trans: list[tuple[TaskId, TaskStatus]] = Field(default_factory=list)
    # catch-all for when something irreparable goes wrong at the executor.
    # TODO replace with fine-grained retriable causes
    failures: list[str] = Field(default_factory=list)
    # TODO str here is b64 of bytes... fix!
    ds_fetch: list[tuple[DatasetId, str]] = Field(default_factory=list)

class ActionDatasetPurge(BaseModel):
    ds: list[DatasetId]
    workers: list[WorkerId]
    at: HostId

class ActionDatasetTransmit(BaseModel):
    ds: list[DatasetId]
    fr: list[HostId]
    to: list[WorkerId]

class ActionSubmit(BaseModel):
    at: WorkerId
    tasks: list[TaskId]
    outputs: list[DatasetId] # this is presumably subset of tasks -- some need not be published

Action = ActionDatasetPurge|ActionDatasetTransmit|ActionSubmit

class TransmitPayload(BaseModel):
    # corresponds to ActionDatasetTransmit but used for remote transmits, to one of the sides
    other_url: str
    other_worker: WorkerId
    this_host: HostId
    datasets: list[DatasetId]
