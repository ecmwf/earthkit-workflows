"""
This module defines all messages used to communicate in between executor instances, as well
as externally to eg Controller or Runner
"""

# NOTE about representation -- we could have gone with pydantic, but since we wouldnt use
# its native serde to json (due to having binary data, sets, etc), validation or swagger
# generation, there is no point in the overhead. We are sticking to plain dataclasses

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cascade.low.core import TaskId, DatasetId

## Meta

VERSION = 1 # for serde compatibility

@runtime_checkable
class Message(Protocol)
    @property
    def __sdver__(self) -> int:
        raise NotImplementedError

def element(clazz):
    """Any non-primitive class that can appear in a message"""
    # NOTE possibly add more ext like forcing __slot__
    return dataclass(frozen=True)(clazz)

def message(clazz):
    """A top-level standalone message that can be serialized"""
    clazz = element(clazz)
    clazz.__sdver__ = VERSION
    return clazz

## Msgs

BackboneAddress: str # eg zmq address

@message
class TaskSequence:
    worker: WorkerId # worker for running those tasks
    tasks: list[TaskId] # to be executed in the given order
    publish: set[DatasetId] # set of outputs to be published

@message
class ExecutionContext:
    """A projection of JobInstance relevant to particular TaskSequence"""
    # NOTE once we have long lived workers, this would be replaced by full JobInstance present at the worker
    tasks: dict[TaskId, TaskInstance]
    # int: pos argument, str: kw argument. Values are dsid + annotation
    param_source: dict[TaskId, dict[int|str, tuple[DatasetId, str]]
    callback: BackboneAddress

@message
class TaskFailure:
    worker: WorkerId
    task: TaskId|None
    detail: str

@message
class TaskSuccess:
    worker: WorkerId
    ts: TaskId

@message
class DatasetPublished:
    host: HostId
    ds: DatasetId
