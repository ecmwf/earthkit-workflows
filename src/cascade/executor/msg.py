"""
This module defines all messages used to communicate in between executor instances, as well
as externally to eg Controller or Runner
"""

# NOTE about representation -- we could have gone with pydantic, but since we wouldnt use
# its native serde to json (due to having binary data, sets, etc), validation or swagger
# generation, there is no point in the overhead. We are sticking to plain dataclasses

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cascade.low.core import TaskId, DatasetId, WorkerId, TaskInstance, HostId

## Meta

VERSION = 1 # for serde compatibility

# TODO use dataclass_transform to get mypy understand that @message/@element produces dataclasses, and that @message produces Messages
# Then replace the `@dataclass`es below with @message/@element
# NOTE how would we deal with existing dataclasses like WorkerId?

@runtime_checkable
class _Message(Protocol):
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

BackboneAddress = str # eg zmq address

@dataclass(frozen=True)
class TaskSequence:
    worker: WorkerId # worker for running those tasks
    tasks: list[TaskId] # to be executed in the given order
    publish: set[DatasetId] # set of outputs to be published

@dataclass(frozen=True)
class ExecutionContext:
    """A projection of JobInstance relevant to particular TaskSequence"""
    # NOTE once we have long lived workers, this would be replaced by full JobInstance present at the worker
    tasks: dict[TaskId, TaskInstance]
    # int: pos argument, str: kw argument. Values are dsid + annotation
    param_source: dict[TaskId, dict[int|str, tuple[DatasetId, str]]]
    callback: BackboneAddress

@dataclass(frozen=True)
class TaskFailure:
    worker: WorkerId
    task: TaskId|None
    detail: str

@dataclass(frozen=True)
class TaskSuccess:
    worker: WorkerId
    ts: TaskId

@dataclass(frozen=True)
class DatasetPublished:
    host: HostId
    ds: DatasetId
    from_transmit: bool # just for tracking purposes

@dataclass(frozen=True)
class DatasetPurge:
    ds: DatasetId

@dataclass(frozen=True)
class DatasetTransmitCommand:
    # NOTE atm we don't need source/target, but they are useful for tracing purposes. And we may need it later
    source: HostId
    target: HostId
    daddress: BackboneAddress
    ds: DatasetId

@dataclass(frozen=True)
class DatasetTransmitPayload:
    ds: DatasetId
    value: bytes

@dataclass(frozen=True)
class DatasetTransmitFailure:
    host: HostId

@dataclass(frozen=True)
class ExecutorFailure:
    host: HostId
    detail: str

@dataclass(frozen=True)
class ExecutorExit:
    host: HostId

@dataclass(frozen=True)
class ExecutorRegistration:
    host: HostId
    maddress: BackboneAddress
    daddress: BackboneAddress
    workers: list[WorkerId]
    # TODO resource capacity etc... reuse the Environment?

@dataclass(frozen=True)
class ExecutorShutdown:
    pass

# this explicit list is a disgrace -- see the _Message protocol above
Message = TaskSequence|TaskFailure|TaskSuccess|DatasetPublished|DatasetPurge|DatasetTransmitCommand|DatasetTransmitPayload|ExecutorFailure|ExecutorExit|ExecutorRegistration|ExecutorShutdown|DatasetTransmitFailure
