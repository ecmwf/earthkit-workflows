# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module defines all messages used to communicate in between executor instances, as well
as externally to eg Controller or Runner
"""

# TODO split into message categories: worker-only, data-only, regular -- or smth like that

# NOTE about representation -- we could have gone with pydantic, but since we wouldnt use
# its native serde to json (due to having binary data, sets, etc), validation or swagger
# generation, there is no point in the overhead. We are sticking to plain dataclasses

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cascade.low.core import DatasetId, HostId, TaskId, WorkerId

## Meta

VERSION = 1  # for serde compatibility

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


BackboneAddress = str  # eg zmq address

## Msgs


@dataclass(frozen=True)
class Syn:
    idx: int
    addr: BackboneAddress


@dataclass(frozen=True)
class Ack:
    idx: int


@dataclass(frozen=True)
class TaskSequence:
    worker: WorkerId  # worker for running those tasks
    tasks: list[TaskId]  # to be executed in the given order
    publish: set[DatasetId]  # set of outputs to be published


@dataclass(frozen=True)
class TaskFailure:
    worker: WorkerId
    task: TaskId | None
    detail: str


@dataclass(frozen=True)
class DatasetPublished:
    origin: WorkerId | HostId
    ds: DatasetId
    transmit_idx: int | None


@dataclass(frozen=True)
class DatasetPurge:
    ds: DatasetId


@dataclass(frozen=True)
class DatasetTransmitCommand:
    source: HostId
    target: HostId
    daddress: BackboneAddress
    ds: DatasetId
    idx: int  # TODO consider using in tracing all over. Would need scheduler to assign it


@dataclass(frozen=True)
class DatasetTransmitPayloadHeader:
    confirm_address: BackboneAddress
    confirm_idx: int
    ds: DatasetId
    deser_fun: str


@dataclass(frozen=True)
class DatasetTransmitPayload:
    # NOTE separated into two submessages so that sending over wire can be done as two frames
    header: DatasetTransmitPayloadHeader
    value: bytes


@dataclass(frozen=True)
class DatasetTransmitFailure:
    host: HostId
    detail: str


@dataclass(frozen=True)
class ExecutorFailure:
    host: HostId
    detail: str


@dataclass(frozen=True)
class ExecutorExit:
    host: HostId


@dataclass(frozen=True)
class Worker:
    # NOTE keep in sync with low.core.Worker
    worker_id: WorkerId
    cpu: int
    gpu: int
    memory_mb: int


@dataclass(frozen=True)
class ExecutorRegistration:
    host: HostId
    maddress: BackboneAddress
    daddress: BackboneAddress
    workers: list[Worker]


@dataclass(frozen=True)
class ExecutorShutdown:
    pass


@dataclass(frozen=True)
class WorkerReady:
    worker: WorkerId


@dataclass(frozen=True)
class WorkerShutdown:
    pass


# this explicit list is a disgrace -- see the _Message protocol above
Message = (
    Syn
    | Ack
    | TaskSequence
    | TaskFailure
    | DatasetPublished
    | DatasetPurge
    | DatasetTransmitCommand
    | DatasetTransmitPayload
    | ExecutorFailure
    | ExecutorExit
    | ExecutorRegistration
    | ExecutorShutdown
    | DatasetTransmitFailure
    | WorkerReady
    | WorkerShutdown
)
