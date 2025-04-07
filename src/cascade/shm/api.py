# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Type, runtime_checkable

from typing_extensions import Self

# TODO too much manual serde... either automate it based on dataclass field inspection, or just pickle it
# (mind the server.recv/client.recv comment tho)
# Also, consider switching from GetRequest, PurgeRequest, to DatasetRequest(get|purge|...)


def ser_str(s: str) -> bytes:
    return len(s).to_bytes(4, "big") + s.encode("ascii")


def deser_str(b: memoryview) -> tuple[str, memoryview]:
    l = int.from_bytes(b[:4], "big")
    return str(b[4 : 4 + l], "ascii"), b[4 + l :]


@runtime_checkable
class Comm(Protocol):
    def ser(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        raise NotImplementedError


@dataclass(frozen=True)
class GetRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


@dataclass(frozen=True)
class PurgeRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


@dataclass(frozen=True)
class DatasetStatusRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


class DatasetStatus(int, Enum):
    not_ready = auto()
    ready = auto()
    not_present = auto()


@dataclass(frozen=True)
class DatasetStatusResponse:
    status: DatasetStatus

    def ser(self) -> bytes:
        return self.status.value.to_bytes(4, "big")

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        status, _ = DatasetStatus(int.from_bytes(data[:4], "big")), data[4:]
        return cls(status=status)


@dataclass(frozen=True)
class GetResponse:
    shmid: str
    l: int
    rdid: str
    error: str
    deser_fun: str

    def ser(self) -> bytes:
        return (
            self.l.to_bytes(4, "big")
            + ser_str(self.deser_fun)
            + ser_str(self.shmid)
            + ser_str(self.rdid)
            + ser_str(self.error)
        )

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        l, data = int.from_bytes(data[:4], "big"), data[4:]
        deser_fun, data = deser_str(data)
        shmid, data = deser_str(data)
        rdid, data = deser_str(data)
        error, _ = deser_str(data)
        return cls(l=l, shmid=shmid, rdid=rdid, error=error, deser_fun=deser_fun)


@dataclass(frozen=True)
class AllocateRequest:
    key: str
    l: int
    deser_fun: str

    def ser(self) -> bytes:
        return self.l.to_bytes(8, "big") + ser_str(self.deser_fun) + ser_str(self.key)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        l, data = int.from_bytes(data[:8], "big"), data[8:]
        deser_fun, data = deser_str(data)
        key, _ = deser_str(data)
        return cls(l=l, key=key, deser_fun=deser_fun)


@dataclass(frozen=True)
class AllocateResponse:
    shmid: str
    error: str

    def ser(self) -> bytes:
        return ser_str(self.shmid) + ser_str(self.error)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        shmid, data = deser_str(data)
        error, _ = deser_str(data)
        return cls(shmid=shmid, error=error)


@dataclass(frozen=True)
class CloseCallback:
    key: str
    rdid: str

    def ser(self) -> bytes:
        return ser_str(self.key) + ser_str(self.rdid)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        key, data = deser_str(data)
        rdid, _ = deser_str(data)
        return cls(key=key, rdid=rdid)


class EmptyCommand:

    def ser(self) -> bytes:
        return b""

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        return cls()


class ShutdownCommand(EmptyCommand):
    pass


class StatusInquiry(EmptyCommand):
    pass


class FreeSpaceRequest(EmptyCommand):
    pass


@dataclass(frozen=True)
class OkResponse:
    error: str = ""

    def ser(self) -> bytes:
        return ser_str(self.error)

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        error, _ = deser_str(data)
        return cls(error=error)


@dataclass(frozen=True)
class FreeSpaceResponse:
    free_space: int

    def ser(self) -> bytes:
        return self.free_space.to_bytes(4, "big")

    @classmethod
    def deser(cls, data: memoryview) -> Self:
        free_space = int.from_bytes(data[:4], "big")
        return cls(free_space=free_space)


b2c: dict[bytes, Type[Comm]] = {
    b"\x01": GetRequest,
    b"\x02": GetResponse,
    b"\x03": AllocateRequest,
    b"\x04": AllocateResponse,
    b"\x05": ShutdownCommand,
    b"\x06": StatusInquiry,
    b"\x07": FreeSpaceRequest,
    b"\x08": FreeSpaceResponse,
    b"\x09": OkResponse,
    b"\x0a": CloseCallback,
    b"\x0b": PurgeRequest,
    b"\x0c": DatasetStatusRequest,
}
c2b: dict[Type[Comm], bytes] = {v: k for k, v in b2c.items()}


def ser(comm: Comm) -> bytes:
    m = c2b[type(comm)] + comm.ser()
    return m


def deser(data: bytes) -> Comm:
    data = memoryview(data)
    return b2c[data[:1]].deser(data[1:])


client_port_envvar = "CASCADE_SHM_PORT"


def publish_client_port(port: int) -> None:
    os.environ[client_port_envvar] = str(port)


def get_client_port() -> int:
    port = os.getenv(client_port_envvar)
    if not port:
        raise ValueError("missing port")
    return int(port)
