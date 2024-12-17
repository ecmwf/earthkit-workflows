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


def deser_str(b: bytes) -> tuple[str, bytes]:
    l = int.from_bytes(b[:4], "big")
    return b[4 : 4 + l].decode("ascii"), b[4 + l :]


@runtime_checkable
class Comm(Protocol):
    def ser(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def deser(cls, data: bytes) -> Self:
        raise NotImplementedError


@dataclass
class GetRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


@dataclass
class PurgeRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


@dataclass
class DatasetStatusRequest:
    key: str

    def ser(self) -> bytes:
        return ser_str(self.key)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        key, _ = deser_str(data)
        return cls(key=key)


class DatasetStatus(int, Enum):
    not_ready = auto()
    ready = auto()
    not_present = auto()


@dataclass
class DatasetStatusResponse:
    status: DatasetStatus

    def ser(self) -> bytes:
        return self.status.value.to_bytes(4, "big")

    @classmethod
    def deser(cls, data: bytes) -> Self:
        status, _ = DatasetStatus(int.from_bytes(data[:4], "big")), data[4:]
        return cls(status=status)


@dataclass
class GetResponse:
    shmid: str
    l: int
    rdid: str
    error: str

    def ser(self) -> bytes:
        return (
            self.l.to_bytes(4, "big")
            + ser_str(self.shmid)
            + ser_str(self.rdid)
            + ser_str(self.error)
        )

    @classmethod
    def deser(cls, data: bytes) -> Self:
        l, data = int.from_bytes(data[:4], "big"), data[4:]
        shmid, data = deser_str(data)
        rdid, data = deser_str(data)
        error, _ = deser_str(data)
        return cls(l=l, shmid=shmid, rdid=rdid, error=error)


@dataclass
class AllocateRequest:
    key: str
    l: int

    def ser(self) -> bytes:
        return self.l.to_bytes(8, "big") + ser_str(self.key)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        l, data = int.from_bytes(data[:8], "big"), data[8:]
        key, _ = deser_str(data)
        return cls(l=l, key=key)


@dataclass
class AllocateResponse:
    shmid: str
    error: str

    def ser(self) -> bytes:
        return ser_str(self.shmid) + ser_str(self.error)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        shmid, data = deser_str(data)
        error, _ = deser_str(data)
        return cls(shmid=shmid, error=error)


@dataclass
class CloseCallback:
    key: str
    rdid: str

    def ser(self) -> bytes:
        return ser_str(self.key) + ser_str(self.rdid)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        key, data = deser_str(data)
        rdid, _ = deser_str(data)
        return cls(key=key, rdid=rdid)


class EmptyCommand:

    def ser(self) -> bytes:
        return b""

    @classmethod
    def deser(cls, data: bytes) -> Self:
        return cls()


class ShutdownCommand(EmptyCommand):
    pass


class StatusInquiry(EmptyCommand):
    pass


class FreeSpaceRequest(EmptyCommand):
    pass


@dataclass
class OkResponse:
    error: str = ""

    def ser(self) -> bytes:
        return ser_str(self.error)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        error, _ = deser_str(data)
        return cls(error=error)


@dataclass
class FreeSpaceResponse:
    free_space: int

    def ser(self) -> bytes:
        return self.free_space.to_bytes(4, "big")

    @classmethod
    def deser(cls, data: bytes) -> Self:
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
    return b2c[data[:1]].deser(data[1:])


client_port_envvar = "CASCADE_SHM_PORT"


def publish_client_port(port: int) -> None:
    os.environ[client_port_envvar] = str(port)


def get_client_port() -> int:
    port = os.getenv(client_port_envvar)
    if not port:
        raise ValueError("missing port")
    return int(port)
