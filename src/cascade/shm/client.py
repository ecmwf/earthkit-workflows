# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import multiprocessing.resource_tracker
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Type, TypeVar

import cascade.shm.api as api

logger = logging.getLogger(__name__)


class ConflictError(Exception):
    pass


if (sys.version_info.major, sys.version_info.minor) >= (3, 13):
    is_unregister = False
    shm_kwargs = {"track": False}
else:
    is_unregister = True
    shm_kwargs = {}


class AllocatedBuffer:
    def __init__(
        self,
        shmid: str,
        l: int,
        create: bool,
        close_callback: Callable[[], None] | None,
        deser_fun: str,
    ):
        self.shm: SharedMemory | None
        try:
            self.shm = SharedMemory(shmid, create=create, size=l, **shm_kwargs)
        except FileExistsError:
            # NOTE this is quite wrong as instead of crashing, it would lead to undefined behaviour
            # However, as the systems we operate on don't seem to be reliable wrt cleanup/isolation,
            # it ends up being a lesser evil
            logger.error(
                f"attempted opening {shmid=} but gotten FileExists. Will delete and retry"
            )
            _shm = SharedMemory(shmid, create=False, **shm_kwargs)
            _shm.close()
            _shm.unlink()
            self.shm = SharedMemory(shmid, create=create, size=l, **shm_kwargs)
        self.l = l
        self.readonly = not create
        self.close_callback = close_callback
        self.deser_fun = deser_fun
        if is_unregister:
            multiprocessing.resource_tracker.unregister(self.shm._name, "shared_memory")  # type: ignore # _name

    def view(self) -> memoryview:
        if not self.shm:
            raise ValueError("shm already closed!")
        mv = self.shm.buf[: self.l]
        if self.readonly:
            mv = mv.toreadonly()
        return mv

    def close(self) -> None:
        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            if self.close_callback:
                self.close_callback()
            self.shm = None

    def __del__(self) -> None:
        if hasattr(self, "shm") and self.shm is not None:
            try:
                logger.error(f"missed close() call on {self.shm._name}")  # type: ignore # _name
            except Exception as e:
                logger.exception(f"failed to log due to {repr(e)}")

    # TODO context manager


T = TypeVar("T", bound=api.Comm)


def _send_command(comm: api.Comm, resp_class: Type[T], timeout_sec: float = 60.0) -> T:
    timeout_i = 0.1
    coeff = 1
    # timeout_i and coeff determine rate of busy-waits: coeff=1 is additive, =2 is exponential
    # eventually this busy-waits will go away as we switch to event driven behaviour
    while timeout_sec > 0:
        sock = api.get_client_socket()
        logger.debug(f"sending message {comm}")
        sock.send(api.ser(comm))
        # TODO rewrite to poller with timeout
        response_raw = sock.recv(1024)  # TODO or recv(4) + recv(int.from_bytes)?
        sock.close()
        response_com = api.deser(response_raw)
        logger.debug(f"received response {response_com}")
        # NOTE we first check for presence of error, and only then for Type,
        # because a server error may change response class
        if hasattr(response_com, "error") and response_com.error:
            if response_com.error == "wait":
                logger.debug(f"gotten a wait, will sleep for {timeout_i}")
                time.sleep(timeout_i)
                timeout_sec -= timeout_i
                timeout_i *= coeff
                timeout_i = min(timeout_i, timeout_sec)
                continue
            elif response_com.error == "conflict":
                raise ConflictError
            raise ValueError(response_com.error)
        if not isinstance(response_com, resp_class):
            raise TypeError(type(response_com))
        return response_com
    raise TimeoutError


def close_callback(key: str, rdid: str) -> None:
    comm = api.CloseCallback(key=key, rdid=rdid)
    _send_command(comm, api.OkResponse)


def allocate(
    key: str, l: int, deser_fun: str, timeout_sec: float = 60.0
) -> AllocatedBuffer:
    comm = api.AllocateRequest(key=key, l=l, deser_fun=deser_fun)
    resp = _send_command(comm, api.AllocateResponse, timeout_sec)
    callback = lambda: close_callback(key, "")
    return AllocatedBuffer(
        shmid=resp.shmid, l=l, create=True, close_callback=callback, deser_fun=deser_fun
    )


def get(key: str, timeout_sec: float = 60.0) -> AllocatedBuffer:
    comm = api.GetRequest(key=key)
    resp = _send_command(comm, api.GetResponse, timeout_sec)
    callback = lambda: close_callback(key, resp.rdid)
    return AllocatedBuffer(
        shmid=resp.shmid,
        l=resp.l,
        deser_fun=resp.deser_fun,
        create=False,
        close_callback=callback,
    )


def purge(key: str) -> None:
    comm = api.PurgeRequest(key=key)
    _ = _send_command(comm, api.OkResponse)


def status(key: str) -> api.DatasetStatus:
    comm = api.DatasetStatusRequest(key=key)
    response = _send_command(comm, api.DatasetStatusResponse)
    return response.status


def shutdown(timeout_sec: float = 2.0) -> None:
    comm = api.ShutdownCommand()
    _send_command(comm, api.OkResponse, timeout_sec)


def ensure() -> None:
    """Loop StatusInquiry until shm server responds with Ok"""
    logger.debug("entering shm ensure loop")
    comm = api.StatusInquiry()
    while True:
        try:
            _send_command(comm, api.OkResponse)
            logger.debug("shm server responds ok, leaving ensure loop")
        except (ConnectionRefusedError, FileNotFoundError):
            time.sleep(0.1)
            continue
        break


def get_free_space() -> int:
    comm = api.FreeSpaceRequest()
    resp = _send_command(comm, api.FreeSpaceResponse)
    return resp.free_space
