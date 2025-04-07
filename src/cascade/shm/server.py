# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import logging.config
import signal
import socket
from typing import Any

import cascade.shm.api as api
import cascade.shm.dataset as dataset

logger = logging.getLogger(__name__)


class LocalServer:
    """Handles the socket communication, and the invocation of dataset.Manager which has the business logic"""

    def __init__(self, port: int, shm_pref: str, capacity: int | None = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server = ("0.0.0.0", port)
        self.sock.bind(server)
        self.manager = dataset.Manager(shm_pref, capacity)
        signal.signal(signal.SIGINT, self.atexit)
        signal.signal(signal.SIGTERM, self.atexit)

    def receive(self) -> tuple[api.Comm, str]:
        b, addr = self.sock.recvfrom(1024)  # TODO or recv(4) + recv(int.from_bytes)?
        return api.deser(b), addr

    def respond(self, comm: api.Comm, address: str) -> None:
        self.sock.sendto(api.ser(comm), address)

    def atexit(self, signum: int, frame: Any) -> None:
        self.manager.atexit()
        self.sock.close()

    def start(self):
        while True:
            payload, client = self.receive()
            try:
                if isinstance(payload, api.AllocateRequest):
                    shmid, error = self.manager.add(
                        payload.key, payload.l, payload.deser_fun
                    )
                    if error:
                        response = api.AllocateResponse(shmid="", error=error)
                    else:
                        response = api.AllocateResponse(shmid=shmid, error="")
                elif isinstance(payload, api.CloseCallback):
                    self.manager.close_callback(payload.key, payload.rdid)
                    response = api.OkResponse()
                elif isinstance(payload, api.GetRequest):
                    shmid, l, rdid, deser_fun, error = self.manager.get(payload.key)
                    if error:
                        response = api.GetResponse(
                            shmid="", l=0, rdid=rdid, deser_fun=deser_fun, error=error
                        )
                    else:
                        response = api.GetResponse(
                            shmid=shmid, l=l, rdid=rdid, deser_fun=deser_fun, error=""
                        )
                elif isinstance(payload, api.ShutdownCommand):
                    response = api.OkResponse()
                    self.respond(response, client)
                    break
                elif isinstance(payload, api.StatusInquiry):
                    response = api.OkResponse()
                elif isinstance(payload, api.FreeSpaceRequest):
                    free_space = self.manager.free_space
                    response = api.FreeSpaceResponse(free_space=free_space)
                elif isinstance(payload, api.PurgeRequest):
                    self.manager.purge(payload.key)
                    response = api.OkResponse()
                elif isinstance(payload, api.DatasetStatusRequest):
                    ds = self.manager.datasets.get(payload.key, None)
                    if not ds:
                        status = api.DatasetStatus.not_present
                    elif ds.status == dataset.DatasetStatus.created:
                        status = api.DatasetStatus.not_present
                    elif ds.status in (
                        dataset.DatasetStatus.in_memory,
                        dataset.DatasetStatus.paging_out,
                        dataset.DatasetStatus.on_disk,
                        dataset.DatasetStatus.paged_in,
                    ):
                        status = api.DatasetStatus.ready
                    response = api.DatasetStatusResponse(status=status)
                else:
                    raise ValueError(f"unsupported: {type(payload)}")
            except Exception as e:
                logger.exception(f"failure during handling of {payload}")
                response = api.OkResponse(error=repr(e))
            self.respond(response, client)


def entrypoint(
    port: int,
    capacity: int | None = None,
    logging_config: dict | None = None,
    shm_pref: str = "shm",
):
    if logging_config:
        logging.config.dictConfig(logging_config)
    server = LocalServer(port, shm_pref, capacity)
    logger.info(
        f"shm server starting on {port=} with {capacity=} and prefix {shm_pref}"
    )
    try:
        server.start()
    except Exception as e:
        # we always get a Bad file descriptor due to sigterm handler calling sock close mid-read
        logger.warning(f"shutdown issue: {e}")
    server.atexit(0, None)
