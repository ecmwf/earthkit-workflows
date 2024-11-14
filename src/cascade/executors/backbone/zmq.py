"""
ZMQ-based implementation of backbone

Partially thread safe -- the only disallowed is to call recv from multiple threads simultaneously,
as well as send_message to the *same* host from multiple threads simultaneously. Using callbacks
is completely thread safe.
"""

import logging
import threading
from cascade.executors.backbone.serde import Message, serialize, deserialize, RegisterRequest, RegisterResponse, Shutdown, DataTransmitObject
from cascade.executors.backbone.interface import merge_environments, HostId
from cascade.low.core import Environment
from cascade.controller.core import Event
from typing import Callable
import zmq

logger = logging.getLogger(__name__)

# TODO split in common abc and server/client classes
# NOTE consider always bind in server and always connect in client:
# - the event queue would be bind.PULL and connect.PUSH
# - the command queue would be bind.PUB and connect.SUB, with the first bytes (sub filter) being the hostId _and_ broadcast

class ZmqBackbone:
    def __init__(self, addr: str, controller_url: str|None=None, host_id: str|None = None, environment: Environment|None=None, expected_workers: int|None=None) -> None:
        """For the controller's backbone, provide no url but a count for workers; for
        the worker's backbone, the other way around. The init blocks until comms is
        established"""
        self.context = zmq.Context()
        self.m_socket = self.context.socket(zmq.PULL)
        self.addr = addr
        self.m_socket.bind(addr)
        self.controller_url = controller_url

        if controller_url is not None and environment is not None and host_id is not None:
            self.s_sockets = {'controller': self.context.socket(zmq.PUSH)}
            self.s_sockets['controller'].connect(controller_url)
            self.send_message('controller', RegisterRequest(url=addr, environment=environment, host_id = host_id))
            resp = self._recv_one(True)
            if not isinstance(resp, RegisterResponse):
                raise ValueError(f"unexpected register response: {resp}")
            logger.debug(f"finished backbone init for host {host_id} at {addr}")
            return

        if expected_workers is None:
            raise TypeError("unexpected combination of init params")

        envs: list[Environment] = []
        self.s_sockets = {}
        self.urls = {}
        while expected_workers > 0:
            for m in self.recv_messages():
                if not isinstance(m, RegisterRequest):
                    raise TypeError(m)
                expected_workers -= 1
                self.s_sockets[m.host_id] = self.context.socket(zmq.PUSH)
                self.s_sockets[m.host_id].connect(m.url)
                self.urls[m.host_id] = m.url
                self.send_message(m.host_id, RegisterResponse())
                envs.append(m.environment)

        self.environment = merge_environments(envs)
        logger.debug(f"finished backbone init for controller at {addr}")
        return

    def send_message(self, host: HostId, message: Message) -> None:
        b = serialize(message)
        s = self.s_sockets[host]
        s.send(b)

    def send_event_callback(self) -> Callable[[Event], None]:
        if self.controller_url is None:
            raise TypeError
        controller_url = self.controller_url
        addr = self.addr

        def callback(event: Event):
            # we need a new context because we are in different process
            # generally we expect one invocation of callback per process -- if this is to change,
            # consider storing the context in the thread local; otherwise we have extra io thread
            context = zmq.Context() 
            socket = context.socket(zmq.PUSH)
            # TODO this should instead report to the local backbone only, because when the controller
            # is unreachable, we want the backbone process to hold on to the controller report, not the
            # task's process. If the linger below would be None, we hang infinitely in case the controller
            # has sent a shutdown prior
            socket.set(zmq.LINGER, 1000)
            socket.connect(controller_url)
            b = serialize(event)
            logger.debug(f"callback {event} send controller")
            socket.send(b)
            socket = context.socket(zmq.PUSH)
            socket.connect(addr)
            logger.debug(f"callback {event} send locally")
            socket.send(b)
            logger.debug(f"callback {event} done")

        return callback

    def send_data(self, url: str, data: DataTransmitObject) -> None:
        socket = self.context.socket(zmq.PUSH)
        socket.connect(url)
        b = serialize(data)
        # NOTE consider multipart?
        socket.send(b)

    def _recv_one(self, block: bool) -> Message:
        flags = 0 if block else zmq.DONTWAIT
        return deserialize(self.m_socket.recv(flags=flags))

    def recv_messages(self) -> list[Message]:
        messages: list[Message] = []
        logger.debug(f"receiving messages")
        message = self._recv_one(True)
        messages.append(message)
        while True:
            try:
                message = self._recv_one(False)
                messages.append(message)
            except zmq.Again:
                break
        return messages

    def broadcast_shutdown(self) -> None:
        for host_id in self.s_sockets:
            logger.debug(f"shutting down {host_id}")
            self.send_message(host_id, Shutdown())

    def get_environment(self) -> Environment:
        return self.environment

    def url_of(self, host: HostId) -> str:
        return self.urls[host]

    def shutdown(self) -> None:
        self.context.destroy()
