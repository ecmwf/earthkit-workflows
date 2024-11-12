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

class ZmqBackbone:
    def __init__(self, addr: str, controller_url: str|None=None, host_id: str|None = None, environment: Environment|None=None, expected_workers: int|None=None) -> None:
        """For the controller's backbone, provide no url but a count for workers; for
        the worker's backbone, the other way around. The init blocks until comms is
        established"""
        context = zmq.Context()
        self.m_socket = context.socket(zmq.PULL)
        self.addr = addr
        self.m_socket.bind(addr)
        self.controller_url = controller_url

        if controller_url is not None and environment is not None and host_id is not None:
            self.s_sockets = {'controller': context.socket(zmq.PUSH)}
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
                self.s_sockets[m.host_id] = context.socket(zmq.PUSH)
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
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
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
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect(url)
        b = serialize(data)
        socket.send(b)

    def _recv_one(self, block: bool) -> Message:
        flags = 0 if block else zmq.NOBLOCK
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
            except zmq.ZMQError:
                # NOTE check code etc for confirm its due to no messages avail
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
