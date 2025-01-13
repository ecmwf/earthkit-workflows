"""
This module handles basic communication structures and functions
"""

import threading
import logging
import time

import zmq

from cascade.executor.msg import BackboneAddress, Message, DatasetTransmitCommand, DatasetTransmitPayload
from cascade.executor.serde import ser_message, des_message, ser_dmessage, des_dmessage

logger = logging.getLogger(__name__)
default_timeout_sec = 2

class GraceWatcher:
    """For watching whether certain event occurred more than `grace_ms` ago"""
    def __init__(self, grace_ms: int):
        self.time_ms = 0
        self.grace_ms = grace_ms

    def _now(self) -> int:
        return int(time.time_ns() / 1_000_000)

    def step(self) -> None:
        """Notify that event has occurred recently"""
        self.time_ms = self._now()

    def is_breach(self) -> bool:
        """Has last `step()` occurred more than `grace_ms` ago?"""
        if self._now() > self.time_ms + self.grace_ms:
            return True
        else:
            return False

    def elapsed_ms(self) -> int:
        """How many ms elapsed since last `step()`"""
        return self._now() - self.time_ms

def get_context() -> zmq.Context:
    local = threading.local()
    if not hasattr(local, 'context'):
        local.context = zmq.Context()
    return local.context

def get_socket(address: BackboneAddress) -> zmq.Socket:
    socket = get_context().socket(zmq.PUSH) 
    # NOTE we set the linger in case the executor dies before consuming a message sent
    # by the child -- otherwise the child process would hang indefinitely
    socket.set(zmq.LINGER, 1000)
    socket.connect(address)
    return socket

def callback(address: BackboneAddress, msg: Message):
    socket = get_socket(address)
    byt = ser_message(msg)
    socket.send(byt)

def send_data(address: BackboneAddress, data: DatasetTransmitPayload):
    socket = get_socket(address)
    byt = ser_dmessage(data)
    socket.send_multipart(byt)

class Listener:
    def __init__(self, address: BackboneAddress):
        self.address = address
        self.socket = get_context().socket(zmq.PULL)
        self.socket.bind(address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, flags=zmq.POLLIN)
    
    def _recv_one(self, timeout_sec: int|None) -> Message|None:
        ready = self.poller.poll(timeout_sec * 1_000 if timeout_sec is not None else None)
        if len(ready) > 1:
            raise ValueError(f"unexpected number of socket events: {len(ready)}")
        if not ready:
            return None
        else:
            data = ready[0][0].recv()
            return des_message(data)

    def recv_messages(self, timeout_sec: int|None = default_timeout_sec) -> list[Message]:
        messages: list[Message] = []
        logger.debug(f"receiving messages on {self.address} with {timeout_sec=}")
        message = self._recv_one(timeout_sec)
        if message is not None:
            messages.append(message)
            while True:
                message = self._recv_one(0)
                if message is None:
                    break
                else:
                    messages.append(message)
        return messages

    def recv_dmessage(self, timeout_sec: int|None = default_timeout_sec) -> DatasetTransmitCommand|DatasetTransmitPayload|None:
        logger.debug(f"receiving data on {self.address} with {timeout_sec=}")
        ready = self.poller.poll(timeout_sec * 1_000 if timeout_sec is not None else None)
        if len(ready) > 1:
            raise ValueError(f"unexpected number of socket events: {len(ready)}")
        if not ready:
            return None
        else:
            m = ready[0][0].recv_multipart()
            return des_dmessage(m)
