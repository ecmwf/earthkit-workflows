"""
This module handles basic communication structures and functions
"""

import threading
import logging

import zmq

from cascade.executor.msg import BackboneAddress, Message
from cascade.executor.serde import ser_message, des_message

logger = logging.getLogger(__name__)

def get_context() -> zmq.Context:
    local = threading.local()
    if not hasattr(local, 'context'):
        local.context = zmq.Context()
    return local.context

def callback(address: BackboneAddress, msg: Message):
    socket = get_context().socket(zmq.PUSH) 
    # TODO check we dont need linger -- we assume this to be local comms only
    # socket.set(zmq.LINGER, 1000)
    socket.connect(address)
    byt = ser_message(msg)
    socket.send(byt)

class Listener:
    def __init__(self, address: BackboneAddress):
        self.address = address
        self.socket = get_context().socket(zmq.PULL)
        self.socket.bind(address)
    
    def _recv_one(self, block: bool) -> Message:
        flags = 0 if block else zmq.DONTWAIT
        return des_message(self.socket.recv(flags=flags))

    def recv_messages(self) -> list[Message]:
        messages: list[Message] = []
        logger.debug(f"receiving messages on {self.address}")
        message = self._recv_one(True)
        messages.append(message)
        while True:
            try:
                message = self._recv_one(False)
                messages.append(message)
            except zmq.Again:
                break
        return messages

