"""
This module handles basic communication structures and functions
"""

import threading
import logging

import zmq

from cascade.executor.msg import BackboneAddress, Message, DatasetTransmitCommand, DatasetTransmitPayload
from cascade.executor.serde import ser_message, des_message, ser_dmessage, des_dmessage

logger = logging.getLogger(__name__)

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

    def recv_dmessage(self) -> DatasetTransmitCommand|DatasetTransmitPayload:
        m = self.socket.recv_multipart()
        return des_dmessage(m)
