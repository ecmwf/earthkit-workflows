"""
This module handles basic communication structures and functions
"""

import threading

import zmq

from cascade.executor.msg import BackboneAddress, Message
from cascade.executor.serde import ser, des

def callback(address: BackboneAddress, msg: Message):
    local = threading.local()
    if not hasattr(local, 'context'):
        local.context = zmq.Context()
    socket = local.context.socket(zmq.PUSH) 
    # TODO check we dont need linger -- we assume this to be local comms only
    # socket.set(zmq.LINGER, 1000)
    socket.connect(address)
    byt = ser(msg)
    socket.send(byt)
    
