"""
This module handles basic communication structures and functions
"""

# TODO introduce something to make *everything* reliable/retriable. `executor.data_server` contains
# a skeleton thereof with DatasetConfirmation, but we either want to go the explicit way and
# develop a similar track-retry in `controller.act`, *or* rework it on a general level here in `comms`.
# The difference between the two is for example when a TaskSequence is sent to worker -- it may happen
# that a confirmation would be lost, but we don't really care as long as it gets computed. On the other
# hand, if the initial command gets lost, it would take longer to recover from it

from dataclasses import dataclass
import threading
import logging
import time

import zmq

from cascade.low.core import HostId
from cascade.executor.msg import BackboneAddress, Message, DatasetTransmitCommand, DatasetTransmitPayload, DatasetTransmitConfirm, Syn, Ack
from cascade.executor.serde import ser_message, des_message, ser_dmessage, des_dmessage

logger = logging.getLogger(__name__)
default_timeout_sec = 5

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
            data = ready[0][0].recv_multipart()
            if len(data) == 1:
                return des_message(data[0])
            elif len(data) == 2:
                m1 = des_message(data[0])
                if isinstance(m1, Syn):
                    callback(m1.addr, Ack(idx=m1.idx))
                else:
                    raise NotImplementedError(f"expected Syn but gotten {type(m1)}")
                return des_message(data[1])
            else:
                raise NotImplementedError(f"unsupported multipart message length: {len(data)}")

    def recv_messages(self, timeout_sec: int|None = default_timeout_sec) -> list[Message]:
        messages: list[Message] = []
        # logger.debug(f"receiving messages on {self.address} with {timeout_sec=}")
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

    def recv_dmessage(self, timeout_sec: int|None = default_timeout_sec) -> DatasetTransmitCommand|DatasetTransmitPayload|DatasetTransmitConfirm|None:
        # logger.debug(f"receiving data on {self.address} with {timeout_sec=}")
        ready = self.poller.poll(timeout_sec * 1_000 if timeout_sec is not None else None)
        if len(ready) > 1:
            raise ValueError(f"unexpected number of socket events: {len(ready)}")
        if not ready:
            return None
        else:
            m = ready[0][0].recv_multipart()
            return des_dmessage(m)

@dataclass
class _InFlightRecord:
    host: HostId
    message: tuple[bytes, bytes]
    at: int

class ReliableSender():
    def __init__(self, maddress: BackboneAddress) -> None:
        self.hosts: dict[HostId, zmq.Socket] = {}
        self.inflight: dict[int, _InFlightRecord] = {}
        self.idx = 0
        self.resend_grace = 2 * 1_000_000_000 # two seconds
        self.maddress = maddress

    def add_host(self, host: HostId, socket: zmq.Socket) -> None:
        self.hosts[host] = socket

    def send(self, host: HostId, m: Message) -> None:
        raw = ser_message(m)
        syn = ser_message(Syn(idx=self.idx, addr=self.maddress))
        self.inflight[self.idx] = _InFlightRecord(host=host, message=(syn, raw), at=time.time_ns())
        self.hosts[host].send_multipart((syn, raw))
        self.idx += 1

    def ack(self, idx: int) -> None:
        if idx in self.inflight:
            self.inflight.pop(idx)
        # NOTE double pop would mean a second ack arriving, presumably after we syn'd twice -- checking prolly not worth it
        
    def maybe_retry(self) -> None:
        watermark = time.time_ns() - self.resend_grace
        for idx, record in self.inflight.items():
            if record.at < watermark:
                logger.warning(f"retrying message {idx} due to not having it confirmed after {watermark-record.at}ns")
                if (socket := self.hosts.get(record.host, None)) is not None:
                    socket.send_multipart(record.message)
                    self.inflight[idx].at = time.time_ns()
                else:
                    logger.warning(f"{record.host=} not present, cannot retry message {idx=}. Presumably we are at shutdown")
            
