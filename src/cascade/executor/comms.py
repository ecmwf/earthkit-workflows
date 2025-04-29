# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module handles basic communication structures and functions"""

import logging
import pickle
import threading
import time
from dataclasses import dataclass

import zmq

from cascade.executor.msg import (
    Ack,
    BackboneAddress,
    DatasetTransmitPayload,
    DatasetTransmitPayloadHeader,
    Message,
    Syn,
)
from cascade.executor.serde import des_message, ser_message
from cascade.low.core import HostId

logger = logging.getLogger(__name__)
default_timeout_ms = 1_000
default_message_resend_ms = 800


class GraceWatcher:
    """For watching whether certain event occurred more than `grace_ms` ago"""

    def __init__(self, grace_ms: int):
        self.step_time_ms = 0
        self.log_time_ms = 0
        self.grace_ms = grace_ms

    def _now(self) -> int:
        return int(time.time_ns() / 1_000_000)

    def step(self) -> None:
        """Notify that event has occurred recently"""
        self.step_time_ms = self._now()

    def is_breach(self) -> int:
        """If the last `step()` occurred less than `grace_ms` ago, returns = 0, otherwise > 0.
        If the last return of > 0 occurred more than `grace_ms` ago, return 2, otherwise 1.
        The 2-vs-1 should be used for rate limiting logs, whereas business logic should heed
        0-vs-non0
        """
        now = self._now()
        breachStep = self._now() > self.step_time_ms + self.grace_ms
        breachLog = self._now() > self.log_time_ms + self.grace_ms
        if breachLog:
            self.log_time_ms = now
            return 1
        elif breachStep:
            return 2
        else:
            return 0

    def elapsed_ms(self) -> int:
        """How many ms elapsed since last `step()`"""
        return self._now() - self.step_time_ms


def get_context() -> zmq.Context:
    local = threading.local()
    if not hasattr(local, "context"):
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
    # NOTE should be used for local comms only, as does not prepend with Syn message
    # TODO enforce that -- but needs a refactor for executor to open more sockets & poll correctly
    socket = get_socket(address)
    byt = ser_message(msg)
    socket.send(byt)


def send_data(address: BackboneAddress, data: DatasetTransmitPayload, syn: Syn) -> None:
    socket = get_socket(address)
    byt = (ser_message(syn), pickle.dumps(data.header), data.value)
    socket.send_multipart(byt)


class Listener:
    def __init__(self, address: BackboneAddress):
        self.address = address
        self.socket = get_context().socket(zmq.PULL)
        self.socket.bind(address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, flags=zmq.POLLIN)
        self.acked: set[Syn] = (
            set()
        )  # TODO eventually pop things from here (timestamp lapse?) to prevent mem growth

    def _recv_one(self, timeout_ms: int | None) -> Message | None:
        ready = self.poller.poll(timeout_ms if timeout_ms is not None else None)
        if len(ready) > 1:
            raise ValueError(f"unexpected number of socket events: {len(ready)}")
        if not ready:
            return None
        else:
            # TODO move parts of this to serde, including corresponding `send_data` and `ReliableSender.send` parts
            data = ready[0][0].recv_multipart()
            if len(data) == 0:
                raise ValueError("unexpected empty message")
            m0 = des_message(data[0])
            if isinstance(m0, Syn):
                callback(m0.addr, Ack(idx=m0.idx))
                if len(data) == 1:
                    raise ValueError("unexpected message with Syn only")
                if m0 in self.acked:
                    logger.warning(
                        f"received already-acked {m0}, assuming retry, dropping message"
                    )
                    return None
                else:
                    self.acked.add(m0)
            elif isinstance(m0, DatasetTransmitPayloadHeader):
                if len(data) != 2:
                    raise ValueError(
                        f"first message was payload header, but {len(data)=} != 2"
                    )
                else:
                    return DatasetTransmitPayload(header=m0, value=data[1])
            else:
                if len(data) != 1:
                    raise ValueError(f"expected len 1 but gotten {len(data)}")
                return m0
            m1 = des_message(data[1])
            if isinstance(m1, Syn):
                raise ValueError("unexpected double Syn")
            elif isinstance(m1, DatasetTransmitPayloadHeader):
                if len(data) != 3:
                    raise ValueError(
                        f"second message was payload header, but {len(data)=} != 3"
                    )
                else:
                    return DatasetTransmitPayload(header=m1, value=data[2])
            else:
                if len(data) != 2:
                    raise ValueError(f"expected {len(data)=} to equal 2")
                return m1

    def recv_messages(
        self, timeout_ms: int | None = default_timeout_ms
    ) -> list[Message]:
        messages: list[Message] = []
        # logger.debug(f"receiving messages on {self.address} with {timeout_sec=}")
        message = self._recv_one(timeout_ms)
        if message is not None:
            messages.append(message)
            while True:
                message = self._recv_one(0)
                if message is None:
                    break
                else:
                    messages.append(message)
        return messages


@dataclass
class _InFlightRecord:
    host: HostId
    message: tuple[bytes, bytes]
    clazz: str
    at: int
    remaining: int


max_retries_per_message = 20


class ReliableSender:
    def __init__(self, address: BackboneAddress, resend_grace_ms: int) -> None:
        self.hosts: dict[HostId, tuple[zmq.Socket, BackboneAddress]] = {}
        self.inflight: dict[int, _InFlightRecord] = {}
        self.idx = 0
        self.resend_grace = resend_grace_ms * 1_000_000
        self.address = address

    def add_host(self, host: HostId, address: BackboneAddress) -> None:
        self.hosts[host] = (get_socket(address), address)

    def send(self, host: HostId, m: Message) -> None:
        raw = ser_message(m)
        syn = ser_message(Syn(idx=self.idx, addr=self.address))
        self.inflight[self.idx] = _InFlightRecord(
            host=host,
            message=(syn, raw),
            at=time.time_ns(),
            clazz=type(m).__name__,
            remaining=max_retries_per_message,
        )
        self.hosts[host][0].send_multipart((syn, raw))
        self.idx += 1

    def ack(self, idx: int) -> None:
        if idx in self.inflight:
            self.inflight.pop(idx)
        else:
            logger.warning(f"repeated ack of {idx=}, assuming repeated syn")
            # NOTE probably not worth checking syn counter, but we could

    def maybe_retry(self) -> None:
        watermark = time.time_ns() - self.resend_grace
        for idx, record in self.inflight.items():
            if record.at < watermark:
                logger.warning(
                    f"retrying message {idx} ({record.clazz}) due to no confirmation after {watermark-record.at}ns"
                )
                if (socket := self.hosts.get(record.host, None)) is not None:
                    socket[0].send_multipart(record.message)
                    self.inflight[idx].at = time.time_ns()
                    self.inflight[idx].remaining -= 1
                    if self.inflight[idx].remaining <= 0:
                        raise ValueError(
                            f"message {idx} ({record.clazz}) retried too many times"
                        )
                else:
                    logger.warning(
                        f"{record.host=} not present, cannot retry message {idx=}. Presumably we are at shutdown"
                    )
