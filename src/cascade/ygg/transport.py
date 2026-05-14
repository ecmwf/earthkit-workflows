# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import threading
from typing import Any

import zmq

from cascade.low.exceptions import CascadeInternalError
from cascade.ygg.types import Lane

logger = logging.getLogger(__name__)

_local = threading.local()


def get_context() -> zmq.Context:
    if not hasattr(_local, "context"):
        _local.context = zmq.Context.instance()
    return _local.context


class OutboundTransport:
    def __init__(self, linger_ms: int) -> None:
        self._linger_ms = linger_ms
        self._sockets: dict[str, zmq.Socket] = {}

    def _socket_for(self, address: str) -> zmq.Socket:
        socket = self._sockets.get(address)
        if socket is not None:
            return socket
        socket = get_context().socket(zmq.PUSH)
        socket.set(zmq.LINGER, self._linger_ms)
        socket.connect(address)
        self._sockets[address] = socket
        return socket

    def describe_state(self) -> dict[str, Any]:
        return {address: _describe_socket(socket) for address, socket in self._sockets.items()}

    def send_multipart(self, address: str, frames: tuple[bytes, ...], copy: bool = True) -> None:
        self._socket_for(address).send_multipart(frames, copy=copy)

    def send_single(self, address: str, frame: bytes) -> None:
        self._socket_for(address).send(frame)

    def close(self) -> None:
        for socket in self._sockets.values():
            socket.close()
        self._sockets.clear()


class MultiLaneListener:
    def __init__(self, bind_addresses: dict[Lane, str], linger_ms: int) -> None:
        self._poller = zmq.Poller()
        self._socket_by_lane: dict[Lane, zmq.Socket] = {}
        self._lane_by_socket_id: dict[int, Lane] = {}
        self._addresses: dict[Lane, str] = {}

        for lane, bind_address in bind_addresses.items():
            socket = get_context().socket(zmq.PULL)
            socket.set(zmq.LINGER, linger_ms)
            address = self._bind(socket, bind_address)
            self._socket_by_lane[lane] = socket
            self._lane_by_socket_id[id(socket)] = lane
            self._addresses[lane] = address
            self._poller.register(socket, flags=zmq.POLLIN)

    def describe_state(self) -> dict[str, Any]:
        return {lane: _describe_socket(socket) for lane, socket in self._socket_by_lane.items()}

    def _bind(self, socket: zmq.Socket, bind_address: str) -> str:
        if bind_address.endswith(":*"):
            base = bind_address[: -len(":*")]
            port = socket.bind_to_random_port(base)
            return f"{base}:{port}"
        socket.bind(bind_address)
        return bind_address

    def address_for(self, lane: Lane) -> str:
        address = self._addresses.get(lane)
        if address is None:
            raise CascadeInternalError(f"ygg listener does not expose lane {lane}")
        return address

    def poll(self, timeout_ms: int | None) -> list[tuple[Lane, list[bytes]]]:
        ready = self._poller.poll(timeout_ms if timeout_ms is not None else None)
        messages: list[tuple[Lane, list[bytes]]] = []
        for socket, _ in ready:
            lane = self._lane_by_socket_id[id(socket)]
            messages.append((lane, socket.recv_multipart()))
        return messages

    def close(self) -> None:
        for socket in self._socket_by_lane.values():
            socket.close()
        self._socket_by_lane.clear()
        self._lane_by_socket_id.clear()
        self._addresses.clear()


def _describe_socket(socket: zmq.Socket) -> dict[str, Any]:
    return {
        "type": socket.getsockopt(zmq.TYPE),
        "events": socket.getsockopt(zmq.EVENTS),
        "endpoint": socket.getsockopt_string(zmq.LAST_ENDPOINT),
        "fd": socket.getsockopt(zmq.FD),
    }
