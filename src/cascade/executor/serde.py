# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module is responsible for Serialization & Deserialization of messages and outputs"""

import pickle
from typing import Any, Callable, Type

import cloudpickle

from cascade.executor.msg import Message
from cascade.low.func import resolve_callable

# NOTE for start, we simply pickle the msg classes -- this makes it possible
# to serialize eg `set`s (unlike `msgpack` or json-based serializers) while
# being reasonably performant for both small messages and large binary objects
# However, we want to switch over to a more data-oriented serializer (like
# `msgpack`) which is reasonable performant -- a custom format would be best,
# since the set of messages is fixed and small. However, beating pickle is *hard*
# with just python, even with `struct` or manual `int.to_bytes` etc
# NOTE that as those message are being shipped over zmq, we may want to delay
# some object concatenation to zmq submits -- otherwise we do memcpy twice,
# costing us both time and memory overhead. This would be a core feature of the
# custom serde. The message where this matters is DatasetTransmitPayload


def ser_message(m: Message) -> bytes:
    return pickle.dumps(m)


def des_message(b: bytes) -> Message:
    return pickle.loads(b)


class SerdeRegistry:
    # NOTE the contract is a bit odd -- Callable for ser, str for deser
    # We could switch to both being Callable, by having shm/DatasetTransmitPayload
    # operate with Type instead of deser_fun. But I'm a bit worried about Type being
    # reliable -- we may instead be forced to extracting the serde pair from Instance
    # rather than from Class
    serde: dict[Type, tuple[Callable, str]] = {}

    @classmethod
    def register(cls, t: Type, ser: str, des: str) -> None:
        cls.serde[t] = (
            resolve_callable(ser),
            des,
        )


def ser_output(v: Any, annotation: str) -> tuple[bytes, str]:
    """Utilizes `custom_ser` attr if present, otherwise defaults to cloudpickle as the most
    robust general purpose serde
    """
    if (serde := SerdeRegistry.serde.get(type(v), None)) is not None:
        value, deser_fun = serde[0](v), serde[1]
    else:
        value, deser_fun = cloudpickle.dumps(v), "cloudpickle.loads"
    return value, deser_fun


def des_output(v: bytes, annotation: str, deser_fun: str) -> Any:
    if deser_fun == "cloudpickle.loads":
        return cloudpickle.loads(v)
    else:
        return resolve_callable(deser_fun)(v)
