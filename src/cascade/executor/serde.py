"""
This module is responsible for Serialization & Deserialization of messages and outputs
"""

from cascade.low.func import assert_never
from cascade.low.core import DatasetId
from cascade.executor.msg import Message, DatasetTransmitCommand, DatasetTransmitPayload, DatasetTransmitConfirm
from typing import Any
import cloudpickle
import pickle


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

# NOTE we cloudpickle here as that should be a bit more robust. However, we want
# to exploit custom serialization to a greater effect in selected cases

def ser_output(v: Any, annotation: str) -> bytes:
    return cloudpickle.dumps(v)

def des_output(v: bytes, annotation: str) -> Any:
    return cloudpickle.loads(v)

def ser_dmessage(m: DatasetTransmitCommand|DatasetTransmitPayload|DatasetTransmitConfirm) -> tuple[bytes]|tuple[bytes,bytes]:
    """Optimized variant for payloads and multipart send/recv"""
    if isinstance(m, DatasetTransmitCommand|DatasetTransmitConfirm):
        return ser_message(m),
    elif isinstance(m, DatasetTransmitPayload):
        # TODO this is messy, introduce a proper header-body hierarchy
        return (pickle.dumps([m.ds, m.confirm_address, m.confirm_idx]), m.value)
    else:
        assert_never(m)

def des_dmessage(bs: list[bytes]) -> DatasetTransmitCommand|DatasetTransmitPayload|DatasetTransmitConfirm:
    h = pickle.loads(bs[0])
    if isinstance(h, DatasetTransmitCommand|DatasetTransmitConfirm):
        if len(bs) != 1:
            raise ValueError(f"expected list of len 1, gotten {len(bs)}")
        return h
    elif isinstance(h, list):
        if len(bs) != 2:
            raise ValueError(f"expected list of len 2, gotten {len(bs)}")
        if len(h) != 3:
            raise ValueError(f"expected list of len 3, gotten {len(bs)}")
        return DatasetTransmitPayload(ds=h[0], confirm_address=h[1], confirm_idx=h[2], value=bs[1])
    else:
        raise ValueError(f"unexpected type: {type(h)}")
