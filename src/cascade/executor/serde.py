"""
This module is responsible for Serialization & Deserialization of messages
"""

# NOTE for start, we simply pickle the msg classes -- this makes it possible
# to serialize eg `set`s (unlike `msgpack` or json-based serializers) while
# being reasonably performant for both small messages and large binary objects
# However, we want to switch over to a more data-oriented serializer (like
# `msgpack`) which is reasonable performant -- a custom format would be best,
# since the set of messages is fixed and small.
# NOTE that as those message are being shipped over zmq, we may want to delay
# some object concatenation to zmq submits -- otherwise we do memcpy twice,
# costing us both time and memory overhead. This would be a core feature of the
# custom serde.

from cascade.executor.msg import Message

def ser(m: Message) -> bytes:
    raise NotImplementedError

def des(b: bytes) -> Message:
    raise NotImplementedError
