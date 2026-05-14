# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

from cascade.low.exceptions import CascadeInternalError

_MAGIC = b"YGG1"
_TAG_SYN = b"S"
_TAG_ACK = b"A"


@dataclass(frozen=True)
class Syn:
    idx: int
    ack_address: str


@dataclass(frozen=True)
class Ack:
    idx: int


def serialize_syn(value: Syn) -> bytes:
    ack = value.ack_address.encode("utf-8")
    return _MAGIC + _TAG_SYN + value.idx.to_bytes(8, "big", signed=False) + len(ack).to_bytes(4, "big", signed=False) + ack


def serialize_ack(value: Ack) -> bytes:
    return _MAGIC + _TAG_ACK + value.idx.to_bytes(8, "big", signed=False)


def parse_envelope(frame: bytes) -> Syn | Ack | None:
    """Parse frame as ygg Syn/Ack envelope. Returns None for non-envelope or malformed data.

    Never raises -- malformed frames are silently treated as plain payloads.
    """
    try:
        if not frame.startswith(_MAGIC):
            return None
        if len(frame) < len(_MAGIC) + 1:
            return None
        tag = frame[len(_MAGIC) : len(_MAGIC) + 1]
        body = frame[len(_MAGIC) + 1 :]
        if tag == _TAG_ACK:
            if len(body) != 8:
                return None
            return Ack(idx=int.from_bytes(body, "big", signed=False))
        if tag == _TAG_SYN:
            if len(body) < 12:
                return None
            idx = int.from_bytes(body[:8], "big", signed=False)
            length = int.from_bytes(body[8:12], "big", signed=False)
            ack_raw = body[12:]
            if len(ack_raw) != length:
                return None
            return Syn(idx=idx, ack_address=ack_raw.decode("utf-8"))
        return None
    except Exception:
        return None
