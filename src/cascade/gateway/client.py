"""Handles request & response communication for classes in gateway.api"""

import logging
import threading
from typing import cast

import orjson
import zmq

import cascade.gateway.api as api

logger = logging.getLogger(__name__)


def request_response(m: api.CascadeGatewayAPI, url: str) -> api.CascadeGatewayAPI:
    """Sends a Request message, provides a corresponding Response message in a blocking manner"""
    local = threading.local()
    if not hasattr(local, "context"):
        local.context = zmq.Context()

    try:
        d = (
            m.model_dump()
        )  # NOTE we don't `mode='json'` here for now as it's a bit too smart
        if "clazz" in d:
            raise ValueError("field `clazz` must not be present in the message")
        d["clazz"] = type(m).__name__
        if not d["clazz"].endswith("Request"):
            raise ValueError("message must be a Request")
        b = orjson.dumps(d)
    except Exception as e:
        logger.exception(f"failed to serialize message: {repr(m)[:32]}")
        raise ValueError(
            f"failed to serialize message: {repr(m)[:32]} => {repr(e)[:32]}"
        )

    try:
        s = local.context.socket(zmq.REQ)
        s.connect(url)
        s.send(b)
        rr = s.recv()
    except Exception as e:
        logger.exception(f"failed to communicate on {url=}")
        raise ValueError(f"failed to communicate on {url=} => {repr(e)[:32]}")

    try:
        rd = orjson.loads(rr)
        rdc = rd.pop("clazz")
        if not rdc.endswith("Response"):
            raise ValueError("recieved message is not a Response")
        if d["clazz"][: -len("Request")] != rdc[: -len("Response")]:
            raise ValueError("mismatch between sent and received classes")
        if rdc not in api.__dict__.keys():
            raise ValueError("message clazz not understood")
        return cast(api.CascadeGatewayAPI, api.__dict__[rdc](**rd))
    except Exception as e:
        logger.exception(f"failed to parse message: {rr[:32]}")
        raise ValueError(f"failed to parse message: {rr[:32]} => {repr(e)[:32]}")


def parse_request(rr: bytes) -> api.CascadeGatewayAPI:
    try:
        rd = orjson.loads(rr)
        rdc = rd.pop("clazz")
        if not rdc.endswith("Request"):
            raise ValueError("recieved message is not a Request")
        if rdc not in api.__dict__.keys():
            raise ValueError("message clazz not understood")
        return cast(api.CascadeGatewayAPI, api.__dict__[rdc](**rd))
    except Exception as e:
        logger.exception(f"failed to parse message: {rr[:32]!r}")
        raise ValueError(f"failed to parse message: {rr[:32]!r} => {repr(e)[:32]}")


def serialize_response(m: api.CascadeGatewayAPI) -> bytes:
    try:
        d = m.dict()
        if "clazz" in d:
            raise ValueError("field `clazz` must not be present in the message")
        d["clazz"] = type(m).__name__
        if not d["clazz"].endswith("Response"):
            raise ValueError("message must be a Response")
        return orjson.dumps(d)
    except Exception as e:
        logger.exception(f"failed to serialize message: {repr(m)[:32]}")
        raise ValueError(
            f"failed to serialize message: {repr(m)[:32]} => {repr(e)[:32]}"
        )
