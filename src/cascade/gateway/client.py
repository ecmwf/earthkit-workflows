# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles request & response communication for classes in gateway.api"""

import logging
import threading
from typing import cast

import orjson
import zmq

import cascade.gateway.api as api
from cascade.low.exceptions import CascadeInfrastructureError, CascadeInternalError

logger = logging.getLogger(__name__)


def request_response(m: api.CascadeGatewayAPI, url: str, timeout_ms: int = 1000) -> api.CascadeGatewayAPI:
    """Sends a Request message, provides a corresponding Response message in a blocking manner"""
    local = threading.local()
    if not hasattr(local, "context"):
        local.context = zmq.Context()

    try:
        # NOTE it would be nice to avoid orjson and go with 'model_dump_json()' and
        # 'model_validate_json()'. That would require constructing two-frame message instead,
        # with first frame being the `clazz`
        d = m.model_dump(mode="json")
        if "clazz" in d:
            raise CascadeInternalError("field `clazz` must not be present in the message")
        d["clazz"] = type(m).__name__
        if not d["clazz"].endswith("Request"):
            raise CascadeInternalError("message must be a Request")
        b = orjson.dumps(d)
    except Exception as e:
        logger.exception(f"failed to serialize message: {repr(m)[:32]}")
        raise CascadeInternalError(f"failed to serialize message: {repr(m)[:32]} => {repr(e)[:32]}", parent=e)

    try:
        s = local.context.socket(zmq.REQ)
        s.set(zmq.LINGER, timeout_ms)
        s.connect(url)
        s.send(b)
        mask = s.poll(timeout_ms, flags=zmq.POLLIN)
        if mask == 0:
            raise TimeoutError  # NOTE consider setting `err` on the response instead
        else:
            rr = s.recv()
    except TimeoutError as e:
        raise CascadeInfrastructureError(f"timed out on {url=} => {repr(e)[:32]}", parent=e)
    except Exception as e:
        logger.exception(f"failed to communicate on {url=}")
        raise CascadeInfrastructureError(f"failed to communicate on {url=} => {repr(e)[:32]}", parent=e)

    try:
        rd = orjson.loads(rr)
        rdc = rd.pop("clazz")
        if not rdc.endswith("Response"):
            raise CascadeInternalError("recieved message is not a Response")
        if d["clazz"][: -len("Request")] != rdc[: -len("Response")]:
            raise CascadeInternalError("mismatch between sent and received classes")
        if rdc not in api.__dict__.keys():
            raise CascadeInternalError("message clazz not understood")
        return cast(api.CascadeGatewayAPI, api.__dict__[rdc](**rd))
    except Exception as e:
        logger.exception(f"failed to parse message: {rr[:32]}")
        raise CascadeInternalError(f"failed to parse message: {rr[:32]} => {repr(e)[:32]}", parent=e)


def parse_request(rr: bytes) -> api.CascadeGatewayAPI:
    try:
        rd = orjson.loads(rr)
        rdc = rd.pop("clazz")
        if not rdc.endswith("Request"):
            raise CascadeInternalError("recieved message is not a Request")
        if rdc not in api.__dict__.keys():
            raise CascadeInternalError("message clazz not understood")
        return cast(api.CascadeGatewayAPI, api.__dict__[rdc](**rd))
    except Exception as e:
        logger.exception(f"failed to parse message: {rr[:32]!r}")
        raise CascadeInternalError(f"failed to parse message: {rr[:32]!r} => {repr(e)[:32]}", parent=e)


def serialize_response(m: api.CascadeGatewayAPI) -> bytes:
    try:
        d = m.model_dump()
        if "clazz" in d:
            raise ValueError("field `clazz` must not be present in the message")
        d["clazz"] = type(m).__name__
        if not d["clazz"].endswith("Response"):
            raise ValueError("message must be a Response")
        return orjson.dumps(d)
    except Exception as e:
        logger.exception(f"failed to serialize message: {repr(m)[:32]}")
        raise ValueError(f"failed to serialize message: {repr(m)[:32]} => {repr(e)[:32]}")
