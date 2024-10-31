"""
Interface for tracing important events that can be used for extracting performance information

Currently, the export is handled just by logging, assuming to be parsed later. We log at debug
level since this is assumed to be high level tracing
"""

import logging
from typing import Literal
from cascade.low.func import assert_never
import os
import time

d: dict[str, str] = {}

logger = logging.getLogger(__name__)

def _labels(labels: dict) -> str:
    # TODO a bit crude to call this at every mark -- precache some scoping
    return ";".join(f"{k}={v}" for k, v in labels.items())

def label(key: str, value: str) -> None:
    """Makes all subsequent marks contain this KV. Carries over to later-forked subprocesses, but
    not to forkspawned"""
    global d
    d[key] = value

def mark(labels: dict) -> None:
    at = time.perf_counter_ns()
    global d
    event = _labels({**d, **labels})
    logger.debug(f"{event};{at=}")
