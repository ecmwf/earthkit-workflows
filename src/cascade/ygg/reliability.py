# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time
from dataclasses import dataclass

from cascade.low.exceptions import CascadeInfrastructureError
from cascade.ygg.types import HostId, Lane, RetryPolicy, YggConfig


@dataclass
class InflightMessage:
    idx: int
    host: HostId
    lane: Lane
    address: str
    syn_frame: bytes
    payload_frame: bytes
    attempts: int = 0
    sent_at_ns: int = 0


class DedupCache:
    def __init__(self, ttl_ms: int) -> None:
        self._ttl_ns = ttl_ms * 1_000_000
        self._seen: dict[tuple[int, str], int] = {}

    def _purge(self, now_ns: int) -> None:
        watermark = now_ns - self._ttl_ns
        stale = [key for key, seen_at in self._seen.items() if seen_at < watermark]
        for key in stale:
            self._seen.pop(key, None)

    def is_duplicate(self, idx: int, source: str, now_ns: int | None = None) -> bool:
        now = now_ns if now_ns is not None else time.time_ns()
        self._purge(now)
        key = (idx, source)
        if key in self._seen:
            return True
        self._seen[key] = now
        return False


class RetryPlanner:
    def __init__(self, config: YggConfig) -> None:
        self._config = config

    def is_due(self, inflight: InflightMessage, now_ns: int | None = None) -> bool:
        now = now_ns if now_ns is not None else time.time_ns()
        if inflight.attempts == 0:
            return False
        policy = self._policy(inflight.lane)
        return now - inflight.sent_at_ns >= policy.retry_interval_ms * 1_000_000

    def assert_not_exhausted(self, inflight: InflightMessage, now_ns: int | None = None) -> None:
        now = now_ns if now_ns is not None else time.time_ns()
        if inflight.attempts == 0:
            return
        policy = self._policy(inflight.lane)
        if now - inflight.sent_at_ns < policy.retry_interval_ms * 1_000_000:
            return
        if inflight.attempts >= self._max_attempts(policy):
            raise CascadeInfrastructureError(
                f"ygg delivery exhausted for idx={inflight.idx}, host={inflight.host}, lane={inflight.lane}, attempts={inflight.attempts}"
            )

    def _policy(self, lane: Lane) -> RetryPolicy:
        return self._config.policy_for(lane)

    def _max_attempts(self, policy: RetryPolicy) -> int:
        return policy.max_retries + 1
