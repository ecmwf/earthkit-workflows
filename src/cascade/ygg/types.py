# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from typing import Literal, NewType

Lane = Literal["control", "bulk"]
HostId = NewType("HostId", str)


@dataclass(frozen=True)
class HostEndpoints:
    control: str
    bulk: str | None = None


@dataclass(frozen=True)
class RetryPolicy:
    retry_interval_ms: int
    max_retries: int


@dataclass(frozen=True)
class YggConfig:
    control: RetryPolicy = RetryPolicy(retry_interval_ms=800, max_retries=20)
    bulk: RetryPolicy = RetryPolicy(retry_interval_ms=4_000, max_retries=10)
    dedup_ttl_ms: int = 60_000
    linger_ms: int = 1_000

    def policy_for(self, lane: Lane) -> RetryPolicy:
        if lane == "control":
            return self.control
        return self.bulk


@dataclass(frozen=True)
class IncomingMessage:
    payload: bytes
    lane: Lane
    source_address: str | None
    message_id: int | None
