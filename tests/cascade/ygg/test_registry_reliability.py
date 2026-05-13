import time

import pytest

from cascade.low.exceptions import CascadeInfrastructureError, CascadeInternalError
from cascade.ygg.registry import HostRegistry
from cascade.ygg.reliability import DedupCache, InflightMessage, RetryPlanner
from cascade.ygg.types import HostEndpoints, HostId, RetryPolicy, YggConfig


def test_registry_resolve_and_bulk_validation() -> None:
    registry = HostRegistry()
    registry.register(HostId("worker-a"), HostEndpoints(control="tcp://127.0.0.1:1001"))
    assert registry.resolve(HostId("worker-a"), "control") == "tcp://127.0.0.1:1001"
    with pytest.raises(CascadeInternalError):
        registry.resolve(HostId("worker-a"), "bulk")


def test_dedup_cache_marks_duplicates() -> None:
    dedup = DedupCache(ttl_ms=5_000)
    assert dedup.is_duplicate(3, "tcp://source:1") is False
    assert dedup.is_duplicate(3, "tcp://source:1") is True


def test_retry_planner_exhaustion() -> None:
    planner = RetryPlanner(
        YggConfig(
            control=RetryPolicy(retry_interval_ms=1, max_retries=1),
            bulk=RetryPolicy(retry_interval_ms=5, max_retries=1),
            dedup_ttl_ms=1000,
        )
    )
    inflight = InflightMessage(
        idx=1,
        host=HostId("worker-a"),
        lane="control",
        address="tcp://127.0.0.1:1001",
        syn_frame=b"syn",
        payload_frame=b"payload",
        attempts=2,
        sent_at_ns=time.time_ns() - 5_000_000,
    )
    with pytest.raises(CascadeInfrastructureError):
        planner.assert_not_exhausted(inflight)
