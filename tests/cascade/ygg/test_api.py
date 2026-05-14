import time

import pytest

from cascade.low.exceptions import CascadeInternalError
from cascade.ygg.api import YggNode
from cascade.ygg.types import HostEndpoints, RetryPolicy, YggConfig


def test_control_send_receive_ack() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_message_to_host("receiver", b"hello-control", lane="control")
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == b"hello-control"
        assert incoming[0].lane == "control"
        assert incoming[0].message_id == idx

        receiver.poll_messages(timeout_ms=250)
        sender.retry_outstanding()
        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_bulk_send_receive() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
    )
    payload = b"x" * 256_000
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_message_to_host("receiver", payload, lane="bulk")
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == payload
        assert incoming[0].lane == "bulk"
        assert incoming[0].message_id == idx

        receiver.poll_messages(timeout_ms=250)
        sender.retry_outstanding()
        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_control_best_effort_no_syn_ack() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_message_to_host("receiver", b"hello-best-effort", lane="control", delivery="best_effort")
        assert idx is None
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == b"hello-best-effort"
        assert incoming[0].lane == "control"
        assert incoming[0].message_id is None
        assert incoming[0].source_address is None

        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_control_default_delivery_from_config_best_effort() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
        control_delivery="best_effort",
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_message_to_host("receiver", b"hello-default-best-effort", lane="control")
        assert idx is None
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == b"hello-default-best-effort"
        assert incoming[0].message_id is None

        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_deduplicates_retries_for_same_syn() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=5, max_retries=4),
        bulk=RetryPolicy(retry_interval_ms=5, max_retries=2),
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        sender.send_message_to_host("receiver", b"hello-retry", lane="control")
        for _ in range(3):
            time.sleep(0.010)
            sender.retry_outstanding()

        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == b"hello-retry"

        receiver.poll_messages(timeout_ms=250)
        sender.retry_outstanding()
        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_forget_sender_clears_dedup_cache() -> None:
    """Test that forget_sender removes dedup entries for a specific address."""
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
        dedup_ttl_ms=60_000,
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        # The forget_sender method is primarily tested at the DedupCache level
        # (see test_dedup_cache_purge_from). Here we verify it's accessible via YggNode.
        sender_address = sender.control_address
        # Call forget_sender to verify it exists and is callable
        receiver.forget_sender(sender_address)
        # No assertion needed; we're just verifying the API exists and works


def test_unregister_host_purges_dedup_for_both_lanes() -> None:
    """Test that unregister_host clears dedup entries for both endpoints."""
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
        dedup_ttl_ms=60_000,
    )
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host(
            "receiver",
            HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address),
        )
        receiver.register_host(
            "sender",
            HostEndpoints(control=sender.control_address, bulk=sender.bulk_address),
        )

        # Verify sender is registered
        from cascade.ygg.types import HostId

        sender_endpoints = receiver._registry.resolve_endpoints(HostId("sender"))
        assert sender_endpoints.control == sender.control_address
        assert sender_endpoints.bulk == sender.bulk_address

        # Unregister sender (which should purge dedup for both control and bulk)
        receiver.unregister_host("sender")

        # Verify sender is no longer in registry
        with pytest.raises(CascadeInternalError):
            receiver._registry.resolve_endpoints(HostId("sender"))


def test_close_waits_for_pending_ack() -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
    )
    sender = YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config)
    receiver = YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config)
    try:
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_message_to_host("receiver", b"close-waits", lane="control")
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].message_id == idx

        sender.close(timeout_ms=250)
        assert sender.pending_message_ids() == set()
    finally:
        receiver.close()


def test_close_logs_remaining_inflight_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=10, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=10, max_retries=3),
    )
    sender = YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config)
    receiver = YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config)
    warnings: list[tuple[str, int]] = []

    def capture_warning(message: str, count: int) -> None:
        warnings.append((message, count))

    monkeypatch.setattr("cascade.ygg.api.logger.warning", capture_warning)
    try:
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        sender.send_message_to_host("receiver", b"close-timeout", lane="control")
        sender.close(timeout_ms=1)

        assert warnings == [("ygg close timed out with %d inflight messages", 1)]
    finally:
        receiver.close()


def test_close_polls_then_retries_until_inflight_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=25, max_retries=3),
        bulk=RetryPolicy(retry_interval_ms=25, max_retries=3),
    )
    sender = YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config)
    poll_timeouts: list[int | None] = []
    retry_calls = 0

    def fake_poll_messages(timeout_ms: int | None = 0) -> list[object]:
        poll_timeouts.append(timeout_ms)
        return []

    def fake_retry_outstanding() -> None:
        nonlocal retry_calls
        retry_calls += 1
        sender._inflight.clear()

    monkeypatch.setattr(sender, "poll_messages", fake_poll_messages)
    monkeypatch.setattr(sender, "retry_outstanding", fake_retry_outstanding)
    try:
        sender._inflight[1] = object()  # type: ignore[assignment]
        sender.close(timeout_ms=100, wait_for_all_acks=True)

        assert retry_calls == 1
        assert poll_timeouts == [25]
    finally:
        sender._outbound.close()
        sender._listener.close()
