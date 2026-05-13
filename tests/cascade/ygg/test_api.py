import time

from cascade.ygg.api import YggNode
from cascade.ygg.types import HostEndpoints, RetryPolicy, YggConfig


def test_control_send_receive_ack() -> None:
    config = YggConfig(control=RetryPolicy(retry_interval_ms=10, max_retries=3), bulk=RetryPolicy(retry_interval_ms=10, max_retries=3))
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

        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()


def test_bulk_send_receive() -> None:
    config = YggConfig(control=RetryPolicy(retry_interval_ms=10, max_retries=3), bulk=RetryPolicy(retry_interval_ms=10, max_retries=3))
    payload = b"x" * 256_000
    with (
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as sender,
        YggNode("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", config=config) as receiver,
    ):
        sender.register_host("receiver", HostEndpoints(control=receiver.control_address, bulk=receiver.bulk_address))
        receiver.register_host("sender", HostEndpoints(control=sender.control_address, bulk=sender.bulk_address))

        idx = sender.send_large_message_to_host("receiver", payload)
        incoming = receiver.poll_messages(timeout_ms=250)
        assert len(incoming) == 1
        assert incoming[0].payload == payload
        assert incoming[0].lane == "bulk"
        assert incoming[0].message_id == idx

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
    config = YggConfig(control=RetryPolicy(retry_interval_ms=5, max_retries=4), bulk=RetryPolicy(retry_interval_ms=5, max_retries=2))
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

        sender.poll_messages(timeout_ms=250)
        assert sender.pending_message_ids() == set()
