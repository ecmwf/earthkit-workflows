# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for controller.report — verifies ygg-backed transport for controller→gateway reporting."""

import time
from time import monotonic_ns

from cascade.controller.report import (
    ControllerReport,
    JobId,
    JobProgress,
    Reporter,
    ReporterChannel,
    deserialize,
    serialize,
)
from cascade.low.core import DatasetId, TaskId
from cascade.ygg.api import YggNode
from cascade.ygg.types import HostEndpoints, RetryPolicy, YggConfig


def _fast_config() -> YggConfig:
    return YggConfig(
        control=RetryPolicy(retry_interval_ms=20, max_retries=5),
    )


def _make_gateway_ygg() -> YggNode:
    return YggNode("tcp://127.0.0.1:*", config=_fast_config())


def test_reporter_channel_sends_via_ygg() -> None:
    """ReporterChannel delivers a ControllerReport to a listening YggNode (gateway side)."""
    with _make_gateway_ygg() as gateway_ygg:
        report_address = f"{gateway_ygg.control_address},test-job-1"
        channel = ReporterChannel(report_address)
        try:
            report = ControllerReport(
                job_id=JobId("test-job-1"),
                current_status=JobProgress.progressed(0.5),
                timestamp=monotonic_ns(),
                results=[],
            )
            channel.send(report)

            # Give message time to arrive; drain gateway ygg
            deadline = time.time() + 2.0
            received: list[bytes] = []
            while time.time() < deadline and not received:
                msgs = gateway_ygg.poll_messages(timeout_ms=50)
                for msg in msgs:
                    received.append(msg.payload)

            assert len(received) == 1
            parsed = deserialize(received[0])
            assert parsed.job_id == "test-job-1"
            assert parsed.current_status is not None
            assert parsed.current_status.pct == "50.00"
        finally:
            channel.close()


def test_reporter_channel_reliable_delivery_ack_flow() -> None:
    """Sent message eventually gets acked and inflight is cleared."""
    with _make_gateway_ygg() as gateway_ygg:
        report_address = f"{gateway_ygg.control_address},test-job-2"
        channel = ReporterChannel(report_address)
        try:
            report = ControllerReport(
                job_id=JobId("test-job-2"),
                current_status=JobProgress.succeeded(),
                timestamp=monotonic_ns(),
                results=[],
            )
            channel.send(report)

            # Drain gateway to trigger ack send back to controller
            deadline = time.time() + 2.0
            while time.time() < deadline:
                gateway_ygg.poll_messages(timeout_ms=50)
                channel._ygg.poll_messages(timeout_ms=0)
                channel._ygg.retry_outstanding()
                if not channel._ygg.pending_message_ids():
                    break

            assert channel._ygg.pending_message_ids() == set(), "inflight should be cleared after ack"
        finally:
            channel.close()


def test_reporter_channel_close_is_idempotent() -> None:
    with _make_gateway_ygg() as gateway_ygg:
        report_address = f"{gateway_ygg.control_address},test-job-close"
        channel = ReporterChannel(report_address)
        channel.close()
        # Closing again should not raise
        channel.close()


def test_reporter_none_address_is_noop() -> None:
    """Reporter with no address silently skips all send methods."""
    reporter = Reporter(None)
    reporter.send_failure("some error")
    reporter.success()
    reporter.close()


def test_reporter_sends_result() -> None:
    """Reporter.send_result delivers the result payload to the gateway."""
    with _make_gateway_ygg() as gateway_ygg:
        report_address = f"{gateway_ygg.control_address},test-job-result"
        reporter = Reporter(report_address)
        try:
            dataset_id = DatasetId(TaskId("my-task"), "out")
            reporter.send_result(dataset_id, b"result-bytes")

            deadline = time.time() + 2.0
            received: list[ControllerReport] = []
            while time.time() < deadline and not received:
                msgs = gateway_ygg.poll_messages(timeout_ms=50)
                for msg in msgs:
                    received.append(deserialize(msg.payload))

            assert len(received) == 1
            assert received[0].results == [(dataset_id, b"result-bytes")]
        finally:
            reporter.close()


def test_reporter_dedup_retries_not_delivered_twice() -> None:
    """If the ack is slow, the controller retries; gateway must deduplicate."""
    with _make_gateway_ygg() as gateway_ygg:
        report_address = f"{gateway_ygg.control_address},test-job-dedup"
        channel = ReporterChannel(report_address)
        try:
            report = ControllerReport(
                job_id=JobId("test-job-dedup"),
                current_status=JobProgress.progressed(0.1),
                timestamp=monotonic_ns(),
                results=[],
            )
            channel.send(report)

            # Simulate retries without acking: send again with retry_outstanding
            for _ in range(3):
                time.sleep(0.025)
                channel._ygg.retry_outstanding()

            # Now drain all messages from gateway
            time.sleep(0.1)
            all_msgs: list[bytes] = []
            deadline = time.time() + 1.0
            while time.time() < deadline:
                msgs = gateway_ygg.poll_messages(timeout_ms=50)
                for msg in msgs:
                    all_msgs.append(msg.payload)
                if not msgs:
                    break

            # Despite retries, dedup ensures only one delivery
            assert len(all_msgs) == 1
        finally:
            channel.close()
