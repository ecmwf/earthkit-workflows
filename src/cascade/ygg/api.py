# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import time
from typing import Iterable

from cascade.low.exceptions import CascadeInternalError
from cascade.ygg.protocol import Ack, Syn, parse_envelope, serialize_ack, serialize_syn
from cascade.ygg.registry import HostRegistry
from cascade.ygg.reliability import DedupCache, InflightMessage, RetryPlanner
from cascade.ygg.transport import MultiLaneListener, OutboundTransport
from cascade.ygg.types import Delivery, HostEndpoints, HostId, IncomingMessage, Lane, YggConfig

logger = logging.getLogger(__name__)


class YggNode:
    def __init__(self, control_bind_address: str, bulk_bind_address: str | None = None, config: YggConfig = YggConfig()) -> None:
        self._config = config
        bind_addresses: dict[Lane, str] = {"control": control_bind_address}
        if bulk_bind_address is not None:
            bind_addresses["bulk"] = bulk_bind_address
        self._listener = MultiLaneListener(bind_addresses=bind_addresses, linger_ms=config.linger_ms)
        self._outbound = OutboundTransport(linger_ms=config.linger_ms)
        self._registry = HostRegistry()
        self._dedup = DedupCache(ttl_ms=config.dedup_ttl_ms)
        self._retry = RetryPlanner(config=config)
        self._inflight: dict[int, InflightMessage] = {}
        self._next_idx = 0
        self.control_address = self._listener.address_for("control")
        self.bulk_address = self._listener.address_for("bulk") if bulk_bind_address is not None else None

    def register_host(self, host_id: str, endpoints: HostEndpoints) -> None:
        self._registry.register(HostId(host_id), endpoints)

    def unregister_host(self, host_id: str) -> None:
        endpoints = self._registry.resolve_endpoints(HostId(host_id))
        self._registry.unregister(HostId(host_id))
        for address in [endpoints.control, endpoints.bulk]:
            if address is not None:
                self._dedup.purge_from(address)

    def forget_sender(self, address: str) -> None:
        """Explicitly purge dedup cache entries from a sender address.

        Use this when a particular sender finishes sending and you want to
        reclaim dedup cache memory without waiting for TTL expiration.

        This is called automatically by unregister_host for the host's endpoints.
        """
        self._dedup.purge_from(address)

    def send_message_to_host(
        self,
        host_id: str,
        payload: bytes,
        lane: Lane = "control",
        delivery: Delivery | None = None,
    ) -> int | None:
        host = HostId(host_id)
        address = self._registry.resolve(host, lane)
        resolved_delivery = self._resolve_delivery(lane=lane, delivery=delivery)
        if resolved_delivery == "best_effort":
            self._outbound.send_single(address=address, frame=payload)
            return None
        idx = self._next_idx
        self._next_idx += 1
        record = InflightMessage(
            idx=idx,
            host=host,
            lane=lane,
            address=address,
            syn_frame=serialize_syn(Syn(idx=idx, ack_address=self.control_address)),
            payload_frame=payload,
        )
        self._inflight[idx] = record
        self._send_record(record)
        return idx

    def broadcast(
        self,
        payload: bytes,
        lane: Lane = "control",
        hosts: Iterable[str] | None = None,
        delivery: Delivery | None = None,
    ) -> list[int | None]:
        target_hosts = hosts if hosts is not None else self._registry.hosts()
        sent: list[int | None] = []
        for host in target_hosts:
            sent.append(
                self.send_message_to_host(
                    host_id=str(host),
                    payload=payload,
                    lane=lane,
                    delivery=delivery,
                )
            )
        return sent

    def acknowledge(self, idx: int) -> bool:
        return self._inflight.pop(idx, None) is not None

    def pending_message_ids(self) -> set[int]:
        return set(self._inflight.keys())

    def poll_messages(self, timeout_ms: int | None = 0) -> list[IncomingMessage]:
        """Poll for incoming messages on registered lanes. Polls at most timeout_ms
        (or unlimited if None), then returns *all* messages that arrived. With timeout=0,
        returns all ready messages without blocking.

        Does NOT automatically call retry_outstanding. Callers should manage retries explicitly.
        """
        messages: list[IncomingMessage] = []
        for lane, frames in self._listener.poll(timeout_ms=timeout_ms):
            incoming = self._handle_incoming(lane, frames)
            if incoming is not None:
                messages.append(incoming)
        return messages

    def retry_outstanding(self) -> None:
        now = time.time_ns()
        for idx in list(self._inflight.keys()):
            record = self._inflight[idx]
            self._retry.assert_not_exhausted(record, now_ns=now)
            if self._retry.is_due(record, now_ns=now):
                logger.debug(f"retrying message {idx=}")
                self._send_record(record)

    def close(self, timeout_ms: int = 1000, wait_for_all_acks: bool = True) -> None:
        if wait_for_all_acks:
            deadline_ns = time.monotonic_ns() + timeout_ms * 1_000_000
            while self._inflight:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                poll_timeout_ms = min(remaining_ns // 1_000_000, self._config.control.retry_interval_ms)
                self.poll_messages(timeout_ms=poll_timeout_ms)
                self.retry_outstanding()
            if self._inflight:
                logger.warning("ygg close timed out with %d inflight messages", len(self._inflight))
        self._outbound.close()
        self._listener.close()

    def __enter__(self) -> "YggNode":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _send_record(self, record: InflightMessage) -> None:
        copy = record.lane == "control"
        self._outbound.send_multipart(record.address, (record.syn_frame, record.payload_frame), copy=copy)
        record.attempts += 1
        record.sent_at_ns = time.time_ns()

    def _resolve_delivery(self, lane: Lane, delivery: Delivery | None) -> Delivery:
        if delivery is not None:
            return delivery
        return self._config.delivery_for(lane)

    def _handle_incoming(self, lane: Lane, frames: list[bytes]) -> IncomingMessage | None:
        if not frames:
            raise CascadeInternalError("received empty ygg multipart message")
        envelope = parse_envelope(frames[0])
        if isinstance(envelope, Ack):
            if len(frames) != 1:
                raise CascadeInternalError("unexpected payload frame with ygg Ack")
            self.acknowledge(envelope.idx)
            return None
        if isinstance(envelope, Syn):
            if len(frames) != 2:
                raise CascadeInternalError("expected single payload frame after ygg Syn")
            self._outbound.send_single(envelope.ack_address, serialize_ack(Ack(idx=envelope.idx)))
            if self._dedup.is_duplicate(envelope.idx, envelope.ack_address):
                return None
            return IncomingMessage(payload=frames[1], lane=lane, source_address=envelope.ack_address, message_id=envelope.idx)
        if len(frames) != 1:
            raise CascadeInternalError("unexpected multipart ygg message without envelope")
        return IncomingMessage(payload=frames[0], lane=lane, source_address=None, message_id=None)

    def describe_state(self) -> dict[str, object]:
        return {
            "control_address": self.control_address,
            "bulk_address": self.bulk_address,
            "pending": len(self._inflight),
            "pending_ids": sorted(self._inflight.keys()),
            "listener": self._listener.describe_state(),
            "outbound": self._outbound.describe_state(),
        }
