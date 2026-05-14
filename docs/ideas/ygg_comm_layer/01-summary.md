# ygg communication layer: summary

`ygg` is a shortened name derived from Yggdrasil (a communication backbone concept), but the module name and API remain simply `ygg`.

## Purpose

`cascade.ygg` should be the single communication layer for internal process-to-process and host-to-host messaging over ZeroMQ, with these goals:

- centralize transport concerns currently spread across multiple modules
- provide one consistent reliability model for both regular and large messages
- hide socket wiring and host routing details from business/domain code
- stay domain-agnostic (no assumptions about specific dataclasses or workflow entities)

## Outer API shape

The top-level API should be operational and routing-focused, for example:

- `register_host(host_id, endpoints)`
- `unregister_host(host_id)`
- `forget_sender(address)` (explicit dedup cache purge)
- `send_message_to_host(host_id, payload, lane=...)`
- `broadcast(payload, lane=..., hosts=...)`
- `poll_messages(timeout_ms=...)`
- `acknowledge(...)` (if exposed explicitly; otherwise internal)

Domain modules should not need to manage socket instances directly or manually track address maps.

## Key concerns owned by ygg

1. **Reliable send/receive**
   - transport-level delivery protocol with retry and deduplication
   - inflight tracking, ack correlation, timeout/retry policy
   - clear failure signaling when delivery guarantees cannot be met

2. **Routing and endpoint registry**
   - host registration as the single source of endpoint truth
   - lane-aware endpoint selection (regular vs large-message path)
   - removal of ad hoc host naming/address conventions from callers

3. **Protocol framing**
   - ownership of `Syn`/`Ack` helper types and wire framing
   - consistent frame contract for control and bulk lanes
   - strict separation between transport envelope and application payload

4. **Lane model for message size/behavior**
   - at least two lanes:
     - **control lane** for regular small messages
     - **bulk lane** for large payload workflows
   - lane choice can be explicit parameterization (preferred) and/or policy-based helpers

## Non-goals

- no embedding of cascade workflow semantics into ygg core
- no requirement that callers use specific dataclass types
- no migration coupling in this document (migration is covered separately)
