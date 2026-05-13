# ygg communication layer: implementation notes

## Design principles

- **Domain-neutral core:** transport deals in bytes plus metadata, not business objects.
- **Explicit contracts:** typed configuration/state objects for reliability, endpoints, and lanes.
- **Single responsibility:** ygg owns transport protocol, routing registry, and delivery mechanics.
- **Policy over ad hoc logic:** retry, dedup, and timeout behavior are configurable policies.

## Proposed internal structure

Suggested package layout:

- `cascade/ygg/protocol.py`
  - transport envelope models (`Syn`, `Ack`, optional envelope metadata)
  - frame encoding/decoding helpers
- `cascade/ygg/registry.py`
  - host and endpoint registration
  - lane-aware endpoint lookup
- `cascade/ygg/reliability.py`
  - inflight store, retry scheduler, ack processing, dedup cache
  - resend/backoff policy
- `cascade/ygg/transport.py`
  - ZeroMQ context/socket lifecycle management
  - send/recv multipart primitives, poll integration
- `cascade/ygg/api.py`
  - high-level façade consumed by other modules (`register_host`, `send_message_to_host`, ...)
- `cascade/ygg/types.py`
  - shared config/state dataclasses (frozen where possible)

## Configuration model

Introduce a typed `YggConfig` (or equivalent) with:

- default lane (`control`/`bulk`)
- per-lane timeout and retry settings
- max retries and retry interval/backoff mode
- dedup retention window (count- or time-based)
- socket options relevant to reliability (linger, high-water mark, etc.)

Per-lane override should be supported:

- control lane: lower timeout, tighter retry cadence
- bulk lane: higher timeout, potentially fewer but longer retries

## Reliability contract

### Send path

1. Resolve destination endpoint from registry and selected lane.
2. Wrap payload in transport envelope with unique message id.
3. Record inflight entry (message id, host, lane, attempt state, timestamp).
4. Send framed message.
5. On ack, clear inflight entry.
6. On timeout, retry according to policy.
7. On retry exhaustion, raise/report delivery failure.

### Receive path

1. Parse incoming frames into envelope + payload.
2. If envelope is `Syn`, immediately emit `Ack`.
3. Perform dedup check by message id/source.
4. Drop duplicates after ack.
5. Emit payload to caller only for first-seen messages.

## Large vs regular message handling

Support both through one API with lane semantics:

- `send_message_to_host(..., lane="control")`
- `send_message_to_host(..., lane="bulk")`

Bulk lane requirements:

- multipart-friendly framing
- copy-avoidance options where possible
- reliability behavior equivalent in guarantees, with lane-specific timing policy

## Error model

- Infrastructure/transport failures should propagate as explicit errors.
- Invalid protocol/framing should be treated as internal/protocol errors.
- No silent drops in reliability paths.
- Logging should include message id, host id, lane, and attempt count.

## Testing expectations for ygg implementation

Implementation agents using this document should include:

- unit tests for protocol framing/parsing
- unit tests for retry/ack/dedup behavior
- unit tests for registry and lane endpoint resolution
- integration tests (in-process) for end-to-end send/ack on both lanes
- failure-path tests: timeout, duplicate ack, unknown host, malformed frames
