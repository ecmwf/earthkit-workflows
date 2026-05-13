# ygg communication layer: migration guidance

This document is for migration-only agents that should consume the new `ygg` APIs without re-designing transport internals.

## Migration objective

Replace direct ZeroMQ and legacy comms usage in a target region with `cascade.ygg` calls, while preserving existing behavior and message semantics.

## Scope boundaries for migration agents

- Use `01-summary.md` as behavioral contract.
- Use this document for migration mechanics.
- Do not re-architect ygg internals during migration.
- Keep changes local to the assigned region unless required by interface adaptation.

## Migration rules

1. Replace direct socket/address handling with:
   - host registration
   - ygg send/poll APIs
   - lane selection (`control` or `bulk`) at call site
2. Remove local retry/ack/dedup logic once equivalent ygg behavior is wired.
3. Keep application payload schema unchanged unless explicitly required.
4. Preserve error behavior and logging intent; adapt to ygg error types as needed.
5. Avoid introducing new transport shortcuts outside ygg.

## Adapter-first strategy

When a region is tightly coupled to legacy comms:

- add a thin adapter layer in that region that maps old call patterns to ygg
- migrate call sites to the adapter first
- then simplify/remove adapter once stable

This keeps refactors incremental and lowers blast radius.

## Lane mapping guidance

Choose lane by communication intent:

- use **control lane** for routine command/event/report messages
- use **bulk lane** for large dataset/payload transfer paths

If a call site currently uses a dedicated large-message path, map it to bulk lane explicitly.

## Behavior parity checklist

For each migrated region, ensure:

- host/endpoint resolution no longer depends on local socket maps
- delivery retries are provided by ygg, not duplicated locally
- duplicate delivery handling is still correct
- timeout/failure propagation remains explicit
- shutdown/cleanup paths still close resources cleanly

## Testing guidance for migrated regions

Each migration should update or add tests in that region to cover:

- nominal send/receive behavior through ygg
- retry/failure propagation at region boundary
- lane selection correctness for large-message paths
- no regression in existing message flow outcomes

Prefer preserving existing test shape and only replacing transport plumbing assumptions.

## Completion criteria for a migrated region

A region is considered migrated when:

- it no longer creates/manages transport sockets directly for ygg-managed flows
- it no longer owns bespoke ack/retry/dedup logic already provided by ygg
- it uses ygg host registration and send/poll APIs
- region tests pass with ygg-backed transport behavior
