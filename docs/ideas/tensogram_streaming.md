# Tensogram Streaming in Cascade

## Motivation

Cascade tasks that produce tensogram output currently have two choices: return a single
`tensogram.Message` (the whole field in one go) or manually `yield` individual messages
using the existing multi-output generator support.  The manual-yield path works but places
the chunking decision entirely on the user and offers no pipelining: each `yield` maps to
a statically declared, named output dataset, so the number of chunks must be known at graph
build time, and downstream tasks cannot start until the upstream task finishes yielding
everything.

What we want instead is **system-driven, streaming output**: the user writes a loop that
pushes data into an encoder, marks each logical message complete, and cascade takes care of
forwarding chunks to consumers as they arrive — allowing downstream tasks to start processing
the first message while the producer is still encoding the second.  For tensogram specifically,
the wire format provides the natural chunk boundary: one complete tensogram message (preamble +
body + postamble) is self-describing, independently decodable, and exactly the right unit to
pipeline.  No user-level chunking policy is needed.

The design splits cleanly into two phases.  **Phase 1** establishes the stable user-facing API
and wires it through the existing data path (shared memory + controller routing), accepting the
latency overhead of routing each chunk through the controller.  **Phase 2** is a performance
optimisation that replaces the routing path with a direct producer-to-consumer ZMQ channel and
extends the scheduler state machine accordingly; the user-facing API is unchanged.

---

## Phase 0: Adopt dedicated Tensogram serde

In executor/runner/entrypoint.py, in the SerdeRegistry registration (or even better at
the default constructor), include registration of tensogram serdes, conditioned on try-wrapped
tensogram import. The functions should live in a cascade module, like
"executor.serde.contrib.tensogram", and consist of:
 - for tensogram.Message: `tensogram.message_to_bytes(v)`, `tensogram.decode(b)`
 - for tensogram.RawMessage: `cast(RawMessage, v)._raw`, `RawMessage(b)`

---

## Phase 1: Stable API, existing data path

The goal of this phase is a fully functional end-to-end streaming implementation that an
implementer can build according to this spec, with the understanding that the routing is not
yet optimal.  All user-visible API decisions made here are final.

### User-facing contract

#### Producer side

A streaming producer is a regular Python generator function whose return type is annotated as
`Iterator[tensogram.Message]`.  The function yields complete tensogram messages one at a time.
Cascade detects this annotation, treats the output as a homogeneous sequence of chunks (not N
separately named outputs as in the existing multi-output generator model), and forwards each
yielded chunk to waiting consumers as it arrives.

```python
import tensogram
from typing import Iterator

def encode_fields(fields: list[Field]) -> Iterator[tensogram.Message]:
    enc = tensogram.StreamingEncoder(global_meta={})
    for field in fields:
        descriptor, data = prepare(field)
        enc.push_object(descriptor, data)
        raw_bytes = enc.finish_and_reset()           # tensogram: finalise + reuse encoder
        yield tensogram.decode(raw_bytes)
```

`finish_and_reset()` is a new tensogram API (tracked in tensogram's IDEAS.md) that finalises
the current message and returns its wire bytes while resetting the encoder in-place, ready for
the next message.  Until that API lands, the equivalent is `finish()` followed by constructing
a new `StreamingEncoder`.

The cascade graph builder must accept `Iterator[tensogram.Message]` (and `Iterator[T]` in
general for a single T) as a valid single-output annotation.  Currently `builders.py` recognises
`"typing.Iterator"` only to suppress a type-mismatch warning (marked TODO); that TODO becomes
real here.  An `Iterator[T]` return type means exactly one logical output dataset whose values
arrive incrementally.

#### Consumer side

A consumer that receives a streaming dataset declares its parameter as `Iterator[tensogram.Message]`:

```python
def consume_fields(stream: Iterator[tensogram.Message]) -> SomeResult:
    total = 0
    for msg in stream:
        total += process(msg)
    return total
```

The iterator blocks on `__next__` until the next chunk is available, and raises `StopIteration`
when the producer has finished.  From the consumer's perspective this is an ordinary Python
iterator; it has no knowledge of ZMQ, SHM, or cascade internals.

#### Edge type annotation

In the graph DAG, the edge connecting a streaming producer output to a consumer input carries
the type `Iterator[tensogram.Message]` (or the generalised `Iterator[T]`).  `JobBuilder.with_edge()`
must be extended to validate that a streaming-typed output connects only to a streaming-typed
input parameter.  Connecting a streaming output to a non-iterator input, or vice versa, is an
error at graph build time.

---

### Runner changes (producer side)

In `cascade/executor/runner/runner.py`, the existing generator branch (`isinstance(result, Generator)`)
maps each yield to a named output slot.  A new branch is needed for the streaming case:

1. Detect `Iterator[tensogram.Message]` (or `Iterator[T]`) return annotation on the task definition.
2. For each value `v` yielded by the generator:
   - Assign it a chunk key derived from the output name and a monotonically increasing counter,
     e.g. `DatasetId(task_id, f"{output_name}:chunk:{n}")`.
   - Call `memory.handle(chunk_id, chunk_schema, v, publish=True)` — this serialises `v` via the
     tensogram serde (see below) and puts it into SHM, then sends `DatasetPublished` to the
     controller.
3. After the generator is exhausted, send a `StreamComplete(ds=DatasetId(task_id, output_name))`
   message to the controller.  This is a new message type that tells the controller the stream
   is finished and no more chunks will arrive.
4. If the generator raises an exception, send `StreamError(ds=..., error=...)` instead of
   `StreamComplete`.  The controller propagates this as a task failure.

The chunk key scheme (`output_name:chunk:N`) must be parseable by the controller to identify
which logical dataset a chunk belongs to.  A simple helper `chunk_id(base_ds, n) -> DatasetId`
and `base_ds_of(chunk_id) -> DatasetId` suffices.

---

### Serialisation

The tensogram serde registered in `cascade/serde_tensogram.py` (see the broader integration
plan) handles serialisation of `tensogram.Message` objects.  Each chunk is serialised to
tensogram wire bytes and stored in SHM.  The `deser_fun` stored alongside in SHM is
`"cascade.serde_tensogram.des"` (i.e. `tensogram.decode`), same as for non-streaming messages.
No new serialisation code is needed; the streaming path reuses the single-message serde.

---

### Controller changes

The controller must learn to forward stream chunks to consumers and to dispatch consumers early
(on first chunk) rather than waiting for `StreamComplete`.

**New message types** (add to `cascade/executor/msg.py`):
- `StreamComplete(ds: DatasetId)` — producer finished cleanly.
- `StreamError(ds: DatasetId, error: str)` — producer failed mid-stream.

**`notify.py` changes:**
- When a `DatasetPublished` arrives whose dataset key matches the `chunk_id` pattern, the
  controller identifies the parent logical dataset and notes that a chunk has arrived.
- On the **first chunk** of a given logical dataset: mark the logical dataset as `streaming`
  (a new `DatasetStatus` value — see Phase 2 for the full state machine treatment; for Phase 1
  a simple boolean flag `streaming_datasets: set[DatasetId]` on `JobExecutionContext` is
  sufficient) and run `gang_check_ready` for consumers that depend on it.
  The readiness check must be extended: a consumer is ready when all its non-streaming inputs
  are `available` AND each of its streaming inputs has at least one chunk (i.e. appears in
  `streaming_datasets`).
- On `StreamComplete`: mark the logical dataset as `available` (for the purpose of checkpointing
  and external output collection), remove it from `streaming_datasets`, and notify any consumer
  that was still waiting for this signal to know iteration is over.
- On `StreamError`: propagate as task failure to the affected consumer(s).

**Transmit path:** chunk datasets are small, named SHM entries.  The existing `data_server`
transmit mechanism works as-is — a chunk is fetched by the consumer worker exactly like any
other dataset.  The extra round-trip latency through the controller is the accepted cost of
Phase 1.

---

### Consumer runner changes

When a consumer task has a streaming input, `memory.provide(input_id)` must return an iterator
instead of a single value.  Concretely:

- At task dispatch time the runner knows (from the task's input annotation) which inputs are
  `Iterator[T]`.  For those inputs, instead of blocking until the full dataset is available,
  it hands the task a `StreamReader` object.
- `StreamReader.__next__` requests the next chunk key from the controller (a new lightweight
  request `NextChunk(base_ds) -> DatasetId | StopIteration | Error`), fetches that chunk from
  SHM via the normal `shm_client.get` path, deserialises it, and returns the value.
- When the controller has seen `StreamComplete` and there are no more chunks to hand out, it
  replies with a sentinel that causes `StreamReader.__next__` to raise `StopIteration`.
- When the controller has seen `StreamError`, the sentinel carries the error string and
  `StreamReader.__next__` raises it as a `CascadeUserError`.

`StreamReader` is created inside `memory.provide()` and is transparent to the user task — the
task just iterates over it with a `for` loop.

---

### What is explicitly deferred to Phase 2

- Direct producer-to-consumer ZMQ socket (no controller round-trip per chunk).
- `DatasetStatus.streaming` as a proper enum value and full state machine treatment.
- Scheduler-level optimisations (chunk prefetching, consumer affinity to producer host).
- Multi-stream fan-in (consumer with two or more streaming inputs); Phase 1 supports exactly
  one streaming input per consumer task.
- Backpressure beyond ZMQ's default high-water mark.
- Non-tensogram streaming types (the infrastructure is type-agnostic, but only the tensogram
  serde is wired up in Phase 1).

---

## Phase 2: Performance through direct routing and scheduler rework

This section is **design input**, not an implementation spec.  The scheduler and controller
state machines are likely to be reworked for independent reasons; the streaming requirements
below should be treated as constraints and requirements for that rework, not as a prescription
for how to implement it.

### The core problem with Phase 1 routing

In Phase 1, every chunk takes the path:

```
producer SHM → DatasetPublished → controller → NextChunk request → consumer fetches from SHM
```

For small messages or high chunk rates this round-trip dominates latency.  The controller
becomes a bottleneck and a single point of failure for in-flight stream data.  Phase 2 replaces
the data plane with a direct PUSH/PULL ZMQ socket between producer worker and consumer worker,
retaining the controller only for signalling (stream start, end, error).

### Direct producer-consumer socket

Before the producer task starts, the controller must pre-allocate a dedicated ZMQ PUSH address
for the stream and communicate it to both sides:
- The producer worker receives a `StreamSend(ds, push_address)` command alongside the normal
  task assignment; it binds a PUSH socket there.
- The consumer worker receives a `StreamRecv(ds, push_address)` command; it connects a PULL
  socket there.

The producer pushes each chunk (tensogram wire bytes) directly to the consumer without
involving the controller.  An EOF frame (zero-length or a well-known magic prefix) signals
end-of-stream.  Error frames carry an encoded exception.

This is a new ZMQ socket pair per streaming dataset per job.  Socket lifecycle (bind, unbind,
cleanup on error) must be handled carefully to avoid resource leaks, particularly when workers
crash or are restarted.

### Scheduler state machine requirements

The existing `DatasetStatus` (missing → preparing → available → purged) is insufficient for
streaming.  The new state machine must express at minimum:

- **stream_pending**: the dataset is declared as streaming; no chunks have arrived yet.
- **stream_active**: at least one chunk has been sent; the consumer may be running.
- **available** (terminal, reused): the producer finished cleanly; all chunks delivered.
- **failed** (terminal): the producer failed mid-stream.

Readiness logic for consumer dispatch should be encoded in the scheduler, not scattered across
`notify.py`.  A consumer with a streaming input becomes dispatchable when:
1. All non-streaming inputs are `available`.
2. The streaming input is at least `stream_active` (first chunk ready to receive).

The existing `gang_check_ready` / `gang_preparation` machinery should be extended to express
this mixed readiness condition, ideally without special-casing streaming at the call sites.

### Backpressure

With a direct PUSH socket and ZMQ's default high-water mark, a faster producer will eventually
block on `send()` when the consumer's receive buffer fills.  This is acceptable behaviour (the
producer naturally pauses) but should be documented and the HWM should be configurable.
Explicit flow control (consumer sends ACKs back to producer) is an option if blocking turns out
to be problematic in practice but should not be implemented speculatively.

### Multi-stream fan-in

A consumer with two or more streaming inputs must be able to make progress on whichever stream
has data available, without blocking on a specific one.  `zmq.Poller` across the multiple PULL
sockets is the standard mechanism and requires no new infrastructure beyond registering the
sockets.  The `StreamReader` from Phase 1 should be designed with this in mind: it should
accept an external poller so that a future multi-stream wrapper can share one polling loop
across all inputs.

### Error propagation and worker crash

If the producer worker dies (OOM, segfault) without sending an EOF frame, the consumer will
block indefinitely on `__next__`.  The controller must detect worker death (via its existing
health-check mechanism) and synthesise a `StreamError` that causes the consumer's PULL socket
to receive an error sentinel.  This requires the controller to track which streaming sockets
are alive and to close or poison them on worker failure.

### Cross-host streaming

Phase 1 routes through SHM and the controller, which already handles cross-host transfer.
Phase 2's direct socket works for both same-host and cross-host scenarios — ZMQ PUSH/PULL is
network-transparent — but address allocation must use routable addresses (not `ipc://`) when
producer and consumer are on different hosts.  The controller already knows host assignments
and can supply the correct address format.
