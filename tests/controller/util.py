"""
Util functions for generating larger graphs + their execution records

Generates graphs that look like this:
- one big source node
- multiple map layers, where each node either has input source node (think "select")
  or two random nodes from (some) previous layer (think "ensemble mean")
- multiple sink layers which consume a fraction of (some) previous layer
"""

import uuid
from dataclasses import dataclass, field

from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobExecutionRecord, TaskExecutionRecord

# NOTE ideally we replace it with representative real world usecases


def mapMonad(b: bytes) -> bytes:
    return b


def mapDiad(a: bytes, b: bytes) -> bytes:
    return a + b


def sourceFunc() -> bytes:
    return b""


def sinkFunc(*args) -> str:
    return "result_url"


@dataclass
class BuilderGroup:
    job: JobBuilder = field(default_factory=lambda: JobBuilder())
    record: JobExecutionRecord = field(
        default_factory=lambda: JobExecutionRecord(tasks={}, datasets_mb={})
    )
    layers: list[int] = field(default_factory=list)


def add_large_source(
    builder: BuilderGroup, runtime: int, runmem: int, outsize: int
) -> None:
    builder.job = builder.job.with_node("source", TaskBuilder.from_callable(sourceFunc))
    builder.record.tasks["source"] = TaskExecutionRecord(
        cpuseconds=runtime, memory_mb=runmem
    )
    builder.record.datasets_mb[("source", Node.DEFAULT_OUTPUT)] = outsize
    builder.layers = [1]


def add_postproc(
    builder: BuilderGroup,
    from_layer: int,
    n: int,
    runtime: int,
    runmem: int,
    outsize: int,
):
    for i in range(n):
        node = f"pproc{len(builder.layers)}-{i}"
        if from_layer == 0:
            builder.job = builder.job.with_node(
                node, TaskBuilder.from_callable(mapMonad)
            )
            builder.job = builder.job.with_edge("source", node, "b")
        else:
            builder.job = builder.job.with_node(
                node, TaskBuilder.from_callable(mapDiad)
            )
            builder.job = builder.job.with_edge(
                f"pproc{from_layer}-{(i+131)%builder.layers[from_layer]}", node, "a"
            )
            builder.job = builder.job.with_edge(
                f"pproc{from_layer}-{(i+71)%builder.layers[from_layer]}", node, "b"
            )
        builder.record.tasks[node] = TaskExecutionRecord(
            cpuseconds=runtime, memory_mb=runmem
        )
        builder.record.datasets_mb[(node, Node.DEFAULT_OUTPUT)] = outsize
    builder.layers.append(n)


def add_sink(
    builder: BuilderGroup,
    from_layer: int,
    frac: int,
    runtime: int,
    runmem: int,
    outsize: int,
):
    node = f"sink{uuid.uuid4()}"
    builder.job = builder.job.with_node(node, TaskBuilder.from_callable(sinkFunc))
    for i in range(builder.layers[from_layer] // frac):
        source = ((i * frac) + 157) % builder.layers[from_layer]
        builder.job = builder.job.with_edge(f"pproc{from_layer}-{source}", node, i)
    builder.record.tasks[node] = TaskExecutionRecord(
        cpuseconds=runtime, memory_mb=runmem
    )
