# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Util functions for generating larger graphs + their execution records

Generates graphs that look like this:
- one big source node
- multiple map layers, where each node either has input source node (think "select")
  or two random nodes from (some) previous layer (think "ensemble mean")
- multiple sink layers which consume a fraction of (some) previous layer
"""

import uuid
from dataclasses import dataclass, field

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import (
    DatasetId,
    Environment,
    JobExecutionRecord,
    JobInstance,
    TaskExecutionRecord,
    Worker,
    WorkerId,
)
from earthkit.workflows.graph import Node

# NOTE ideally we replace it with representative real world usecases

## *** graph builders ***


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
    builder.record.datasets_mb[DatasetId("source", Node.DEFAULT_OUTPUT)] = outsize
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
            e1 = f"pproc{from_layer}-{(i+131)%builder.layers[from_layer]}"
            e2 = f"pproc{from_layer}-{(i+53)%builder.layers[from_layer]}"
            builder.job = builder.job.with_node(
                node, TaskBuilder.from_callable(mapDiad)
            )
            builder.job = builder.job.with_edge(e1, node, "a")
            builder.job = builder.job.with_edge(e2, node, "b")
            # print(f"adding {node} with edges {e1}, {e2}")
        builder.record.tasks[node] = TaskExecutionRecord(
            cpuseconds=runtime, memory_mb=runmem
        )
        builder.record.datasets_mb[DatasetId(node, Node.DEFAULT_OUTPUT)] = outsize
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


def get_job0() -> tuple[JobInstance, JobExecutionRecord]:
    """One source, one pproc, one sink"""
    builder = BuilderGroup()
    add_large_source(builder, 10, 6, 4)
    add_postproc(builder, 0, 1, 1, 1, 1)
    add_sink(builder, 1, 1, 10, 10, 1)
    return builder.job.build().get_or_raise(), builder.record


def get_job1() -> tuple[JobInstance, JobExecutionRecord]:
    """One large source branching out into two sets of sinks"""
    builder = BuilderGroup()
    # data source: 10 minutes consuming 6G mem and producing 4G output
    add_large_source(builder, 10, 6, 4)
    # first processing layer -- each node selects disjoint 1G subset, in 1 minute and with 2G overhead
    add_postproc(builder, 0, 4, 1, 2, 1)
    # second processing layer -- 2 medium compute nodes, 6 minutes and 4G overhead, 1g output
    add_postproc(builder, 1, 2, 6, 4, 1)
    # sink for this branch, no big overhead/runtime
    # 2G output == prev layer has 2 nodes with 1G output each
    add_sink(builder, 2, 1, 1, 1, 2)

    # two more layers, parallel to the previous one: first reads layer 1, second reads the previous.
    # Less compute heavy and less mem, but 8 nodes each
    add_postproc(builder, 1, 8, 2, 1, 1)
    add_postproc(builder, 3, 8, 2, 1, 2)
    # sink for this branch, no big overhead/runtime
    # 16G output == prev layer has 8 nodes with 2G output each
    add_sink(builder, 4, 1, 1, 1, 16)
    return builder.job.build().get_or_raise(), builder.record


## *** environment builders ***
def get_env(hosts: int, workers_per_host: int) -> Environment:
    return Environment(
        workers={
            WorkerId(f"h{h}", f"w{w}"): Worker(cpu=1, gpu=0, memory_mb=1000)
            for h in range(hosts)
            for w in range(workers_per_host)
        }
    )
