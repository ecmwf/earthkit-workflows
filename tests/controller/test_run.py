"""
For a given graph and instant executor, check that things complete
"""

from cascade.controller.impl import run
from cascade.executors.instant import InstantExecutor
from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import Environment, JobExecutionRecord, TaskExecutionRecord
from cascade.scheduler.graph import precompute

def _payload(a: int, b: int) -> int:
    return a + b

def test_simple():
    # 2-node graph
    task1 = TaskBuilder.from_callable(_payload).with_values(a=1, b=2)
    task2 = TaskBuilder.from_callable(_payload).with_values(a=1)
    job = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .with_edge("task1", "task2", "b")
        .build()
        .get_or_raise()
    )
    preschedule = precompute(job)

    executor1 = InstantExecutor(1, job)
    run(job, executor1, preschedule)

def test_para():
    # 3-component graph:
    # c1: 2 sources, 4 sinks
    # c2: 2 sources, 1 sink
    # c3: 1 source, 1 sink
    source = TaskBuilder.from_callable(_payload).with_values(a=1, b=2)
    sink1 = TaskBuilder.from_callable(_payload).with_values(a=1)
    sink2 = TaskBuilder.from_callable(_payload)
    job = (
        JobBuilder()
        .with_node("c1i1", source)
        .with_node("c1i2", source)
        .with_node("c1o1", sink2)
        .with_edge("c1i1", "c1o1", "a")
        .with_edge("c1i2", "c1o1", "b")
        .with_node("c1o2", sink2)
        .with_edge("c1i1", "c1o2", "a")
        .with_edge("c1i2", "c1o2", "b")
        .with_node("c1o3", sink2)
        .with_edge("c1i1", "c1o3", "a")
        .with_edge("c1i2", "c1o3", "b")
        .with_node("c1o4", sink2)
        .with_edge("c1i1", "c1o4", "a")
        .with_edge("c1i2", "c1o4", "b")
        .with_node("c2i1", source)
        .with_node("c2i2", source)
        .with_node("c2o1", sink2)
        .with_edge("c2i1", "c2o1", "a")
        .with_edge("c2i2", "c2o1", "b")
        .with_node("c3i1", source)
        .with_node("c3o1", sink1)
        .with_edge("c3i1", "c3o1", "b")
        .build()
        .get_or_raise()
    )

    preschedule = precompute(job)

    executor1 = InstantExecutor(1, job)
    run(job, executor1, preschedule)

    executor2 = InstantExecutor(2, job)
    run(job, executor2, preschedule)

    executor3 = InstantExecutor(3, job)
    run(job, executor3, preschedule)
