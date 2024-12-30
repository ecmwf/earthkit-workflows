"""
For a given graph and instant executor, check that things complete
"""

from cascade.controller.impl import run
from cascade.executors.instant import InstantExecutor
from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import Environment, JobExecutionRecord, TaskExecutionRecord
from cascade.scheduler.graph import precompute


def test_simple():
    def simple_func(a: int, b: int) -> int:
        return a + b

    # 2-node graph
    task1 = TaskBuilder.from_callable(simple_func).with_values(a=1, b=2)
    task2 = TaskBuilder.from_callable(simple_func).with_values(a=1)
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

    # TODO bigger graph
    # executor2 = InstantExecutor(2, job)
    # run(job, executor2, preschedule)
