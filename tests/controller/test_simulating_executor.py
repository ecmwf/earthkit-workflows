from math import isclose

import pytest

from cascade.controller.api import PurgingPolicy
from cascade.controller.impl import CascadeController
from cascade.controller.simulator import SimulatingExecutor
from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import Environment, Host, JobExecutionRecord, TaskExecutionRecord
from cascade.low.scheduler import schedule


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
    env = Environment(hosts={"h1": Host(cpu=2, gpu=0, memory_mb=2)})
    record_ok = JobExecutionRecord(
        datasets_mb={("task1", Node.DEFAULT_OUTPUT): 1},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
        },
    )
    # will fail when starting task2 because task2 wont fit
    record_bad1 = JobExecutionRecord(
        datasets_mb={("task1", Node.DEFAULT_OUTPUT): 1},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=2),
        },
    )
    # will fail when starting task2 because dataset wont fit
    record_bad2 = JobExecutionRecord(
        datasets_mb={("task1", Node.DEFAULT_OUTPUT): 2},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
        },
    )
    controller = CascadeController()
    policy = PurgingPolicy()

    executor_ok = SimulatingExecutor(env, record_ok)
    sched_ok = schedule(job, executor_ok.get_environment()).get_or_raise()
    controller.submit(job, sched_ok, executor_ok, policy)
    assert isclose(executor_ok.total_time_secs, 10 / 2 + 10 / 2)

    with pytest.raises(ValueError, match=r"host run out of memory by 1"):
        executor_bad1 = SimulatingExecutor(env, record_bad1)
        sched_bad1 = schedule(job, executor_bad1.get_environment()).get_or_raise()
        controller.submit(job, sched_bad1, executor_bad1, policy)

    with pytest.raises(ValueError, match=r"host run out of memory by 1"):
        executor_bad2 = SimulatingExecutor(env, record_bad2)
        sched_bad2 = schedule(job, executor_bad2.get_environment()).get_or_raise()
        controller.submit(job, sched_bad2, executor_bad2, policy)
