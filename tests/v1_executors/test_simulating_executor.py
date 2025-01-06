from math import isclose

import pytest

from cascade.controller.impl import run
from cascade.executors.simulator import SimulatingExecutor
from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import Environment, Worker, JobExecutionRecord, TaskExecutionRecord, DatasetId, WorkerId
from cascade.low.views import param_source
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
    task_inputs = {
        task_id: set(task_param_source.values())
        for task_id, task_param_source in param_source(job.edges).items()
    }
    env = Environment(workers={WorkerId("h0", "w1"): Worker(cpu=2, gpu=0, memory_mb=2)}, colocations=[["h0:w1"]])
    record_ok = JobExecutionRecord(
        datasets_mb={DatasetId("task1", Node.DEFAULT_OUTPUT): 1},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
        },
    )
    # will fail when starting task2 because task2 wont fit
    record_bad1 = JobExecutionRecord(
        datasets_mb={DatasetId("task1", Node.DEFAULT_OUTPUT): 1},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=2),
        },
    )
    # will fail when starting task2 because dataset wont fit
    record_bad2 = JobExecutionRecord(
        datasets_mb={DatasetId("task1", Node.DEFAULT_OUTPUT): 2},
        tasks={
            "task1": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
            "task2": TaskExecutionRecord(cpuseconds=10, memory_mb=1),
        },
    )

    def run_with_record(record: JobExecutionRecord) -> float:
        executor = SimulatingExecutor(env, task_inputs, record)
        preschedule = precompute(job)
        run(job, executor, preschedule)
        return executor.total_time_secs

    ok_run = run_with_record(record_ok)
    assert isclose(ok_run, 10 / 2 + 10 / 2)

    with pytest.raises(ValueError, match=r"worker run out of memory by 1"):
        run_with_record(record_bad1)

    with pytest.raises(ValueError, match=r"worker run out of memory by 1"):
        run_with_record(record_bad2)
