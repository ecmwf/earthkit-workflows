from math import isclose
from typing import Callable

from cascade.controller.impl import run
from cascade.executors.simulator import SimulatingExecutor
from cascade.low.core import Environment, Worker, TaskId, JobInstance
from cascade.low.views import param_source
from cascade.scheduler.core import Schedule, Scheduler
from cascade.scheduler.impl import (
    naive_bfs_layers,
    naive_dfs_layers,
)

from .util import get_job1

# NOTE I use gigabytes & minutes in comments here, but the code actually operates with megabytes and seconds...
# consider it a readability simplification, otherwise we'd have *60 and *1024 all over the place

# NOTE these is a bit flaky test: the numbers asserted *should* change with every impl change of scheduler/controller


def test_2l_2sink():
    job, record = get_job1()
    task_inputs = {
        task_id: set(task_param_source.values())
        for task_id, task_param_source in param_source(job.edges).items()
    }

    # for calculating the min req mem on a single node
    one_biggie_env = Environment(workers={"w1": Worker(cpu=1, gpu=0, memory_mb=1000)})
    # memory_mb=18 chosen so that currently worst-case schedule fits
    two_tightly_sized = Environment(
        workers={
            "w1": Worker(cpu=1, gpu=0, memory_mb=21),
            "w2": Worker(cpu=1, gpu=0, memory_mb=21),
        }
    )

    def test(environment: Environment, scheduler: Scheduler) -> tuple[float, dict[str, int]]:
        executor = SimulatingExecutor(environment, task_inputs, record)
        schedule = scheduler(job, record, set()).get_or_raise()
        run(job, executor, schedule)
        return executor.total_time_secs, {w_id: w_in.mem_record for w_id, w_in in executor.workers.items()}

    r = test(one_biggie_env, naive_bfs_layers)
    assert isclose(r[0], 60.0)
    assert 19 <= r[1]["w1"] and r[1]["w1"] <= 23  # oscillation observed
    print(r[1])

    r = test(one_biggie_env, naive_dfs_layers)
    assert isclose(r[0], 60.0)
    assert 19 <= r[1]["w1"] and r[1]["w1"] <= 25  # oscillation observed
    print(r[1])

    r = test(two_tightly_sized, naive_bfs_layers)
    assert r[0] < 49.0
    assert max(r[1].values()) <= 18 # usually 17, min oscillates in 9,13
    # print(r[1])

    r = test(two_tightly_sized, naive_dfs_layers)
    assert r[0] < 49.0
    assert max(r[1].values()) <= 21 # usually 17, min oscillates in 12,16
    # print(r[1])
