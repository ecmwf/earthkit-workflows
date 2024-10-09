from math import isclose

from cascade.controller.impl import CascadeController
from cascade.controller.simulator import SimulatingExecutor
from cascade.low.core import Environment, Host
from cascade.low.scheduler.api import EnvironmentState
from cascade.low.scheduler.simple import (
    bfs_schedule,
    dfs_one_worker_schedule,
    sink_bfs_redundant_schedule,
)

from .util import BuilderGroup, add_large_source, add_postproc, add_sink

# NOTE I use gigabytes & minutes in comments here, but the code actually operates with megabytes and seconds...
# consider it a readability simplification, otherwise we'd have *60 and *1024 all over the place

# NOTE these is a bit flaky test: the numbers asserted *should* change with every impl change of scheduler/controller


def test_2l_2sink():
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

    controller = CascadeController()
    # for calculating the min req mem on a single node
    one_biggie_env = Environment(hosts={"h1": Host(cpu=1, gpu=0, memory_mb=1000)})

    job = builder.job.build().get_or_raise()

    # bfs
    executor = SimulatingExecutor(one_biggie_env, builder.record)
    schedBfs = bfs_schedule(
        job, executor.get_environment(), builder.record, EnvironmentState()
    ).get_or_raise()
    print(schedBfs)
    controller.submit(job, schedBfs, executor)
    print(executor.hosts["h1"].mem_record)
    assert isclose(executor.total_time_secs, 60.0)
    assert (
        executor.hosts["h1"].mem_record <= 23
    )  # observed to oscillate between 19 and 23

    # dfs
    executor = SimulatingExecutor(one_biggie_env, builder.record)
    schedDfs = dfs_one_worker_schedule(
        job, executor.get_environment(), builder.record, EnvironmentState()
    ).get_or_raise()
    print(schedDfs)
    controller.submit(job, schedDfs, executor)
    print(executor.hosts["h1"].mem_record)
    assert isclose(executor.total_time_secs, 60.0)
    assert (
        executor.hosts["h1"].mem_record <= 25
    )  # observed to oscillate between 19 and 25

    # sink bfs -- one host
    executor = SimulatingExecutor(one_biggie_env, builder.record)
    schedSinkBfs = sink_bfs_redundant_schedule(
        job, executor.get_environment(), builder.record, EnvironmentState()
    ).get_or_raise()
    print(schedSinkBfs)
    controller.submit(job, schedSinkBfs, executor)
    print(executor.hosts["h1"].mem_record)
    assert isclose(executor.total_time_secs, 60.0)
    assert executor.hosts["h1"].mem_record in (
        19,
        23,
    )  # depends on which sink gets computed first

    # sink bfs -- two hosts
    two_aptly_sized = Environment(
        hosts={
            "h1": Host(cpu=1, gpu=0, memory_mb=23),
            "h2": Host(cpu=1, gpu=0, memory_mb=23),
        }
    )
    executor = SimulatingExecutor(two_aptly_sized, builder.record)
    schedSinkBfs = sink_bfs_redundant_schedule(
        job, executor.get_environment(), builder.record, EnvironmentState()
    ).get_or_raise()
    print(schedSinkBfs.host_task_queues["h1"])
    print(schedSinkBfs.host_task_queues["h2"])
    controller.submit(job, schedSinkBfs, executor)
    assert isclose(executor.total_time_secs, 47.0)
    print(executor.hosts["h1"].mem_record)
    print(executor.hosts["h2"].mem_record)
    assert {executor.hosts["h1"].mem_record, executor.hosts["h2"].mem_record} == {
        10,
        19,
    }
