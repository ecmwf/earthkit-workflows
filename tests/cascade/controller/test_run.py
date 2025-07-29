# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""For a given graph and instant executor, check that things complete"""

# NOTE this is a very bulky test, prone to flakyness, interference, incompatible
# with xdist etc. Consider removing from pytest altogether, and reframing
# as integration test script or smth. Until then, we execute it only manually
# by setting RUN_ALL_TESTS=1, as on CI is particularly flaky and incompatible
# with xdist etc


import os
from logging.config import dictConfig
from multiprocessing import Process
from time import sleep

from cascade.controller.impl import run
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.config import logging_config
from cascade.executor.executor import Executor
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance
from cascade.scheduler.core import Preschedule
from cascade.scheduler.precompute import precompute

# see the comment above
run_all_tests = int(os.environ.get("RUN_ALL_TESTS", "0")) == 1


def _payload(a: int, b: int) -> int:
    return a + b


def launch_executor(
    job_instance: JobInstance,
    controller_address: BackboneAddress,
    portBase: int,
    i: int,
):
    dictConfig(logging_config)
    executor = Executor(
        job_instance,
        controller_address,
        2,
        f"test_executor{i}",
        portBase,
        None,
        None,
        "tcp://localhost",
    )
    executor.register()
    executor.recv_loop()


def run_cluster(
    job: JobInstance,
    portBase: int,
    executors: int,
    preschedule: Preschedule | None = None,
):
    # TODO rework the port assignemnt in this whole file -- we waste a lot. Note if there
    # is a port overlap, it causes *very unpleasant* interference, even when the tests
    # are executed sequentially
    if not preschedule:
        preschedule = precompute(job)
    c = f"tcp://localhost:{portBase}"
    m = f"tcp://localhost:{portBase+1}"
    ps = []
    for i, executor in enumerate(range(executors)):
        p = Process(target=launch_executor, args=(job, c, portBase + 1 + i * 10, i))
        p.start()
        ps.append(p)
    try:
        b = Bridge(c, executors)
        run(job, b, preschedule)
    except:
        for p in ps:
            if p.is_alive():
                callback(m, ExecutorShutdown())
                import time

                time.sleep(1)
                p.kill()
        raise


def test_simple():
    if not run_all_tests:
        return
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
    run_cluster(job, 12000, 1)
    sleep(1)  # improves stability


def get_job() -> JobInstance:
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
    return job


def test_para2():
    if not run_all_tests:
        return
    job = get_job()
    run_cluster(job, 12200, 2)
    sleep(1)  # improves stability


def test_para4():
    if not run_all_tests:
        return
    job = get_job()
    run_cluster(job, 12400, 4)
    sleep(1)  # improves stability


def test_fusing():
    if not run_all_tests:
        return

    source = TaskBuilder.from_callable(_payload).with_values(a=1, b=2)
    middle = TaskBuilder.from_callable(_payload).with_values(b=1)

    job = (
        JobBuilder()
        .with_node("source", source)
        .with_node("m1", middle)
        .with_edge("source", "m1", "a")
        .with_node("m2", middle)
        .with_edge("m1", "m2", "a")
        .with_node("m3", middle)
        .with_edge("m2", "m3", "a")
        .with_node("m4", middle)
        .with_edge("m3", "m4", "a")
        .with_node("sink", middle)
        .with_edge("m4", "sink", "a")
        .build()
        .get_or_raise()
    )
    preschedule = precompute(job)
    assert preschedule.components[0].fusing_opportunities["source"] == [
        "source",
        "m1",
        "m2",
        "m3",
        "m4",
        "sink",
    ]
    # TODO we currently dont check that those actually *got fused* -- fix
    run_cluster(job, 12600, 2, preschedule)
    sleep(1)  # improves stability
