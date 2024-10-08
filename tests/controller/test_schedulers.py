from math import isclose

from cascade.controller.api import PurgingPolicy
from cascade.controller.impl import CascadeController
from cascade.controller.simulator import SimulatingExecutor
from cascade.low.core import Environment, Host
from cascade.low.scheduler import schedule

from .util import BuilderGroup, add_large_source, add_postproc, add_sink

# NOTE I use gigabytes & minutes in comments here, but the code actually operates with megabytes and seconds...
# consider it a readability simplification, otherwise we'd have *60 and *1024 all over the place


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
    policy = PurgingPolicy()
    # for calculating the min req mem on a single node
    one_biggie_env = Environment(hosts={"h1": Host(cpu=1, gpu=0, memory_mb=1000)})

    job = builder.job.build().get_or_raise()
    executor = SimulatingExecutor(one_biggie_env, builder.record)
    sched = schedule(job, executor.get_environment()).get_or_raise()
    controller.submit(job, sched, executor, policy)

    # NOTE this is a bit flaky test: the numbers would change with every impl change of scheduler/controller
    # the exec time seems stable with the current impl, the memory oscillates between 18 and 22
    assert isclose(executor.total_time_secs, 60.0)
    assert executor.hosts["h1"].mem_record <= 22
