import numpy as np
import xarray as xr
import pytest


from cascade.executors.dask import DaskLocalExecutor
from cascade.executors.dask_utils.report import Report, duration_in_sec
from cascade.schedulers.depthfirst import DepthFirstScheduler
from cascade.fluent import Fluent, Payload
from cascade.contextgraph import ContextGraph

context_graph = ContextGraph()
context_graph.add_node("worker_1", type="CPU", speed=10, memory=400)
context_graph.add_node("worker_2", type="CPU", speed=10, memory=200)
context_graph.add_edge("worker_1", "worker_2", bandwidth=0.1, latency=1)
context_graph


task_graph = (
    Fluent()
    .source(
        np.random.rand,
        xr.DataArray(
            [np.fromiter([(2, 3) for _ in range(6)], dtype=object) for _ in range(7)],
            dims=["x", "y"],
        ),
    )
    .mean("x")
    .min("y")
    .expand("z", 3, 1, 0)
    .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
    .graph()
)


def test_without_schedule(tmpdir):
    DaskLocalExecutor().execute(task_graph, report=f"{tmpdir}/report-no-schedule.html")


def test_with_schedule(tmpdir):
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)

    # Parse performance report to check task stream is the same as task allocation
    # in schedule
    DaskLocalExecutor().execute(schedule, 2, report=f"{tmpdir}/report-schedule.html")
    report = Report(f"{tmpdir}/report-schedule.html")
    for _, tasks in report.task_stream.stream(True).items():
        print("STREAM", tasks)
        print("ALLOCATION", schedule.task_allocation.values())
        assert tasks in list(schedule.task_allocation.values())

    # Adaptive with minimum number of workers less than workers in context
    # should raise error
    with pytest.raises(AssertionError):
        DaskLocalExecutor().execute(schedule, adaptive_kwargs={"minimum": 0})
        DaskLocalExecutor().execute(schedule, adaptive_kwargs={"maximum": 1})


def test_with_schedule_adaptive(tmpdir):
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)

    DaskLocalExecutor().execute(
        schedule,
        2,
        report=f"{tmpdir}/report-adaptive.html",
        adaptive_kwargs={"maximum": 3},
    )
    report = Report(f"{tmpdir}/report-adaptive.html")
    assert np.any(
        [
            tasks not in list(schedule.task_allocation.values())
            for tasks in report.task_stream.stream(True).values()
        ]
    )


@pytest.mark.parametrize(
    "duration_str, result",
    [
        ["3hr 22m", 12120.0],
        ["275.64 s", 275.64],
        ["48m 5s", 2885.0],
        ["48us", 0.000048],
        ["48ms", 0.048],
    ],
)
def test_duration_conversion(duration_str, result):
    assert duration_in_sec(duration_str) == result
