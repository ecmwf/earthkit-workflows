import pytest
import numpy as np

from cascade.executors.dask import DaskLocalExecutor
from cascade.executors.dask_utils.report import Report, duration_in_sec
from cascade.schedulers.depthfirst import DepthFirstScheduler

from execution_utils import execution_context


def test_without_schedule(tmpdir, execution_context):
    task_graph, _ = execution_context
    DaskLocalExecutor().execute(task_graph, report=f"{tmpdir}/report-no-schedule.html")


def test_with_schedule(tmpdir, execution_context):
    task_graph, context_graph = execution_context
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)

    # Parse performance report to check task stream is the same as task allocation
    # in schedule
    DaskLocalExecutor(n_workers=2).execute(
        schedule, report=f"{tmpdir}/report-schedule.html"
    )
    report = Report(f"{tmpdir}/report-schedule.html")
    for _, tasks in report.task_stream.stream(True).items():
        assert tasks in list(schedule.task_allocation.values())

    # Adaptive with minimum number of workers less than workers in context
    # should raise error
    with pytest.raises(ValueError):
        DaskLocalExecutor(adaptive_kwargs={"minimum": 0}).execute(schedule)
        DaskLocalExecutor(adaptive_kwargs={"maximum": 1}).execute(schedule)


def test_with_schedule_adaptive(tmpdir, execution_context):
    task_graph, context_graph = execution_context
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)

    DaskLocalExecutor(n_workers=2, adaptive_kwargs={"maximum": 3}).execute(
        schedule,
        report=f"{tmpdir}/report-adaptive.html",
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


def test_generate_context_graph():
    DaskLocalExecutor(n_workers=4).create_context_graph()