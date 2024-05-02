import numpy as np
import pytest
from execution_utils import execution_context

from cascade.executors.dask import DaskLocalExecutor
from cascade.executors.dask_utils.report import Report, TaskStream, duration_in_sec
from cascade.schedulers.depthfirst import DepthFirstScheduler


class MockTaskStream(TaskStream):
    def __init__(
        self, stream: dict[str, list[TaskStream.Task]], start: float, end: float
    ):
        self._stream = stream
        self.start = start
        self.end = end


def test_without_schedule(tmpdir, execution_context):
    task_graph, _ = execution_context
    DaskLocalExecutor().execute(task_graph, report=f"{tmpdir}/report-no-schedule.html")


@pytest.mark.skip("Sometimes fails in CI")
def test_with_schedule(tmpdir, execution_context):
    task_graph, context_graph = execution_context
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)

    # Parse performance report to check task stream is the same as task allocation
    # in schedule
    DaskLocalExecutor(n_workers=2).execute(
        schedule, report=f"{tmpdir}/report-schedule.html"
    )
    report = Report(f"{tmpdir}/report-schedule.html")
    for tasks in report.task_stream.stream(True).values():
        print("STREAM", [t.name for t in tasks])
        print("ALLOCATION", list(schedule.task_allocation.values()))
        assert [t.name for t in tasks] in list(schedule.task_allocation.values())

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


def test_task_stream():
    tasks = [
        TaskStream.Task(150, 250, "t2", "worker1", "thread1"),
        TaskStream.Task(175, 200, "t3", "worker1", "thread1"),
        TaskStream.Task(275, 300, "t4", "worker1", "thread1"),
    ]
    stream = MockTaskStream(
        {
            "worker1": tasks,
        },
        0.0,
        350.0,
    )
    assert stream.wall_time() == 350.0
    assert stream.is_enclosed("worker1", tasks[0]) == False
    assert stream.is_enclosed("worker1", tasks[1]) == True
    assert stream.is_enclosed("worker1", tasks[2]) == False


@pytest.mark.parametrize(
    "task_stream, idle_time",
    [
        [
            [
                TaskStream.Task(0, 100, "t1", "worker1", "thread1"),
                TaskStream.Task(150, 100, "t2", "worker1", "thread1"),
            ],
            150.0,
        ],
        [
            [
                TaskStream.Task(150, 100, "t2", "worker1", "thread1"),
                TaskStream.Task(175, 25, "t3", "worker1", "thread1"),
                TaskStream.Task(275, 25, "t4", "worker1", "thread1"),
            ],
            225.0,
        ],
        [
            [
                TaskStream.Task(275, 25, "t4", "worker1", "thread1"),
                TaskStream.Task(290, 60, "t4", "worker1", "thread1"),
            ],
            275.0,
        ],
    ],
    ids=["non-overlapping", "enclosed", "overlapping"],
)
def test_task_stream_idle(task_stream, idle_time):
    stream = MockTaskStream({"worker1": task_stream}, 0.0, 350.0)
    assert stream.idle_time() == {"worker1": idle_time}


@pytest.mark.parametrize(
    "task_stream, blocking_transfer_time, transfer_time",
    [
        [
            [
                TaskStream.Task(0, 100, "t1", "worker1", "thread1"),
                TaskStream.Task(150, 100, "transfer-t1", "worker1", "thread1"),
            ],
            100.0,
            100.0,
        ],
        [
            [
                TaskStream.Task(150, 100, "t1", "worker1", "thread1"),
                TaskStream.Task(175, 25, "transfer-t2", "worker1", "thread1"),
                TaskStream.Task(275, 25, "t2", "worker1", "thread1"),
            ],
            0.0,
            25.0,
        ],
        [
            [
                TaskStream.Task(275, 25, "t1", "worker1", "thread1"),
                TaskStream.Task(290, 60, "transfer-t1", "worker1", "thread1"),
            ],
            50.0,
            60.0,
        ],
    ],
    ids=["blocking", "enclosed", "overlapping"],
)
def test_task_stream_transfer(task_stream, blocking_transfer_time, transfer_time):
    stream = MockTaskStream({"worker": task_stream}, 0.0, 350.0)
    assert stream.transfer_time() == {"worker": blocking_transfer_time}
    assert stream.transfer_time(blocking=False) == {"worker": transfer_time}


def test_generate_context_graph():
    DaskLocalExecutor(n_workers=4).create_context_graph()
