import pytest

from cascade.v0_executors.dask import DaskLocalExecutor
from cascade.profiler import memray_profile, meters_profile
from cascade.v0_schedulers.depthfirst import DepthFirstScheduler


@pytest.mark.parametrize("schedule", [False, True])
def test_memray_profiler(tmpdir, task_graph, schedule):
    executor = DaskLocalExecutor(n_workers=2)
    if schedule:
        task_graph = DepthFirstScheduler().schedule(
            task_graph, executor.create_context_graph()
        )
    _, annotated_graph = memray_profile(task_graph, tmpdir, executor)
    nodes = list(annotated_graph.nodes())
    assert not all([node.duration == 0 for node in nodes])
    assert not all([node.memory == 0 for node in nodes])


@pytest.mark.parametrize("schedule", [False, True])
def test_meters_profiler(tmpdir, task_graph, schedule):
    executor = DaskLocalExecutor(n_workers=2)
    if schedule:
        task_graph = DepthFirstScheduler().schedule(
            task_graph, executor.create_context_graph()
        )
    _, annotated_graph = meters_profile(task_graph, f"{tmpdir}/meters_log", executor)
    nodes = list(annotated_graph.nodes())
    assert not all([node.duration == 0 for node in nodes])
    assert not all([node.memory == 0 for node in nodes])


def test_overwrite_existing_files(tmpdir, task_graph):
    executor = DaskLocalExecutor(n_workers=2)
    memray_profile(task_graph, tmpdir, executor)
    memray_profile(task_graph, tmpdir, executor)
