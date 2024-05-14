import pytest

from cascade.executors.dask import DaskLocalExecutor
from cascade.profiler import profile
from cascade.schedulers.depthfirst import DepthFirstScheduler


@pytest.mark.parametrize("schedule", [False, True])
def test_profiler(tmpdir, task_graph, schedule):
    executor = DaskLocalExecutor(n_workers=2)
    if schedule:
        task_graph = DepthFirstScheduler().schedule(
            task_graph, executor.create_context_graph()
        )
    _, annotated_graph = profile(task_graph, tmpdir, executor)
    nodes = list(annotated_graph.nodes())
    assert not all([node.duration == 0 for node in nodes])
    assert not all([node.memory == 0 for node in nodes])


def test_overwrite_existing_files(tmpdir, task_graph):
    executor = DaskLocalExecutor(n_workers=2)
    profile(task_graph, tmpdir, executor)
    profile(task_graph, tmpdir, executor)
