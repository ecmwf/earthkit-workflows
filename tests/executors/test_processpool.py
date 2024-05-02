import pytest
from execution_utils import execution_context

from cascade.executors.processpool import ProcessPoolExecutor, WorkerPool
from cascade.schedulers.depthfirst import DepthFirstScheduler


@pytest.mark.parametrize("schedule, kwargs", [[False, {"n_workers": 2}], [True, {}]])
def test_processpool(execution_context, schedule, kwargs):
    task_graph, context_graph = execution_context
    if schedule:
        task_graph = DepthFirstScheduler().schedule(task_graph, context_graph)
    executor = ProcessPoolExecutor(**kwargs)
    results = executor.execute(task_graph)
    assert all([x.name in results for x in task_graph.sinks])


@pytest.mark.parametrize(
    "func, args, callback", [[sum, ["Hello World"], print], [sum, [1, 2], sum]]
)
def test_workerpool(func, args, callback):
    def on_error(e):
        raise Exception(e)

    with pytest.raises(Exception):
        with WorkerPool(["worker-1", "worker-2"]) as executor:
            while True:
                executor.submit(
                    "worker-1", func, args, callback=callback, error_callback=on_error
                )
