import time
import pytest

from cascade.executors.processpool import ProcessPoolExecutor, WorkerPool
from cascade.schedulers.depthfirst import DepthFirstScheduler

from execution_utils import execution_context


def test_without_schedule(execution_context):
    task_graph, _ = execution_context
    ProcessPoolExecutor.execute(task_graph, n_workers=2)


def test_with_schedule(execution_context):
    task_graph, context_graph = execution_context
    schedule = DepthFirstScheduler().schedule(task_graph, context_graph)
    ProcessPoolExecutor.execute(schedule)


@pytest.mark.parametrize(
    "func, args, callback", [[sum, ["Hello World"], print], [sum, [1, 2], sum]]
)
def test_workerpool(func, args, callback):
    def on_error(e):
        raise Exception(e)

    with pytest.raises(Exception):
        with WorkerPool(["worker-1", "worker-2"]) as executor:
            executor.submit(
                "worker-1", func, args, callback=callback, error_callback=on_error
            )
            time.sleep(1)
