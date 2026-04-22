from dask._task_spec import DataNode, Task, TaskRef

from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstanceRich, TaskId
from cascade.low.dask import graph2job
from cascade.main import run_locally


def add(x, y):
    return x + y


def test_dask():
    dask_graph = {"x": (x := DataNode(None, 1)), "y": (y := DataNode(None, 2)), "z": Task("z", add, TaskRef("x"), TaskRef("y"))}
    cascade_job = graph2job(dask_graph)
    outputs = run_locally(
        JobInstanceRich(jobInstance=cascade_job, checkpointSpec=None),
        workers=2,
        hosts=1,
    )
    assert outputs == {DatasetId(TaskId("'z'"), DefaultTaskOutput): 3}
