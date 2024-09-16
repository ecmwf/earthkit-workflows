from dask.threaded import get

from cascade.v2.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance
from cascade.v2.dask import job2dask


def test_linear():
    """Tests that a two node graph, defined using v2 api, gives correct result upon dask execution"""
    test_func = lambda x, y: x + y
    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "int", "y": "int"},
        output_schema={"z": "int"},
    )
    job = JobInstance(
        tasks={
            "a": TaskInstance(
                definition=task_definition, static_input={"x": 4, "y": 2}
            ),
            "b": TaskInstance(definition=task_definition, static_input={"x": 3}),
        },
        edges=[
            Task2TaskEdge(
                source_task="a",
                source_output="z",
                sink_task="b",
                sink_input="y",
            )
        ],
    )
    dask = job2dask(job)
    assert 4 + 3 + 2 == get(dask, "b")
