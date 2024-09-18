from dask.threaded import get

from cascade.v2.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance
from cascade.v2.dask import job2dask


def test_linear():
    """Tests that a two node graph, defined using v2 api, gives correct result upon dask execution"""

    def test_func(x, y, z):
        return x + y + z

    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema_kw={"y": "int", "z": "int"},
        input_schema_ps={0: "int"},
        output_schema={"o": "int"},
    )
    job = JobInstance(
        tasks={
            "a": TaskInstance(
                definition=task_definition,
                static_input_kw={"y": 3, "z": 2},
                static_input_ps={0: 1},
            ),
            "b": TaskInstance(
                definition=task_definition,
                static_input_kw={"z": 4, "y": 5},
                static_input_ps={},
            ),
        },
        edges=[
            Task2TaskEdge(
                source_task="a",
                source_output="o",
                sink_task="b",
                sink_input_ps=0,
                sink_input_kw=None,
            )
        ],
    )
    dask = job2dask(job)
    assert 5 + 4 + 3 + 2 + 1 == get(dask, "b")
