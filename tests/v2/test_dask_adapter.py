from dask.threaded import get

from cascade.v2.builders import JobBuilder, TaskBuilder
from cascade.v2.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance
from cascade.v2.dask import job2dask


def test_linear():
    """Tests that a two node graph, defined using v2 core, gives correct result upon dask execution"""

    def test_func(x, y, z):
        return x + y + z

    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "int", "y": "int", "z": "int"},
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


def test_builders():
    """Tests that a two node graph, defined using v2 builders, gives correct result upon dask execution"""

    def test_func(x, y, z):
        return x + y + z

    # 1-node graph
    task = TaskBuilder.from_callable(test_func).with_values(x=1, y=2, z=3)
    job = JobInstance(tasks={"task": task}, edges=[])
    dask = job2dask(job)
    assert 1 + 2 + 3 == get(dask, "task")

    # 2-node graph
    task1 = TaskBuilder.from_callable(test_func).with_values(x=1, y=2, z=3)
    task2 = TaskBuilder.from_callable(test_func).with_values(y=4, z=5)
    job = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .with_edge("task1", "task2", 0)
        .build()
        .get_or_raise()
    )
    dask = job2dask(job)
    assert 5 + 4 + 3 + 2 + 1 == get(dask, "task2")
