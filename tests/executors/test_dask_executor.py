from dask.distributed import LocalCluster
from dask.threaded import get

from cascade.executors.dask_delayed import job2delayed
from cascade.executors.dask_futures import DaskFuturisticExecutor
from cascade.controller.impl import run
from cascade.controller.core import ActionDatasetPurge
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance, JobExecutionRecord, DatasetId
from cascade.scheduler.impl import naive_bfs_layers

# TODO instead of every process launching its own cluster, introduce some global fixture or smth like that


def test_linear():
    """Tests that a two node graph, defined using cascade.low.core, gives correct result upon dask execution"""

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
                source=DatasetId("a", "o"),
                sink_task="b",
                sink_input_ps=0,
                sink_input_kw=None,
            )
        ],
    )
    delayed = job2delayed(job)
    expected = 5 + 4 + 3 + 2 + 1
    assert expected == get(delayed, "b")

    # NOTE processes=False kinda buggy, complaints about unreleased futures... maybe some gil-caused quirk
    with LocalCluster(n_workers=1, processes=True, dashboard_address=":0") as cluster:
        executor = DaskFuturisticExecutor(cluster, job)
        schedule = naive_bfs_layers(job, JobExecutionRecord(), set()).get_or_raise()
        run(job, executor, schedule)
        output_id = DatasetId("b", "o")
        result = executor.fetch_as_value(output_id)
        executor.purge(ActionDatasetPurge(ds={output_id}, at={"0"}))
        assert expected == result


def test_builders():
    """Tests that a two node graph, defined using cascade.low.builders, gives correct result upon dask execution"""

    def test_func(x, y, z):
        return x + y + z

    # 1-node graph
    task = TaskBuilder.from_callable(test_func).with_values(x=1, y=2, z=3)
    job = JobInstance(tasks={"task": task}, edges=[])
    delayed = job2delayed(job)
    assert 1 + 2 + 3 == get(delayed, "task")

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
    delayed = job2delayed(job)
    expected = 5 + 4 + 3 + 2 + 1
    assert expected == get(delayed, "task2")

    # NOTE processes=False kinda buggy, complaints about unreleased futures... maybe some gil-caused quirk
    with LocalCluster(n_workers=1, processes=True, dashboard_address=":0") as cluster:
        executor = DaskFuturisticExecutor(cluster, job)
        schedule = naive_bfs_layers(job, JobExecutionRecord(), set()).get_or_raise()
        run(job, executor, schedule)
        output_id = DatasetId("task2", "__default__")
        result = executor.fetch_as_value(output_id)
        executor.purge(ActionDatasetPurge(ds={output_id}, at={"0"}))
        assert expected == result

