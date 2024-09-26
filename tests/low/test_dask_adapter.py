from dask.threaded import get
from dask.distributed import LocalCluster, Client

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance, Host, Environment
from cascade.low.delayed import job2delayed
from cascade.low.futures import execute_via_futures
from cascade.low.scheduler import schedule

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
                source_task="a",
                source_output="o",
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
    cluster = LocalCluster(n_workers=1, processes=True, dashboard_address=':0')
    env = Environment(hosts={w: Host(memory_mb=1) for w in cluster.workers})
    sched = schedule(job, env).get_or_raise()
    result = execute_via_futures(job, sched, ("b", "o"), Client(cluster))
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
    cluster = LocalCluster(n_workers=1, processes=True, dashboard_address=':0')
    env = Environment(hosts={w: Host(memory_mb=1) for w in cluster.workers})
    sched = schedule(job, env).get_or_raise()
    result = execute_via_futures(job, sched, ("task2", "__default__"), Client(cluster))
    assert expected == result
