# /// script
# dependencies = ["dask[distributed]>=2026.1.2"]
# ///
# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Experimental module to convert dask graphs into cascade jobs. May not preserve
all semantics.

We don't explicitly support legacy dask graph -- you need to invoke
`dask._task_spec.convert_legacy_graph` yourself.
For higher level dask objects, extract the graph via `__dask_graph__()` (defined
eg on dd.DataFrame or dask.delayed objects).
"""

import logging
from typing import Any

from dask._task_spec import Alias, DataNode, Task, TaskRef

from cascade.low.builders import TaskBuilder
from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstance, JobInstanceRich, Task2TaskEdge, TaskId, TaskInstance
from cascade.low.exceptions import CascadeUserError
from cascade.main import run_locally

logger = logging.getLogger(__name__)


def daskKeyRepr(key: str | int | float | tuple) -> str:
    return repr(key)


def task2task(key: str, task: Task) -> tuple[TaskInstance, list[Task2TaskEdge]]:
    instance = TaskBuilder.from_callable(task.func)
    edges: list[Task2TaskEdge] = []

    for i, v in enumerate(task.args):
        if isinstance(v, Alias | TaskRef):
            edge = Task2TaskEdge(
                source=DatasetId(task=TaskId(daskKeyRepr(v.key)), output=DefaultTaskOutput),
                sink_task=TaskId(key),
                sink_input_ps=i,
                sink_input_kw=None,
            )
            edges.append(edge)
        elif isinstance(v, Task):
            # TODO
            raise CascadeUserError("nested Task in dask graph args not yet implemented")
        else:
            instance.static_input_ps[f"{i}"] = v
    for k, v in task.kwargs.items():
        if isinstance(v, Alias | TaskRef):
            edge = Task2TaskEdge(
                source=DatasetId(task=TaskId(daskKeyRepr(v.key)), output=DefaultTaskOutput),
                sink_task=TaskId(key),
                sink_input_kw=k,
                sink_input_ps=None,
            )
            edges.append(edge)
        elif isinstance(v, Task):
            # TODO
            raise CascadeUserError("nested Task in dask graph kwargs not yet implemented")
        else:
            instance.static_input_kw[k] = v

    return instance, edges


def graph2job(dask: dict) -> JobInstance:
    task_nodes: dict[TaskId, TaskInstance] = {}
    edges = []

    ext_outputs = set()
    for node, value in dask.items():
        key = daskKeyRepr(node)
        if isinstance(value, DataNode):

            def provider() -> Any:
                return value.value

            task_nodes[TaskId(key)] = TaskBuilder.from_callable(provider)
            ext_outputs.add(DatasetId(task=TaskId(key), output=DefaultTaskOutput))
        elif isinstance(value, Task):
            node, _edges = task2task(key, value)
            task_nodes[TaskId(key)] = node
            edges.extend(_edges)
            ext_outputs.add(DatasetId(task=TaskId(key), output=DefaultTaskOutput))
            for edge in _edges:
                ext_outputs.discard(edge.source)
        elif isinstance(value, list | tuple | set):
            # TODO implement, consult further:
            # https://docs.dask.org/en/stable/spec.html
            # https://docs.dask.org/en/stable/custom-graphs.html
            # https://github.com/dask/dask/blob/main/dask/_task_spec.py#L829
            logger.warning("encountered nested container => confused ostrich")
            continue
        else:
            raise CascadeUserError(f"unsupported dask graph node type: {type(value)}")

    return JobInstance(
        tasks=task_nodes,
        edges=edges,
        ext_outputs=list(ext_outputs),
    )


def add(x: int, y: int) -> int:
    return x + y


def example_graph_execution() -> None:
    dask_graph = {"x": DataNode(None, 1), "y": DataNode(None, 2), "z": Task("z", add, TaskRef("x"), TaskRef("y"))}
    cascade_job = graph2job(dask_graph)
    outputs = run_locally(
        JobInstanceRich(jobInstance=cascade_job, checkpointSpec=None),
        workers=2,
        hosts=1,
        portBase=32345,
    )
    assert outputs == {DatasetId(TaskId("'z'"), DefaultTaskOutput): 3}


if __name__ == "__main__":
    example_graph_execution()
