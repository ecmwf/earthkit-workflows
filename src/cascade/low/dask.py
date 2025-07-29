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
from cascade.low.core import DatasetId, JobInstance, Task2TaskEdge, TaskInstance
from earthkit.workflows.graph import Node

logger = logging.getLogger(__name__)


def daskKeyRepr(key: str | int | float | tuple) -> str:
    return repr(key)


def task2task(key: str, task: Task) -> tuple[TaskInstance, list[Task2TaskEdge]]:
    instance = TaskBuilder.from_callable(task.func)
    edges: list[Task2TaskEdge] = []

    for i, v in enumerate(task.args):
        if isinstance(v, Alias | TaskRef):
            edge = Task2TaskEdge(
                source=DatasetId(task=daskKeyRepr(v.key), output=Node.DEFAULT_OUTPUT),
                sink_task=key,
                sink_input_ps=str(i),
                sink_input_kw=None,
            )
            edges.append(edge)
        elif isinstance(v, Task):
            # TODO
            raise NotImplementedError
        else:
            instance.static_input_ps[f"{i}"] = v
    for k, v in task.kwargs.items():
        if isinstance(v, Alias | TaskRef):
            edge = Task2TaskEdge(
                source=DatasetId(task=daskKeyRepr(v.key), output=Node.DEFAULT_OUTPUT),
                sink_task=key,
                sink_input_kw=k,
                sink_input_ps=None,
            )
            edges.append(edge)
        elif isinstance(v, Task):
            # TODO
            raise NotImplementedError
        else:
            instance.static_input_kw[k] = v

    return instance, edges


def graph2job(dask: dict) -> JobInstance:
    task_nodes = {}
    edges = []

    for node, value in dask.items():
        key = daskKeyRepr(node)
        if isinstance(value, DataNode):

            def provider() -> Any:
                return value.value

            task_nodes[key] = TaskBuilder.from_callable(provider)
        elif isinstance(value, Task):
            node, _edges = task2task(key, value)
            task_nodes[key] = node
            edges.extend(_edges)
        elif isinstance(value, list | tuple | set):
            # TODO implement, consult further:
            # https://docs.dask.org/en/stable/spec.html
            # https://docs.dask.org/en/stable/custom-graphs.html
            # https://github.com/dask/dask/blob/main/dask/_task_spec.py#L829
            logger.warning("encountered nested container => confused ostrich")
            continue
        else:
            raise NotImplementedError

    return JobInstance(
        tasks=task_nodes,
        edges=edges,
    )
