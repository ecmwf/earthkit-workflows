# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lowering of the earthkit.workflows.graph structures into cascade.low representation"""

import logging
from typing import Any, Callable, cast

from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstance, Task2TaskEdge, TaskDefinition, TaskId, TaskInstance

logger = logging.getLogger(__name__)


def node2task(name: str, node: dict) -> tuple[TaskInstance, list[Task2TaskEdge]]:
    task = node["payload"]
    edges = []
    for index, other in node["inputs"].items():
        if isinstance(other, str):
            source = DatasetId(TaskId(other), DefaultTaskOutput)
        else:
            source = DatasetId(TaskId(other[0]), other[1])
        edges.append(
            Task2TaskEdge(
                source=source,
                sink_task=TaskId(name),
                sink_input_ps=int(index),
                sink_input_kw=None,
            )
        )
    return task, edges


def graph2job(graph: dict) -> JobInstance:
    # graph assumed to be ekw.graph.serialise(ekw.graph.Graph)
    edges = []
    tasks: dict[TaskId, TaskInstance] = {}
    for node_name, node_val in graph.items():
        task, task_edges = node2task(node_name, node_val)
        edges += task_edges
        tasks[TaskId(node_name)] = task
    return JobInstance(tasks=tasks, edges=edges)
