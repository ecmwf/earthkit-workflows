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

from cascade.low.core import DatasetId, DefaultTaskOutput, JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance

logger = logging.getLogger(__name__)


def node2task(name: str, node: dict) -> tuple[TaskInstance, list[Task2TaskEdge]]:

    # TODO this is hotfix. Strict schema and the like required for payload
    if hasattr(node["payload"], "to_tuple"):
        payload_tuple = node["payload"].to_tuple()
    elif isinstance(node["payload"], tuple):
        payload_tuple = node["payload"]

    func_def: dict[str, Any] = (
        {"entrypoint": payload_tuple[0], "func": None}
        if isinstance(payload_tuple[0], str)
        else {
            "func": TaskDefinition.func_enc(cast(Callable, payload_tuple[0])),
            "entrypoint": "",
        }
    )
    args = cast(list[Any], payload_tuple[1])
    kwargs = cast(dict[str, Any], payload_tuple[2])
    metadata: dict[str, Any] = {}

    if len(payload_tuple) > 2:
        metadata = cast(dict[str, Any], payload_tuple[3])

    input_schema: dict[str, str] = {}
    for k in kwargs.keys():
        input_schema[k] = "Any"

    static_input_kw: dict[str, Any] = kwargs.copy()
    static_input_ps: dict[str, Any] = {}
    rev_lookup: dict[str, int] = {}
    for i, e in enumerate(args):
        static_input_ps[str(i)] = e
        # NOTE we may get a "false positive", ie, what is a genuine static string param ending up in rev_lookup
        # But it doesnt hurt, since we only pick `node["inputs"]` later on only.
        # Furthermore, we don't need rev lookup into kwargs since cascade fluent doesnt support that
        if isinstance(e, str):
            rev_lookup[e] = i
    edges = []
    for param, other in node["inputs"].items():
        if isinstance(other, str):
            source = DatasetId(other, DefaultTaskOutput)
        else:
            source = DatasetId(other[0], other[1])
        edges.append(
            Task2TaskEdge(
                source=source,
                sink_task=name,
                sink_input_ps=rev_lookup[param],
                sink_input_kw=None,
            )
        )
        static_input_ps[str(rev_lookup[param])] = None

    outputs = node["outputs"] if node["outputs"] else [DefaultTaskOutput]

    definition = TaskDefinition(
        **func_def,
        environment=cast(list[str], metadata.get("environment", [])),
        input_schema=input_schema,
        output_schema=[(e, "Any") for e in outputs],
        needs_gpu=cast(bool, metadata.get("needs_gpu", False)),
    )
    task = TaskInstance(
        definition=definition,
        static_input_kw=static_input_kw,
        static_input_ps=static_input_ps,
    )

    return task, edges


def graph2job(graph: dict) -> JobInstance:
    # graph assumed to be ekw.graph.serialise(ekw.graph.Graph)
    edges = []
    tasks = {}
    for node_name, node_val in graph.items():
        task, task_edges = node2task(node_name, node_val)
        edges += task_edges
        tasks[node_name] = task
    return JobInstance(tasks=tasks, edges=edges)
