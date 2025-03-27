"""
Lowering of the cascade.graph structures into cascade.low representation
"""

import logging
from typing import Any, Callable, cast

from cascade.graph import Graph, Node, serialise
from cascade.low.core import (
    DatasetId,
    JobInstance,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
)

logger = logging.getLogger(__name__)


def node2task(name: str, node: dict) -> tuple[TaskInstance, list[Task2TaskEdge]]:

    # TODO this is hotfix. Strict schema and the like required for payload
    if isinstance(node["payload"], tuple):
        func = cast(Callable, node["payload"][0])
        args = cast(list[Any], node["payload"][1])
        kwargs = cast(dict[str, Any], node["payload"][2])

        input_schema: dict[str, str] = {}
        for k in kwargs.keys():
            input_schema[k] = "Any"

        static_input_kw: dict[str, Any] = kwargs.copy()
        static_input_ps: dict[int, Any] = {}
        rev_lookup: dict[str, int] = {}
        for i, e in enumerate(args):
            static_input_ps[i] = e
            # NOTE we may get a "false positive", ie, what is a genuine static string param ending up in rev_lookup
            # But it doesnt hurt, since we only pick `node["inputs"]` later on only.
            # Furthermore, we don't need rev lookup into kwargs since cascade fluent doesnt support that
            if isinstance(e, str):
                rev_lookup[e] = i
        edges = []
        for param, other in node["inputs"].items():
            edges.append(
                Task2TaskEdge(
                    source=DatasetId(other, Node.DEFAULT_OUTPUT) if isinstance(other, str) else DatasetId(other[0], other[1]),
                    sink_task=name,
                    sink_input_ps=rev_lookup[param],
                    sink_input_kw=None,
                )
            )
            static_input_ps[rev_lookup[param]] = None

        outputs = node["outputs"] if node["outputs"] else [Node.DEFAULT_OUTPUT]

        definition = TaskDefinition(
            func=TaskDefinition.func_enc(func),
            environment=[],
            entrypoint="",
            input_schema=input_schema,
            output_schema={e: "Any" for e in outputs},
        )
        task = TaskInstance(
            definition=definition,
            static_input_kw=static_input_kw,
            static_input_ps=static_input_ps,
        )
    else:
        raise NotImplementedError(node["payload"])

    return task, edges


def graph2job(graph: Graph) -> JobInstance:
    ser = serialise(graph)  # simpler
    edges = []
    tasks = {}
    for node_name, node_val in ser.items():
        task, task_edges = node2task(node_name, node_val)
        edges += task_edges
        tasks[node_name] = task
    return JobInstance(tasks=tasks, edges=edges)
