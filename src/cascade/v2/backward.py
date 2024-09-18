"""
Backward-compatibility adapter from the v1/graph definition into the v2/core definition
"""

from typing import Any, Callable, cast

from cascade.graph import Graph, Node, serialise
from cascade.schedulers.schedule import Schedule as FluentSchedule
from cascade.v2.core import (
    JobInstance,
    Schedule,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
)


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
            rev_lookup[e] = i
        edges = []
        for param, other in node["inputs"].items():
            edges.append(
                Task2TaskEdge(
                    source_task=other,
                    source_output=Node.DEFAULT_OUTPUT,
                    sink_task=name,
                    sink_input_ps=rev_lookup[param],
                    sink_input_kw=None,
                )
            )
            static_input_ps[i] = None

        if node["outputs"] != [Node.DEFAULT_OUTPUT] and node["outputs"]:
            raise NotImplementedError("multiple outputs are not supported yet")

        definition = TaskDefinition(
            func=TaskDefinition.func_enc(func),
            environment=[],
            entrypoint="",
            input_schema=input_schema,
            output_schema={e: "Any" for e in node["outputs"]},
        )
        task = TaskInstance(
            definition=definition,
            static_input_kw=static_input_kw,
            static_input_ps=static_input_ps,
        )
    elif (
        isinstance(node["payload"], dict)
        and node["payload"].keys() == TaskInstance.model_fields.keys()
    ):
        # NOTE this doesnt really work as edges arent covered
        task = TaskInstance(**node["payload"])
        edges = []

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


def schedule2schedule(fluent_schedule: FluentSchedule) -> tuple[JobInstance, Schedule]:
    job_instance = graph2job(fluent_schedule)
    schedule = Schedule(
        host_task_queues={
            cast(str, k): cast(list[str], list(v))
            for k, v in fluent_schedule.task_allocation.items()
        }
    )
    return job_instance, schedule
