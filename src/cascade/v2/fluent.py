"""
Adapter for the fluent-based definition into the core graph definition
"""

from typing import Any, cast

from cascade.graph import Graph, serialise
from cascade.schedulers.schedule import Schedule as FluentSchedule
from cascade.v2.core import (
    JobInstance,
    Schedule,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
)


def cascade_func_wrap(**kwargs) -> Any:
    kwargs_rl = kwargs["kwargs"]
    func = TaskDefinition.func_dec(kwargs["func"])
    argpos = kwargs["argpos"]
    argdyn = kwargs["argdyn"]
    args_rl = [None] * (len(argpos) + len(argdyn))
    for k, v in argpos.items():
        args_rl[k] = v
    for k, v in argdyn.items():
        # NOTE not exactly sure why we need `int(k)` here -- some quirk of in-fiab serdes?
        args_rl[int(k)] = kwargs[v]
    return func(*args_rl, **kwargs_rl)


def node2task(name: str, node: dict) -> tuple[TaskInstance, list[Task2TaskEdge]]:
    edges = []
    for param, other in node["inputs"].items():
        edges.append(
            Task2TaskEdge(
                source_task=other,
                source_output="__default__",
                sink_task=name,
                sink_input="dynamic_" + param,
            )
        )

    # TODO this is hotfix. Strict schema and the like required for payload
    if isinstance(node["payload"], tuple):
        input_schema = {
            "kwargs": "dict",
            "func": "str",
            "argpos": "dict",
            "argdyn": "dict",
        }
        for edge in edges:
            input_schema[edge.sink_input] = "Any"
        argdyn = {}
        argpos = {}
        for i, pname in enumerate(node["payload"][1]):
            if param in node["inputs"]:
                argdyn[i] = "dynamic_" + param
            else:
                argpos[i] = param

        definition = TaskDefinition(
            func=TaskDefinition.func_enc(cascade_func_wrap),
            environment=[],
            entrypoint="",
            input_schema=input_schema,
            output_schema={e: "Any" for e in node["outputs"]},
        )
        task = TaskInstance(
            definition=definition,
            static_input={
                "kwargs": node["payload"][2],
                "argpos": argpos,
                "argdyn": argdyn,
                "func": TaskDefinition.func_enc(node["payload"][0]),
            },
        )
    elif isinstance(node["payload"], dict) and node["payload"].keys() == TaskInstance.model_fields.keys():
        # NOTE this doesnt really work -- the edges are broken afterwards
        task = TaskInstance(**node["payload"])

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
