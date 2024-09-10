"""
Adapter for the fluent-based definition into the core graph definition
"""

from typing import Any

from cascade.graph import Graph, serialise
from cascade.v2.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance


def cascade_func_wrap(**kwargs) -> Any:
    kwargs_rl = kwargs["kwargs"]
    func = kwargs["func"]
    argpos = kwargs["argpos"]
    argdyn = kwargs["argdyn"]
    args_rl = [None] * (len(argpos) + len(argdyn))
    for k, v in argpos.items():
        args_rl[k] = v
    for k, v in argdyn.items():
        args_rl[k] = kwargs[v]
    return func(*args_rl, **kwargs_rl)


def node2task(name: str, node: dict) -> tuple[TaskInstance, list[Task2TaskEdge]]:
    edges = []
    input_schema = {
        "kwargs": "dict",
        "func": "Callable",
        "argpos": "dict",
        "argdyn": "dict",
    }
    for param, other in node["inputs"].items():
        edges.append(
            Task2TaskEdge(
                source_task=other,
                source_output="__default__",
                sink_task=name,
                sink_input="dynamic_" + param,
            )
        )
        input_schema["dynamic_" + param] = "Any"

    argdyn = {}
    argpos = {}
    for i, pname in enumerate(node["payload"][1]):
        if param in node["inputs"]:
            argdyn[i] = "dynamic_" + param
        else:
            argpos[i] = param

    definition = TaskDefinition(
        func=cascade_func_wrap,
        environment="",
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
            "func": node["payload"][0],
        },
    )
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
