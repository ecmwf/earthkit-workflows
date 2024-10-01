"""
Controller-less execution, via Dask Delayed
"""

# TODO move to cascade.executor, or cascade-dask package

import importlib
from typing import Any, Callable

import dask.utils

from cascade.low.core import JobInstance, TaskDefinition, TaskInstance
from cascade.low.func import ensure
from cascade.low.views import param_source

DaskPayload = tuple[Callable, Callable, list[Any], dict[str, Any]]


def task2delayed(
    task: TaskInstance, input2source: dict[int | str, tuple[str, str]]
) -> DaskPayload:
    if task.definition.environment:
        raise NotImplementedError(task.definition.environment)

    if task.definition.func:
        func = TaskDefinition.func_dec(task.definition.func)
    elif "." in task.definition.entrypoint:
        module_name, function_name = task.definition.entrypoint.rsplit(".", 1)
        func = importlib.import_module(module_name).__dict__[function_name]
    else:
        function_name = task.definition.entrypoint
        func = eval(function_name)

    args: list[Any] = []
    for k, v in task.static_input_ps.items():
        ensure(args, k)
        args[k] = v
    kwargs: dict[str, Any] = task.static_input_kw.copy()
    for s, (onode, _) in input2source.items():
        if isinstance(s, str):
            raise NotImplementedError("dask doesnt support kwargs dyn args")
        elif isinstance(s, int):
            ensure(args, s)
            args[s] = onode

    rv = (
        dask.utils.apply,
        func,
        args,
        kwargs,
    )
    return rv


DaskJob = dict


def job2delayed(job: JobInstance) -> DaskJob:
    task_param_sources = param_source(job.edges)
    for name, instance in job.tasks.items():
        if len(instance.definition.output_schema) > 1:
            # dask supports only single output rn
            raise NotImplementedError(name, instance.definition)

    return {
        name: task2delayed(definition, task_param_sources[name])
        for name, definition in job.tasks.items()
    }
