"""
Adapter of the core graph definition into dask-based execution
"""

import importlib
from typing import Any, Callable

import dask.utils

from cascade.v2.core import JobInstance, TaskInstance, TaskDefinition
from cascade.v2.views import param_source


def dask_func_wrap(*args, **kwargs) -> Any:
    # the problem with dask is that it does not do dataset substitution in kwargs
    func = kwargs["func"]
    arg_names = kwargs["arg_names"]
    kwargs_rl = kwargs["kwargs"]
    kwargs_rl.update({k: v for k, v in zip(arg_names, args)})
    return func(**kwargs_rl)


DaskPayload = tuple[Callable, Callable, list[Any], dict[str, Any]]


def task2dask(
    task: TaskInstance, input2source: dict[str, tuple[str, str]]
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

    dynamic_inputs = [e for e in task.definition.input_schema if e in input2source]
    kwargs = {
        "func": func,
        "kwargs": task.static_input,
        "arg_names": dynamic_inputs,
    }

    return (
        dask.utils.apply,
        dask_func_wrap,
        [input2source[e][0] for e in dynamic_inputs],
        kwargs,
    )


DaskJob = dict


def job2dask(job: JobInstance) -> DaskJob:
    task_param_sources = param_source(job.edges)
    for name, instance in job.tasks.items():
        if len(instance.definition.output_schema) > 1:
            # dask supports only single output rn
            raise NotImplementedError(name, instance.definition)

    return {
        name: task2dask(definition, task_param_sources[name])
        for name, definition in job.tasks.items()
    }
