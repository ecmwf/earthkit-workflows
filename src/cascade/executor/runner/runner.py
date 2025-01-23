"""
Thin wrapper over a single Task/Callable

Just io handling (ie, using Memory api), tracing and callable invocation
"""

import logging
import importlib
from typing import Callable, Any
from time import perf_counter_ns
from dataclasses import dataclass

from cascade.low.core import TaskDefinition, DatasetId, TaskId, TaskInstance
from cascade.low.func import ensure, assert_never, assert_iter_empty, resolve_callable
from cascade.executor.msg import TaskSequence, BackboneAddress
from cascade.executor.comms import callback
from cascade.low.tracing import mark, TaskLifecycle, Microtrace, trace
from cascade.executor.runner.memory import Memory

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExecutionContext:
    """A projection of JobInstance relevant to particular TaskSequence"""
    # NOTE once we have long lived workers, this would be replaced by full JobInstance present at the worker
    tasks: dict[TaskId, TaskInstance]
    # int: pos argument, str: kw argument. Values are dsid + annotation
    param_source: dict[TaskId, dict[int|str, tuple[DatasetId, str]]]
    callback: BackboneAddress
    publish: set[DatasetId]

def run(taskId: TaskId, executionContext: ExecutionContext, memory: Memory) -> None:
    start = perf_counter_ns()
    task = executionContext.tasks[taskId]
    mark({"task": taskId, "action": TaskLifecycle.started})
    logger.debug(f"starting {taskId}")

    # prepare func & inputs
    func: Callable
    if task.definition.func is not None:
        func = TaskDefinition.func_dec(task.definition.func)
    elif task.definition.entrypoint is not None:
        func = resolve_callable(task.definition.entrypoint)
    else:
        raise TypeError("neither entrypoint nor func given")

    args: list[Any] = []
    for idx, arg in task.static_input_ps.items():
        ensure(args, idx)
        args[idx] = arg
    kwargs: dict[str, Any] = {}
    kwargs.update(task.static_input_kw)

    for param_pos, (dataset_id, annotation) in executionContext.param_source[taskId].items():
        value = memory.provide(dataset_id, annotation)
        if isinstance(param_pos, str):
            kwargs[param_pos] = value
        elif isinstance(param_pos, int):
            ensure(args, param_pos)
            args[param_pos] = value
        else:
            assert_never(param_pos)

    outputs = list(task.definition.output_schema.items())
    outputs.sort()
    outputsN = len(outputs)
    if outputsN == 0:
        raise ValueError(f"no output key for task {taskId}")
    mark({"task": taskId, "action": TaskLifecycle.loaded})
    prep_end = perf_counter_ns()

    # invoke
    result = func(*args, **kwargs)
    if outputsN == 1:
        mark({"task": taskId, "action": TaskLifecycle.computed})
        run_end = perf_counter_ns()

    # store outputs
    if outputsN == 1:
        outputKey, outputSchema = outputs[0]
        outputId = DatasetId(taskId, outputKey)
        memory.handle(outputId, outputSchema, result, outputId in executionContext.publish)
        mark({"task": taskId, "action": TaskLifecycle.published})
    else:
        outputsI = iter(outputs)
        for (outputKey, outputSchema), outputValue in zip(outputsI, result):
            outputId = DatasetId(taskId, outputKey)
            memory.handle(outputId, outputSchema, outputValue, outputId in executionContext.publish)
        if not assert_iter_empty(outputsI):
            raise ValueError(f"schema declared more outputs than there were results")
        if not assert_iter_empty(result):
            raise ValueError(f"function produced more results than there were schema outputs")
        # in principle, we should mark computed & calc run_end prior to ultimate publish, but imo not worth it
        mark({"task": taskId, "action": TaskLifecycle.computed})
        run_end = perf_counter_ns()
        mark({"task": taskId, "action": TaskLifecycle.published})
    end = perf_counter_ns()

    trace(Microtrace.wrk_task, end - start)
    logger.debug(f"outer elapsed {(end-start)/1e9: .5f} s in {taskId}")
    trace(Microtrace.wrk_load, prep_end - start)
    logger.debug(f"prep elapsed {(prep_end-start)/1e9: .5f} s in {taskId}")
    trace(Microtrace.wrk_compute, run_end - prep_end)
    logger.debug(f"inner elapsed {(run_end-prep_end)/1e9: .5f} s in {taskId}")
    trace(Microtrace.wrk_publish, end - run_end)
    logger.debug(f"post elapsed {(end-run_end)/1e9: .5f} s in {taskId}")
