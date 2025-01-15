"""
The entrypoint itself
"""

# NOTE there are a few performance optimisations at hand:
# - we could start obtaining inputs while we are venv-installing (but beware deser needing venv!)
# - we could start venv-installing & inputs-obtaining while previous task is running
# - we could be inputs-obtaining in parallel
# Ideally, all deser would be doable outside Python -- then this whole module could be eg rust & threads 

from typing import Callable, Any
import importlib
import os
import logging
import logging.config
from time import perf_counter_ns

from cascade.low.core import TaskDefinition, DatasetId, TaskId
from cascade.low.func import ensure, assert_never, maybe_head
from cascade.executor.msg import ExecutionContext, TaskFailure, TaskSuccess, TaskSequence
from cascade.executor.comms import callback
from cascade.low.tracing import mark, label, TaskLifecycle, Microtrace, trace
from cascade.executor.config import logging_config
from cascade.executor.runner.memory import Memory
from cascade.executor.runner.packages import PackagesEnv

logger = logging.getLogger(__name__)

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
        module_name, function_name = task.definition.entrypoint.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = module.__dict__[function_name]
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

    if len(task.definition.output_schema) > 1:
        raise NotImplementedError("multiple outputs not supported yet")
        # NOTE to implement, just put `result=func` & `memory.handle` below into a for-cycle for generator outputs
    outputId, outputSchema = maybe_head(task.definition.output_schema.items()) # type: ignore
    if not outputId:
        raise ValueError(f"no output key for task {taskId}")
    mark({"task": taskId, "action": TaskLifecycle.loaded})
    prep_end = perf_counter_ns()

    # invoke
    result = func(*args, **kwargs)
    mark({"task": taskId, "action": TaskLifecycle.computed})
    run_end = perf_counter_ns()

    # store outputs
    memory.handle(DatasetId(taskId, outputId), outputSchema, result)
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


def entrypoint(taskSequence: TaskSequence, executionContext: ExecutionContext): 
    taskId: TaskId|None = None
    try:
        logging.config.dictConfig(logging_config)
        with Memory(executionContext.callback, taskSequence.worker, taskSequence.publish) as memory, PackagesEnv() as pckg:
            label("worker", repr(taskSequence.worker))

            if any(task.definition.needs_gpu for task in executionContext.tasks.values()):
                gpu_id = str(taskSequence.worker.worker_num())
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_id)
            # TODO configure OMP_NUM_THREADS, blas, mkl, etc -- not clear how tho

            for taskId in taskSequence.tasks:
                pckg.extend(executionContext.tasks[taskId].definition.environment)
                run(taskId, executionContext, memory)
                callback(
                    executionContext.callback,
                    TaskSuccess(worker=taskSequence.worker, ts=taskId),
                )
    except Exception as e:
        logger.exception("runner failure, about to report")
        callback(
            executionContext.callback,
            TaskFailure(worker=taskSequence.worker, task=taskId, detail=repr(e)),
        )
