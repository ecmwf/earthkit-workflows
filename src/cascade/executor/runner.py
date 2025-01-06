"""
This module is responsible for running actual tasks -- a thin wrapper over a Callable that handles:
 - getting & deserializing the inputs
 - serializing & storing the outputs
 - invoking callback to report task success, dataset publication, failures
 - setting up the environment: packages and envvars

Short-lived process -- lifetime does not extend over that of the task itself. Is spawned by `executor`.
It can execute a single task, or a fixed sequence of tasks to save some serde.
"""

# NOTE there are a few performance optimisations at hand:
# - we could start obtaining inputs while we are venv-installing (but beware deser needing venv!)
# - we could start venv-installing & inputs-obtaining while previous task is running
# - we could be inputs-obtaining in parallel
# Ideally, all deser would be doable outside Python -- then this whole module could be eg rust & threads 

from typing import Callable, Any, Literal
import importlib
import hashlib
from contextlib import AbstractContextManager
import os
import logging
from time import perf_counter_ns
import tempfile
import subprocess
import sys

from cascade.low.core import TaskDefinition, DatasetId, TaskId, WorkerId, NO_OUTPUT_PLACEHOLDER
from cascade.low.func import ensure, assert_never, maybe_head
from cascade.executor.msg import BackboneAddress, ExecutionContext, DatasetPublished, TaskFailure, TaskSuccess, TaskSequence
from cascade.executor.comms import callback
import cascade.shm.client as shm_client
from cascade.low.tracing import mark, label, TaskLifecycle, Microtrace, timer, trace
import cascade.executor.serde as serde

logger = logging.getLogger(__name__)

def ds2shmid(ds: DatasetId) -> str:
    # we cant use too long file names for shm, https://trac.macports.org/ticket/64806
    h = hashlib.new("md5", usedforsecurity=False)
    h.update((ds.task + ds.output).encode())
    return h.hexdigest()[:24]

class Memory(AbstractContextManager):
    def __init__(self, callback: BackboneAddress, worker: WorkerId, publish: set[DatasetId]) -> None:
        self.local: dict[DatasetId, Any] = {}
        self.bufs: dict[DatasetId, shm_client.AllocatedBuffer] = {}
        self.publish = publish
        self.callback = callback
        self.worker = worker

    def handle(self, outputId: DatasetId, outputSchema: str, outputValue: Any) -> None:
        if outputId == NO_OUTPUT_PLACEHOLDER:
            if outputValue is not None:
                logger.warning(f"gotten output of type {type(outputValue)} where none was expected, updating annotation")
                outputSchema = "Any"
            else:
                outputValue = "ok"

        # TODO how do we purge from here over time?
        self.local[outputId] = outputValue

        if outputId in self.publish:
            logger.debug(f"publishing {outputId}")
            shmid = ds2shmid(outputId)
            result_ser = timer(serde.ser_output, Microtrace.wrk_ser)(outputValue, outputSchema)
            l = len(result_ser)
            rbuf = shm_client.allocate(shmid, l)
            rbuf.view()[:l] = result_ser
            rbuf.close()
            callback(
                self.callback,
                DatasetPublished(ds=outputId, host=self.worker.host), # type: ignore # TODO dct
            )

    def provide(self, inputId: DatasetId, annotation: str) -> Any:
        if inputId not in self.local:
            if inputId in self.bufs:
                raise ValueError(f"internal data corruption for {inputId}")
            shmid = ds2shmid(inputId)
            logger.debug(f"asking for {inputId} via {shmid}")
            buf = shm_client.get(shmid)
            self.bufs[inputId] = buf
            self.local[inputId] = timer(serde.des_output, Microtrace.wrk_deser)(buf.view(), annotation)

        return self.local[inputId]

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        # TODO allow for purging via ext events -- drop from local, close in bufs

        # this is required so that the Shm can be properly freed, otherwise you get 'pointers cannot be closed'
        del self.local
        for buf in self.bufs.values():
            buf.close()
        return False

class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory|None = None

    def extend(self, packages: list[str]) -> None:
        if not packages:
            return
        if self.td is None:
            logger.debug("creating a new venv")
            self.td = tempfile.TemporaryDirectory()
            venv_command = ["uv", "venv", self.td.name]
            # NOTE we create a venv instead of just plain directory, because some of the packages create files
            # outside of site-packages. Thus we then install with --prefix, not with --target
            subprocess.run(venv_command, check=True)

        logger.debug(f"installing {len(packages)} packages: {','.join(packages[:3])}{',...' if len(packages) > 3 else ''}")
        install_command = ["uv", "pip", "install", "--prefix", self.td.name]
        if os.environ.get("VENV_OFFLINE", "") == "YES":
            install_command += ["--offline"]
        if cache_dir := os.environ.get("VENV_CACHE", ""):
            install_command += ["--cache-dir", cache_dir]
        install_command.extend(set(packages))
        subprocess.run(install_command, check=True)
        # TODO somehow fragile -- try to obtain it from {td.name}/bin/activate?
        sys.path.append(f"{self.td.name}/lib/python3.11/site-packages/")

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self.td is not None:
            self.td.cleanup()
        return False

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
                    TaskSuccess(worker=taskSequence.worker, ts=taskId), # type: ignore # TODO dct
                )
    except Exception as e:
        logger.exception("runner failure, about to report")
        callback(
            executionContext.callback,
            TaskFailure(worker=taskSequence.worker, task=taskId, detail=repr(e)), # type: ignore # TODO dct
        )
