# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The entrypoint itself"""

import logging
import logging.config
import os

import zmq
from packaging.version import Version

import cascade.executor.platform as platform
import cascade.executor.serde as serde
from cascade.deployment.logging import init_from_obj
from cascade.executor.comms import callback, worker_address
from cascade.executor.msg import (
    BackboneAddress,
    DatasetPublished,
    DatasetPurge,
    RunnerRestartRequest,
    TaskFailure,
    TaskSequence,
    WorkerReady,
    WorkerShutdown,
)
from cascade.executor.runner.memory import Memory
from cascade.executor.runner.packages import PackagesEnv, PkgInstallException
from cascade.executor.runner.runner import run
from cascade.executor.runner.setup import RunnerContext, WorkerSetup, load_runner_ctx_from_shm
from cascade.low.core import DatasetId, TaskId, WorkerId, type_dec
from cascade.low.exceptions import CascadeError, CascadeInfrastructureError, CascadeInternalError, CascadeUserError, ser
from cascade.low.tracing import label
from cascade.ygg.transport import ensure_clean_zmq_state, get_context

logger = logging.getLogger(__name__ if __name__ != "__main__" else "cascade.executor.runner.entrypoint")


def task_sequence_postmortem(ctx: RunnerContext, taskSequence: TaskSequence, cut: TaskId) -> list[tuple[DatasetId, str]]:
    """Assuming a failure at task Cut, identify which datasets from the beginning of
    the sequence should be additionaly published. Returns datasetid + its type"""
    finished = set()
    required = set()
    found = False
    for task in taskSequence.tasks:
        if task == cut or found:
            found = True
            for sourceDataset in ctx.param_source[task].values():
                if sourceDataset.task in finished:
                    required.add(sourceDataset)
        else:
            finished.add(task)
    additionalPublish = required - taskSequence.publish
    return [(ds, dict(ctx.job.tasks[ds.task].definition.output_schema)[ds.output]) for ds in additionalPublish]


def task_sequence_remainder(taskSequence: TaskSequence, cut: TaskId) -> TaskSequence:
    """Assuming a failure at task Cut, calculate new task sequence which starts with Cut
    that represents the still-to-be-done-in-new-worker calculation"""
    remainder = []
    for task in taskSequence.tasks:
        if task == cut or remainder:
            remainder.append(task)
    if not remainder:
        raise CascadeInternalError(f"empty remainder -> task {cut} not part of the orig {taskSequence=}?")
    remainder_set = set(remainder)

    return TaskSequence(
        worker=taskSequence.worker,
        tasks=remainder,
        publish={ds for ds in taskSequence.publish if ds.task in remainder_set},
        extra_env=taskSequence.extra_env,
    )


class Config:
    """Some parameters to drive behaviour. Currently not exposed externally -- no clear argument
    that they should be. As is, just a means of code experimentation.
    """

    # flushing approach -- when we finish a computation of task sequence, there is a question what
    # to do with the output. We could either publish & drop, or publish and retain in memory. The
    # former is is slower -- if the next task sequence needs this output, it requires a fetch & deser
    # from cashme. But the latter is more risky -- we effectively have the same dataset twice in
    # system memory. The `posttask_flush` below goes the former way, the `pretask_flush` is a careful
    # way of latter -- we drop the output from memory only if the *next* task sequence does not need
    # it, ie, we retain a cache of age 1. We could ultimately have controller decide about this, or
    # decide dynamically based on memory pressure -- but neither is easy.
    posttask_flush = False  # after task is done, drop all outputs from memory
    pretask_flush = True  # when we receive a task, we drop those in memory that wont be needed


def execute_sequence(
    taskSequence: TaskSequence,
    memory: Memory,
    pckg: PackagesEnv,
    runnerContext: RunnerContext,
) -> bool:
    """Returns whether ended successfully. If not, it means a failure callback
    was issued, and the outer loop should only wait for WorkerShutdown message."""
    taskId: TaskId | None = None
    try:
        for key, value in taskSequence.extra_env:
            os.environ[key] = value
        executionContext = runnerContext.project(taskSequence)
        for taskId in taskSequence.tasks:
            pckg.extend(executionContext.tasks[taskId].definition.environment)
            run(taskId, executionContext, memory)
        if Config.posttask_flush:
            memory.flush()
        for key, _ in taskSequence.extra_env:
            # NOTE we should in principle restore the previous value, but we dont expect collisions
            del os.environ[key]
        return True
    except PkgInstallException as e:
        if e.was_clean:
            logger.error(f"pkg install failed on a clean env, crashing with user error")
            callback(
                runnerContext.callback,
                TaskFailure(worker=taskSequence.worker, task=taskId, detail=ser(CascadeUserError(repr(e)))),
            )
            return False
        logger.error(f"pkg install failed, will send RunnerRestartRequest: {repr(e)}")
        if not taskId:
            raise CascadeInternalError("Postinstall should not have been raised in the absence of active task")
        additionalPublish = task_sequence_postmortem(runnerContext, taskSequence, taskId)
        logger.debug(f"postinstall failure triggers additional publish of {additionalPublish}")
        remainder = task_sequence_remainder(taskSequence, taskId)
        memory.additional_publish_local(additionalPublish)
        callback(
            runnerContext.callback,
            RunnerRestartRequest(worker=taskSequence.worker, remainder=remainder),
        )
        return False
    except Exception as e:
        logger.exception("runner failure, about to report, propagating as TaskFailure")
        callback(
            runnerContext.callback,
            TaskFailure(worker=taskSequence.worker, task=taskId, detail=ser(e)),
        )
        return False


def entrypoint(workerSetup: WorkerSetup, runnerContext: RunnerContext) -> None:
    """Main runner loop for a worker process."""
    init_from_obj(runnerContext.loggingConfig, f"worker_{workerSetup.workerId.worker}")
    ensure_clean_zmq_state()
    ctx = get_context()
    socket = ctx.socket(zmq.PULL)
    address = worker_address(workerSetup.workerId, workerSetup.workerAttemptCnt)
    logger.debug(f"worker {workerSetup.workerId} binding to {address=}")
    socket.bind(address)
    callback(runnerContext.callback, WorkerReady(workerSetup.workerId))
    logger.debug(f"worker {workerSetup.workerId} sent WorkerReady")

    with (
        Memory(runnerContext.callback, workerSetup.workerId) as memory,
        PackagesEnv({k: Version(v) for k, v in workerSetup.initial_installed.items()}, runnerContext.pip_indices) as pckg,
    ):
        label("host", workerSetup.workerId.host)
        label("worker", repr(workerSetup.workerId))
        worker_num = workerSetup.workerId.worker_num()
        platform.gpu_init(worker_num)
        # TODO configure OMP_NUM_THREADS, blas, mkl, etc -- not clear how tho

        for serdeTypeEnc, (serdeSer, serdeDes) in runnerContext.job.serdes.items():
            serde.SerdeRegistry.register(type_dec(serdeTypeEnc), serdeSer, serdeDes)

        availab_ds: set[DatasetId] = set()
        waiting_ts: TaskSequence | None = None
        missing_ds: set[DatasetId] = set()
        isTerminating = False

        while True:
            mRaw = socket.recv()
            mDes = serde.des_message(mRaw)
            if isinstance(mDes, WorkerShutdown):
                logger.debug(f"worker {workerSetup.workerId} shutting down")
                break
            elif isTerminating:
                logger.warning(f"ignoring message {mDes} because terminating")
                continue
            elif isinstance(mDes, DatasetPublished):
                availab_ds.add(mDes.ds)
                if mDes.ds in missing_ds:
                    missing_ds.remove(mDes.ds)
                    memory.provide(mDes.ds, "Any")
                    if waiting_ts is not None and (not missing_ds):
                        isTerminating = not execute_sequence(waiting_ts, memory, pckg, runnerContext)
                        waiting_ts = None
            elif isinstance(mDes, DatasetPurge):
                memory.pop(mDes.ds)
                availab_ds.discard(mDes.ds)
            elif isinstance(mDes, TaskSequence):
                if waiting_ts is not None:
                    raise CascadeInternalError(f"double task sequence enqueued: 1/ {waiting_ts}, 2/ {mDes}")
                required = {dataset_id for task in mDes.tasks for dataset_id in runnerContext.param_source[task].values()} - {
                    DatasetId(task, key) for task in mDes.tasks for key, _ in runnerContext.job.tasks[task].definition.output_schema
                }
                missing_ds = required - availab_ds
                logger.debug(f"received task sequence {mDes}, and missing {missing_ds}")
                if Config.pretask_flush:
                    extraneous_ds = availab_ds - required
                    memory.flush(extraneous_ds)
                if missing_ds:
                    waiting_ts = mDes
                    for ds in availab_ds.intersection(required):
                        memory.provide(ds, "Any")
                else:
                    isTerminating = not execute_sequence(mDes, memory, pckg, runnerContext)
            else:
                raise CascadeInternalError(f"unexpected message received: {type(mDes)}")


if __name__ == "__main__":
    from cascade.executor.runner.setup import WORKER_SETUP_ENVVAR, WorkerSetup, load_runner_ctx_from_shm

    _setup_str = os.environ[WORKER_SETUP_ENVVAR]
    _worker_setup = WorkerSetup.from_str(_setup_str)
    _runner_ctx = load_runner_ctx_from_shm(_worker_setup.shm_key)

    entrypoint(_worker_setup, _runner_ctx)
