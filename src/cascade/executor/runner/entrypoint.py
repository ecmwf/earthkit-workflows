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
from dataclasses import dataclass

import zmq

import cascade.executor.serde as serde
from cascade.executor.comms import callback
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.msg import (
    BackboneAddress,
    DatasetPublished,
    DatasetPurge,
    TaskFailure,
    TaskSequence,
    WorkerReady,
    WorkerShutdown,
)
from cascade.executor.runner.memory import Memory
from cascade.executor.runner.packages import PackagesEnv
from cascade.executor.runner.runner import ExecutionContext, run
from cascade.low.core import DatasetId, JobInstance, TaskId, WorkerId, type_dec
from cascade.low.tracing import label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerContext:
    """The static runner configuration"""

    workerId: WorkerId
    job: JobInstance
    callback: BackboneAddress
    param_source: dict[TaskId, dict[int | str, DatasetId]]
    log_base: str | None

    def project(self, taskSequence: TaskSequence) -> ExecutionContext:
        schema_lookup: dict[DatasetId, str] = {}
        for task_id, task_instance in self.job.tasks.items():
            for output, fqn in task_instance.definition.output_schema:
                schema_lookup[DatasetId(task_id, output)] = fqn

        param_source_ext: dict[TaskId, dict[int | str, tuple[DatasetId, str]]] = {
            task: {
                k: (dataset_id, schema_lookup[dataset_id])
                for k, dataset_id in self.param_source[task].items()
            }
            for task in taskSequence.tasks
        }
        return ExecutionContext(
            tasks={task: self.job.tasks[task] for task in taskSequence.tasks},
            param_source=param_source_ext,
            callback=self.callback,
            publish=taskSequence.publish,
        )


def worker_address(workerId: WorkerId) -> BackboneAddress:
    return f"ipc:///tmp/{repr(workerId)}.socket"


def execute_sequence(
    taskSequence: TaskSequence,
    memory: Memory,
    pckg: PackagesEnv,
    runnerContext: RunnerContext,
) -> None:
    taskId: TaskId | None = None
    try:
        executionContext = runnerContext.project(taskSequence)
        for taskId in taskSequence.tasks:
            pckg.extend(executionContext.tasks[taskId].definition.environment)
            run(taskId, executionContext, memory)
        memory.flush()
    except Exception as e:
        logger.exception("runner failure, about to report")
        callback(
            runnerContext.callback,
            TaskFailure(worker=taskSequence.worker, task=taskId, detail=repr(e)),
        )


def entrypoint(runnerContext: RunnerContext):
    if runnerContext.log_base:
        log_path = f"{runnerContext.log_base}.{runnerContext.workerId.worker}"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.bind(worker_address(runnerContext.workerId))
    callback(runnerContext.callback, WorkerReady(runnerContext.workerId))
    with (
        Memory(runnerContext.callback, runnerContext.workerId) as memory,
        PackagesEnv() as pckg,
    ):
        label("worker", repr(runnerContext.workerId))
        gpu_id = str(runnerContext.workerId.worker_num())
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_id)
        # NOTE check any(task.definition.needs_gpu) anywhere?
        # TODO configure OMP_NUM_THREADS, blas, mkl, etc -- not clear how tho

        for serdeTypeEnc, (serdeSer, serdeDes) in runnerContext.job.serdes.items():
            serde.SerdeRegistry.register(type_dec(serdeTypeEnc), serdeSer, serdeDes)

        availab_ds: set[DatasetId] = set()
        waiting_ts: TaskSequence | None = None
        missing_ds: set[DatasetId] = set()

        while True:
            mRaw = socket.recv()
            mDes = serde.des_message(mRaw)
            if isinstance(mDes, WorkerShutdown):
                logger.debug(f"worker {runnerContext.workerId} shutting down")
                break
            elif isinstance(mDes, DatasetPublished):
                availab_ds.add(mDes.ds)
                if mDes.ds in missing_ds:
                    missing_ds.remove(mDes.ds)
                    memory.provide(mDes.ds, "Any")
                    if waiting_ts is not None and (not missing_ds):
                        execute_sequence(waiting_ts, memory, pckg, runnerContext)
                        waiting_ts = None
            elif isinstance(mDes, DatasetPurge):
                memory.pop(mDes.ds)
                availab_ds.discard(mDes.ds)
            elif isinstance(mDes, TaskSequence):
                if waiting_ts is not None:
                    raise ValueError(
                        f"double task sequence enqueued: 1/ {waiting_ts}, 2/ {mDes}"
                    )
                required = {
                    dataset_id
                    for task in mDes.tasks
                    for dataset_id in runnerContext.param_source[task].values()
                } - {
                    DatasetId(task, key)
                    for task in mDes.tasks
                    for key, _ in runnerContext.job.tasks[task].definition.output_schema
                }
                missing_ds = required - availab_ds
                if missing_ds:
                    waiting_ts = mDes
                    for ds in availab_ds.intersection(required):
                        memory.provide(ds, "Any")
                else:
                    execute_sequence(mDes, memory, pckg, runnerContext)
            else:
                raise ValueError(f"unexpected message received: {type(mDes)}")
