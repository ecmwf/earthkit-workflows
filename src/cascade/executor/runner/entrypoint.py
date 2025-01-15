"""
The entrypoint itself
"""

import os
import logging
import logging.config
from dataclasses import dataclass
import zmq

from cascade.low.core import WorkerId, JobInstance, TaskId, DatasetId
from cascade.executor.msg import TaskFailure, TaskSequence, BackboneAddress, TaskSuccess, WorkerReady, WorkerShutdown
from cascade.executor.comms import callback
from cascade.low.tracing import label
from cascade.executor.config import logging_config
from cascade.executor.runner.memory import Memory
from cascade.executor.runner.packages import PackagesEnv
from cascade.executor.runner.runner import ExecutionContext, run
import cascade.executor.serde as serde

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RunnerContext:
    """The static runner configuration"""
    workerId: WorkerId
    job: JobInstance
    callback: BackboneAddress
    param_source: dict[TaskId, dict[int | str, DatasetId]]

    def project(self, taskSequence: TaskSequence) -> ExecutionContext:
        param_source_ext: dict[TaskId, dict[int|str, tuple[DatasetId, str]]] = {
            task: {
                k: (
                    dataset_id, 
                    self.job.tasks[dataset_id.task].definition.output_schema[dataset_id.output],
                )
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

def execute_sequence(taskSequence: TaskSequence, memory: Memory, pckg: PackagesEnv, runnerContext: RunnerContext) -> None:
    taskId: TaskId|None = None
    try:
        executionContext = runnerContext.project(taskSequence)
        for taskId in taskSequence.tasks:
            pckg.extend(executionContext.tasks[taskId].definition.environment)
            run(taskId, executionContext, memory)
            callback(
                executionContext.callback,
                TaskSuccess(worker=taskSequence.worker, ts=taskId),
            )
        memory.flush()
    except Exception as e:
        logger.exception("runner failure, about to report")
        callback(
            runnerContext.callback,
            TaskFailure(worker=taskSequence.worker, task=taskId, detail=repr(e)),
        )

def entrypoint(runnerContext: RunnerContext): 
    logging.config.dictConfig(logging_config)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.bind(worker_address(runnerContext.workerId))
    callback(runnerContext.callback, WorkerReady(runnerContext.workerId))
    with Memory(runnerContext.callback, runnerContext.workerId) as memory, PackagesEnv() as pckg:
        label("worker", repr(runnerContext.workerId))
        gpu_id = str(runnerContext.workerId.worker_num())
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_id)
        # NOTE check any(task.definition.needs_gpu) anywhere?
        # TODO configure OMP_NUM_THREADS, blas, mkl, etc -- not clear how tho

        while True:
            mRaw = socket.recv()
            mDes = serde.des_message(mRaw)
            if isinstance(mDes, WorkerShutdown):
                logger.debug(f"worker {runnerContext.workerId} shutting down")
                break
            if not isinstance(mDes, TaskSequence):
                raise ValueError(f"unexpected message received: {type(mDes)}")
            execute_sequence(mDes, memory, pckg, runnerContext)
