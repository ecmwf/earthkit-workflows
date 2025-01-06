"""
Represents the main on-host process, together with the SHM server. Launched at the
cluster startup, torn down when the controller reaches exit. Spawns `runner`s
for every task sequence it receives from controller -- those processes actually run
the tasks themselves.
"""

# NOTE this is an intermediate step toward long lived runners -- they would need to
# have their own zmq server as well as run the callables themselves

from concurrent.futures import Future, ThreadPoolExecutor, Executor as PythonExecutor
from multiprocessing import get_context, Process
from typing import cast
import socket
import logging

from cascade.low.core import WorkerId, DatasetId, JobInstance, DatasetId, TaskId, HostId
from cascade.low.func import assert_never
from cascade.low.views import param_source
from cascade.executor.msg import BackboneAddress, ExecutionContext, TaskSequence, Message, ExecutorExit, ExecutorFailure, ExecutorRegistration, DatasetPurge, ExecutorShutdown, TaskFailure, TaskSuccess, DatasetPublished
from cascade.executor.runner import entrypoint, ds2shmid
from cascade.executor.comms import Listener, callback
import cascade.shm.client as shm_client

logger = logging.getLogger(__name__)

def spawn_task_sequence(taskSequence: TaskSequence, workerId: WorkerId, callback: BackboneAddress, job: JobInstance, param_source: dict[TaskId, dict[int | str, DatasetId]]) -> Process:
    """Assumed to run in a thread pool to not block the main recv loop. Spawns a process with callback
    and exits"""

    param_source_ext: dict[TaskId, dict[int|str, tuple[DatasetId, str]]] = {
        task: {
            k: (
                dataset_id, 
                job.tasks[dataset_id.task].definition.output_schema[dataset_id.output],
            )
            for k, dataset_id in param_source[task].items()
        }
        for task in taskSequence.tasks
    }
    executionContext = ExecutionContext(
        tasks={task: job.tasks[task] for task in taskSequence.tasks},
        param_source=param_source_ext,
        callback=callback,
    ) # type: ignore

    ctx = get_context("fork")  # so far works, but switch to forkspawn if not
    p = ctx.Process(
        target=entrypoint,
        kwargs={"taskSequence": taskSequence, "executionContext": executionContext},
    )
    p.start()
    return cast(Process, p)

class Executor:
    def __init__(self, job_instance: JobInstance, controller_address: BackboneAddress, workers: int, host: HostId, portBase: int) -> None:
        self.job_instance = job_instance
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers = [WorkerId(host, f"w{i}") for i in range(workers)]
        self.mlistener = Listener(f"tcp://{socket.gethostname()}:{portBase}")
        self.dlistener = Listener(f"tcp://{socket.gethostname()}:{portBase+1}") # TODO move this to another process, have it listen receive etc
        # TODO launch shm here?

        self.terminating = False
        self.datasets: set[DatasetId] = set()
        self.task_queue: dict[WorkerId, None|TaskSequence|Future] = {e: None for e in self.workers}
        self.task_prereq: dict[WorkerId, set[DatasetId]] = {e: set() for e in self.workers}
        self.proc_spawn_tp: PythonExecutor = ThreadPoolExecutor(max_workers=1)
        logger.debug("constructed executor")

    def terminate(self) -> None:
        if self.terminating:
            return
        self.terminating = True
        for worker in self.workers:
            logger.debug(f"cleanup worker {worker}")
            try:
                self.maybe_cleanup(worker)
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down {worker}")
        self.proc_spawn_tp.shutdown() # should be empty already due to maybe_cleanup
        # TODO shutdown dprocess, shm

    def to_controller(self, m: Message) -> None:
        callback(self.controller_address, m)

    def register(self) -> None:
        registration = ExecutorRegistration(
            host=self.host,
            address=self.mlistener.address,
            workers=self.workers,
        )
        self.to_controller(registration)

    def maybe_spawn(self, worker: WorkerId) -> None:
        ts = self.task_queue[worker]
        if not isinstance(ts, TaskSequence):
            return
        if not self.task_prereq[worker] <= self.datasets:
            # TODO this is a possible slowdown, we don't wait until everything is available -- but maybe
            # the first task in the queue could already start
            return

        func = spawn_task_sequence
        args = (ts, WorkerId(self.host, f"w{worker}"), self.mlistener.address, self.job_instance, self.param_source)
        self.task_queue[worker] = self.proc_spawn_tp.submit(func, *args)

    def maybe_cleanup(self, worker: WorkerId) -> None:
        """Housekeeping task that checks that previously spawned task has succesfully finished.
        If `self.terminating` is true, awaits both future spawn and process join. Otherwise, expects that both have
        already happened."""
        # TODO call this regularly to report failures earlier -- should neither block nor crash tho
        ts = self.task_queue[worker]
        if isinstance(ts, TaskSequence):
            if not self.terminating:
                raise ValueError(f"premature cleanup on {worker}: {ts} not spawned yet")
        elif isinstance(ts, Future):
            # NOTE this timeout should never be breached! If it happens, increase proc_spawn_tp.max_workers
            proc_handle = ts.result(timeout=2 if not self.terminating else None)
            if not proc_handle.pid:
                if self.terminating:
                    logger.warning(f"process on {worker} failed to start")
                else:
                    raise ValueError(f"process on {worker} failed to start")
            # NOTE this timeout should never be breached! If it happens, increase it?
            proc_handle.join(timeout=2 if not self.terminating else None)
            if proc_handle.pid != 0:
                if self.terminating:
                    logger.warning(f"process on {worker} failed to terminate correctly: {proc_handle.pid}")
                else:
                    raise ValueError(f"process on {worker} failed to terminate correctly: {proc_handle.pid}")
        elif ts is None:
            pass
        else:
            assert_never(ts)

    def enqueue_task(self, ts: TaskSequence) -> None:
        self.maybe_cleanup(ts.worker)

        self.task_queue[ts.worker] = ts
        prereqs = {
            dataset_id
            for task in ts.tasks
            for dataset_id in self.param_source[task].values()
        }
        for task in ts.tasks:
            prereqs -= {DatasetId(task, key) for key in self.job_instance.tasks[task].definition.output_schema.keys()}
        self.task_prereq[ts.worker] = prereqs

        self.maybe_spawn(ts.worker)

    def recv_loop(self) -> None:
        logger.debug("entering recv loop")
        # TODO regular cleanup?
        # TODO monitor dlistener?
        while not self.terminating:
            try:
                for m in self.mlistener.recv_messages():
                    # from controller
                    if isinstance(m, TaskSequence):
                        self.enqueue_task(m)
                    elif isinstance(m, DatasetPurge):
                        if not m.ds in self.datasets:
                            logger.warning(f"unexpected purge of {m.ds}")
                        else:
                            self.datasets.remove(m.ds)
                        shm_client.purge(ds2shmid(m.ds))
                    # -> listener:
                    # TODO fetch
                    # TODO transmit
                    # TODO store
                    elif isinstance(m, ExecutorShutdown):
                        self.terminate()
                    # from entrypoint
                    elif isinstance(m, TaskFailure):
                        self.to_controller(m)
                    elif isinstance(m, TaskSuccess):
                        # NOTE we don't call maybe_cleanup here -- the process may not have joined yet so we'd just block,
                        # and we don't need to free the slot yet anyway
                        self.to_controller(m)
                    elif isinstance(m, DatasetPublished):
                        self.datasets.add(m.ds)
                        for worker in self.workers:
                            self.maybe_spawn(worker)
                        self.to_controller(m)
                    else:
                        raise TypeError(m)
            except Exception as e:
                logger.exception("executor exited, about to report to controller")
                self.to_controller(ExecutorFailure(self.host, repr(e)))
                self.terminate()
                raise
        self.to_controller(ExecutorExit(self.host))
