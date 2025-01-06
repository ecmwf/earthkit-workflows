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

from cascade.low.core import WorkerId, DatasetId, JobInstance, DatasetId, TaskId
from cascade.low.func import assert_never
from cascade.low.views import param_source
from cascade.executor.msg import BackboneAddress, ExecutionContext, TaskSequence
from cascade.executor.runner import entrypoint

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
    def __init__(self, job_instance: JobInstance, controller_address: BackboneAddress, workers: int, host: str) -> None:
        self.job_instance = job_instance
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers = {f"{host}:w{i}" for i in range(workers)}
        self.datasets: set[DatasetId] = set()
        raise NotImplementedError("fix callback below")
        self.callback: BackboneAddress = "" # TODO

        self.task_queue: dict[WorkerId, None|TaskSequence|Future] = {e: None for e in self.workers}
        self.task_prereq: dict[WorkerId, set[DatasetId]] = {e: set() for e in self.workers}
        self.proc_spawn_tp: PythonExecutor = ThreadPoolExecutor(max_workers=1)

    def maybe_spawn(self, worker: WorkerId) -> None:
        ts = self.task_queue[worker]
        if not isinstance(ts, TaskSequence):
            return
        if not self.task_prereq[worker] <= self.datasets:
            # TODO this is a possible slowdown, we don't wait until everything is available -- but maybe
            # the first task in the queue could already start
            return

        func = spawn_task_sequence
        args = (ts, WorkerId(self.host, f"w{worker}"), self.callback, self.job_instance, self.param_source)
        self.task_queue[worker] = self.proc_spawn_tp.submit(func, *args)

    def maybe_cleanup(self, worker: WorkerId) -> None:
        """Housekeeping task that checks that previously spawned task has succesfully finished. Called
        prior to enqueuing a new one"""
        # TODO call this regularly to report failures earlier -- should not crash upon unfinished spawn/run tho
        ts = self.task_queue[worker]
        if isinstance(ts, TaskSequence):
            raise ValueError(f"premature cleanup on {worker}: {ts} not spawned yet")
        elif isinstance(ts, Future):
            # NOTE this timeout should never be breached! If it happens, increase proc_spawn_tp.max_workers
            proc_handle = ts.result(timeout=2)
            if not proc_handle.pid:
                raise ValueError(f"process on {worker} failed to start")
            # NOTE this timeout should never be breached! If it happens, increase it?
            proc_handle.join(timeout=2)
            if proc_handle.pid != 0:
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
        raise NotImplementedError
        # what needs to be handled:
        # submit from controller -> call enqueue
        # purge from controller -> call shm directly?
        # publish from local -> fwd to controller, call maybe spawn for all workers?
        # fail from local -> raise
        # succ from local -> fwd to controller (no need to clean locally I guess)
        # transmit from controller -> transmit tp, submit
        # fetch from controller -> transmit tp, submit

        # receive from exec -> put to shm in blocking? We may want to have a different address, monitored in a separate process? That one would just publish is_done for the main to check enqueue

        # consider running regular cleanups?
        
