"""
Represents the main on-host process, together with the SHM server. Launched at the
cluster startup, torn down when the controller reaches exit. Spawns `runner`s
for every task sequence it receives from controller -- those processes actually run
the tasks themselves.
"""

# NOTE this is an intermediate step toward long lived runners -- they would need to
# have their own zmq server as well as run the callables themselves

import atexit
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from typing import cast
import socket
import logging

from cascade.low.core import WorkerId, DatasetId, JobInstance, DatasetId, TaskId, HostId
from cascade.low.func import assert_never
from cascade.low.views import param_source
from cascade.executor.msg import BackboneAddress, ExecutionContext, TaskSequence, Message, ExecutorExit, ExecutorFailure, ExecutorRegistration, DatasetPurge, ExecutorShutdown, TaskFailure, TaskSuccess, DatasetPublished, DatasetTransmitFailure, ExecutorHeartbeat
from cascade.executor.runner import entrypoint, ds2shmid
from cascade.executor.comms import Listener, callback, default_timeout_sec as comms_default_timeout_sec, GraceWatcher
from cascade.executor.data_server import start_data_server
import cascade.shm.client as shm_client
import cascade.shm.api as shm_api
from cascade.shm.server import entrypoint as shm_server
from cascade.executor.config import logging_config
from cascade.low.tracing import mark, label, TaskLifecycle

logger = logging.getLogger(__name__)
heartbeat_grace_ms = 2*comms_default_timeout_sec*1_000

def spawn_task_sequence(taskSequence: TaskSequence, workerId: WorkerId, callback: BackboneAddress, job: JobInstance, param_source: dict[TaskId, dict[int | str, DatasetId]]) -> BaseProcess:
    """Spawns a process with callback and exits"""
    # TODO replace with a dedicated Factory process keeping a pool of zmq-awkable processes (thats a step towards persistent workers anyway)

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
    )

    # NOTE a bit risky, but forkserver is too slow. Has unhealthy side effects, like preserving imports caused
    # by JobInstance deser
    ctx = get_context("fork") 
    p = ctx.Process(
        target=entrypoint,
        kwargs={"taskSequence": taskSequence, "executionContext": executionContext},
    )
    p.start()
    return p

def address_of(port: int) -> BackboneAddress:
    return f"tcp://{socket.gethostname()}:{port}"

class Executor:
    def __init__(self, job_instance: JobInstance, controller_address: BackboneAddress, workers: int, host: HostId, portBase: int, shm_vol_gb: int|None = None) -> None:
        self.job_instance = job_instance
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers = [WorkerId(host, f"w{i}") for i in range(workers)]

        self.datasets: set[DatasetId] = set()
        self.task_queue: dict[WorkerId, None|TaskSequence|BaseProcess] = {e: None for e in self.workers}
        self.task_prereq: dict[WorkerId, set[DatasetId]] = {e: set() for e in self.workers}
        self.heartbeat_watcher = GraceWatcher(grace_ms = heartbeat_grace_ms)

        self.terminating = False
        atexit.register(self.terminate)
        # NOTE following inits are with potential side effects
        self.mlistener = Listener(address_of(portBase))
        # TODO make the shm server params configurable
        shm_port = portBase+2
        shm_api.publish_client_port(shm_port)
        ctx = get_context("fork")
        self.shm_process = ctx.Process(
            target=shm_server,
            args=(shm_port, shm_vol_gb * (1024**3) if shm_vol_gb else None, logging_config, f"sCasc{host}"),
        )
        self.shm_process.start()
        self.daddress = address_of(portBase+1)
        self.data_server = ctx.Process(
            target=start_data_server,
            args=(self.mlistener.address, self.daddress, self.host, shm_port, logging_config)
        )
        self.data_server.start()
        logger.debug("constructed executor")

    def terminate(self) -> None:
        # NOTE a bit care here:
        # 1/ the call itself can cause another terminate invocation, so we prevent that with a guard var
        # 2/ we can get here during the object construction (due to atexit), so we need to `hasattr`
        # 3/ we try catch everyhting since we dont want to leave any process dangling etc
        #    TODO it would be more reliable to use `prctl` + PR_SET_PDEATHSIG in shm, or check the ppid in there
        logger.debug("terminating")
        if self.terminating:
            return
        self.terminating = True
        for worker in self.workers:
            logger.debug(f"cleanup worker {worker}")
            try:
                self.maybe_cleanup(worker)
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down {worker}")
        if hasattr(self, 'shm_process') and self.shm_process is not None and self.shm_process.is_alive():
            try:
                shm_client.shutdown()
                self.shm_process.join()
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down shm server")
        if hasattr(self, 'data_server') and self.data_server is not None and self.data_server.is_alive():
            self.data_server.kill()

    def to_controller(self, m: Message) -> None:
        self.heartbeat_watcher.step()
        callback(self.controller_address, m)

    def register(self) -> None:
        try:
            shm_client.ensure()
            # TODO some ensure on the data server?
            import time
            time.sleep(1) # TODO delete this after ClusterStarted & retries in place
            logger.debug(f"about to send register message from {self.host}")
            registration = ExecutorRegistration(
                host=self.host,
                maddress=self.mlistener.address,
                daddress=self.daddress,
                workers=self.workers,
            )
            self.to_controller(registration)
        except Exception as e:
            logger.exception("failed during register")
            self.terminate()
        # TODO await ClusterStarted message here, otherwise retry -- cf the sleep above

    def maybe_spawn(self, worker: WorkerId) -> None:
        ts = self.task_queue[worker]
        if not isinstance(ts, TaskSequence):
            return
        if not self.task_prereq[worker] <= self.datasets:
            # TODO this is a possible slowdown, we don't wait until everything is available -- but maybe
            # the first task in the queue could already start
            return

        self.task_queue[worker] = spawn_task_sequence(ts, WorkerId(self.host, f"w{worker}"), self.mlistener.address, self.job_instance, self.param_source)

    def maybe_cleanup(self, worker: WorkerId) -> None:
        """Housekeeping task that checks that previously spawned task has succesfully finished.
        If `self.terminating` is true, awaits both future spawn and process join. Otherwise, expects that both have
        already happened."""
        ts = self.task_queue[worker]
        if isinstance(ts, TaskSequence):
            if not self.terminating:
                raise ValueError(f"premature cleanup on {worker}: {ts} not spawned yet")
        elif isinstance(ts, BaseProcess):
            proc_handle = ts
            if not proc_handle.pid:
                mes = f"process on {worker} failed to start"
                if self.terminating:
                    logger.warning(mes)
                else:
                    raise ValueError(mes)
            # NOTE this timeout should never be breached! If it happens, increase it?
            proc_handle.join(timeout=2 if not self.terminating else None)
            if proc_handle.exitcode != 0:
                mes = f"process on {worker} failed to terminate correctly: {proc_handle.pid} -> {proc_handle.exitcode}"
                if self.terminating:
                    logger.warning(mes)
                else:
                    raise ValueError(mes)
        elif ts is None:
            pass
        else:
            assert_never(ts)

    def healthcheck(self) -> None:
        """Checks that no process died, and sends a heartbeat message in case the last message to controller
        was too long ago"""
        procFail = lambda ex: ex is not None and ex != 0
        for k, e in self.task_queue.items():
            if isinstance(e, BaseProcess) and procFail(e.exitcode):
                ValueError(f"process on {k} failed to terminate correctly: {e.pid} -> {e.exitcode}")
        if procFail(self.shm_process.exitcode):
            ValueError(f"shm server {self.shm_process.pid} failed with {self.shm_process.exitcode}")
        if procFail(self.data_server.exitcode):
            ValueError(f"data server {self.data_server.pid} failed with {self.data_server.exitcode}")
        if self.heartbeat_watcher.is_breach():
            logger.debug(f"grace elapsed without message by {self.heartbeat_watcher.elapsed_ms()} -> sending explicit heartbeat at {self.host}")
            self.to_controller(ExecutorHeartbeat(host=self.host))

    def enqueue_task(self, ts: TaskSequence) -> None:
        for task in ts.tasks:
            mark({"task": task, "worker": repr(ts.worker), "action": TaskLifecycle.enqueued})
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
        while not self.terminating:
            try:
                for m in self.mlistener.recv_messages():
                    logger.debug(f"received {type(m)}")
                    # from controller
                    if isinstance(m, TaskSequence):
                        self.enqueue_task(m)
                    elif isinstance(m, DatasetPurge):
                        if not m.ds in self.datasets:
                            logger.warning(f"unexpected purge of {m.ds}")
                        else:
                            self.datasets.remove(m.ds)
                        shm_client.purge(ds2shmid(m.ds))
                    elif isinstance(m, ExecutorShutdown):
                        self.to_controller(ExecutorExit(self.host))
                        self.terminate()
                        break
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
                    elif isinstance(m, DatasetTransmitFailure):
                        self.to_controller(m)
                    else:
                        # NOTE transmit and store are handled in DataServer (which has its own socket)
                        raise TypeError(m)
                self.healthcheck()
            except Exception as e:
                logger.exception("executor exited, about to report to controller")
                self.to_controller(ExecutorFailure(self.host, repr(e)))
                self.terminate()
