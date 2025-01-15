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
from typing import cast, Iterable
import socket
import logging

from cascade.low.core import WorkerId, DatasetId, JobInstance, DatasetId, TaskId, HostId
from cascade.low.func import assert_never
from cascade.executor.msg import BackboneAddress, TaskSequence, Message, ExecutorExit, ExecutorFailure, ExecutorRegistration, DatasetPurge, ExecutorShutdown, TaskFailure, TaskSuccess, DatasetPublished, DatasetTransmitFailure, WorkerReady, WorkerShutdown
from cascade.executor.runner.entrypoint import entrypoint, RunnerContext, worker_address
from cascade.executor.runner.memory import ds2shmid
from cascade.executor.comms import Listener, callback, default_timeout_sec as comms_default_timeout_sec, GraceWatcher
from cascade.executor.data_server import start_data_server
import cascade.shm.client as shm_client
import cascade.shm.api as shm_api
from cascade.shm.server import entrypoint as shm_server
from cascade.executor.config import logging_config
from cascade.low.tracing import mark, label, TaskLifecycle, Microtrace, timer
from cascade.low.views import param_source

logger = logging.getLogger(__name__)
heartbeat_grace_ms = 2*comms_default_timeout_sec*1_000

def address_of(port: int) -> BackboneAddress:
    return f"tcp://{socket.gethostname()}:{port}"

class Executor:
    def __init__(self, job_instance: JobInstance, controller_address: BackboneAddress, workers: int, host: HostId, portBase: int, shm_vol_gb: int|None = None) -> None:
        self.job_instance = job_instance
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers: dict[WorkerId, BaseProcess|None] = {WorkerId(host, f"w{i}"): None for i in range(workers)}

        self.datasets: set[DatasetId] = set()
        self.task_queue: dict[WorkerId, None|TaskSequence] = {e: None for e in self.workers.keys()}
        self.task_prereq: dict[WorkerId, set[DatasetId]] = {e: set() for e in self.workers.keys()}
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
        self.registration = ExecutorRegistration(
            host=self.host,
            maddress=self.mlistener.address,
            daddress=self.daddress,
            workers=list(self.workers.keys()),
        )
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
        for worker in self.workers.keys():
            logger.debug(f"cleanup worker {worker}")
            try:
                if (proc := self.workers[worker]) is not None:
                    callback(worker_address(worker), WorkerShutdown())
                    proc.join()
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

    def start_workers(self, workers: Iterable[WorkerId]) -> None:
        # TODO this method assumes no other message will arrive to mlistener! Thus cannot be used for workers now
        # NOTE a bit risky, but forkserver is too slow. Has unhealthy side effects, like preserving imports caused
        # by JobInstance deser
        ctx = get_context("fork") 
        for worker in workers:
            runnerContext = RunnerContext(workerId=worker, job=self.job_instance, param_source=self.param_source, callback=self.mlistener.address)
            p = ctx.Process(target=entrypoint, kwargs={"runnerContext": runnerContext})
            p.start()
            self.workers[worker] = p
            logger.debug(f"started process {p.pid} for worker {worker}")

        remaining = set(workers)
        while remaining:
            for m in self.mlistener.recv_messages():
                if not isinstance(m, WorkerReady):
                    raise ValueError(f"expected WorkerReady, gotten {type(m)}")
                logger.debug(f"worker {m.worker} ready")
                remaining.remove(m.worker)

    def register(self) -> None:
        # NOTE we do register explicitly post-construction so that the former one is network-latency-free.
        # However, it is especially important that `bind` (via Listener) happens *before* `register`, as
        # otherwise we may lose messages from the Controller
        try:
            # TODO actually send register first, but then need to handle `start_workers` not interfering with
            # arriving TaskSequence
            shm_client.ensure()
            # TODO some ensure on the data server?
            self.start_workers(self.workers.keys())
            logger.debug(f"about to send register message from {self.host}")
            self.to_controller(self.registration)
        except Exception as e:
            logger.exception("failed during register")
            self.terminate()
        # NOTE we don't mind this registration message being lost -- if that happens, we send it
        # during next heartbeat. But we may want to introduce a check that if no message,
        # including for-this-purpose introduced & guaranteed controller2worker heartbeat, arrived
        # for a long time, we shut down

    def maybe_spawn(self, worker: WorkerId) -> None:
        ts = self.task_queue[worker]
        if not isinstance(ts, TaskSequence):
            return
        if not self.task_prereq[worker] <= self.datasets:
            # TODO this is a possible slowdown, we don't wait until everything is available -- but maybe
            # the first task in the queue could already start... better fit for persistent workers
            return

        if (proc := self.workers[worker]) is None or proc.exitcode is not None:
            raise ValueError(f"worker process {worker} is not alive")
        self.task_queue[worker] = None
        callback(worker_address(worker), ts)

    def healthcheck(self) -> None:
        """Checks that no process died, and sends a heartbeat message in case the last message to controller
        was too long ago"""
        procFail = lambda ex: ex is not None and ex != 0
        for k, e in self.workers.items():
            if e is None:
                ValueError(f"process on {k} is not alive")
            elif procFail(e.exitcode):
                ValueError(f"process on {k} failed to terminate correctly: {e.pid} -> {e.exitcode}")
        if procFail(self.shm_process.exitcode):
            ValueError(f"shm server {self.shm_process.pid} failed with {self.shm_process.exitcode}")
        if procFail(self.data_server.exitcode):
            ValueError(f"data server {self.data_server.pid} failed with {self.data_server.exitcode}")
        if self.heartbeat_watcher.is_breach():
            logger.debug(f"grace elapsed without message by {self.heartbeat_watcher.elapsed_ms()} -> sending explicit heartbeat at {self.host}")
            # NOTE we send registration in place of heartbeat -- it makes the startup more reliable,
            # and the registration's size overhead is negligible
            self.to_controller(self.registration)

    def enqueue_task(self, ts: TaskSequence) -> None:
        for task in ts.tasks:
            mark({"task": task, "worker": repr(ts.worker), "action": TaskLifecycle.enqueued})
        if self.task_queue[ts.worker] is not None:
            raise ValueError(f"attempting to enqueue two tasks!")
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
