# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Represents the main on-host process, together with the SHM server. Launched at the
cluster startup, torn down when the controller reaches exit. Spawns `runner`s
for every task sequence it receives from controller -- those processes actually run
the tasks themselves.
"""

# NOTE this is an intermediate step toward long lived runners -- they would need to
# have their own zmq server as well as run the callables themselves

import atexit
import logging
import os
import socket
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from typing import Iterable

import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from cascade.executor.comms import GraceWatcher, Listener, ReliableSender, callback
from cascade.executor.comms import default_message_resend_ms as resend_grace_ms
from cascade.executor.comms import default_timeout_ms as comms_default_timeout_ms
from cascade.executor.config import logging_config
from cascade.executor.data_server import start_data_server
from cascade.executor.msg import (
    Ack,
    BackboneAddress,
    DatasetPublished,
    DatasetPurge,
    DatasetTransmitFailure,
    ExecutorExit,
    ExecutorFailure,
    ExecutorRegistration,
    ExecutorShutdown,
    Message,
    TaskFailure,
    TaskSequence,
    Worker,
    WorkerReady,
    WorkerShutdown,
)
from cascade.executor.runner.entrypoint import RunnerContext, entrypoint, worker_address
from cascade.low.core import DatasetId, HostId, JobInstance, WorkerId
from cascade.low.tracing import TaskLifecycle, mark
from cascade.low.views import param_source
from cascade.shm.server import entrypoint as shm_server

logger = logging.getLogger(__name__)
heartbeat_grace_ms = 2 * comms_default_timeout_ms


def address_of(port: int) -> BackboneAddress:
    return f"tcp://{socket.gethostname()}:{port}"


class Executor:
    def __init__(
        self,
        job_instance: JobInstance,
        controller_address: BackboneAddress,
        workers: int,
        host: HostId,
        portBase: int,
        shm_vol_gb: int | None = None,
    ) -> None:
        self.job_instance = job_instance
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers: dict[WorkerId, BaseProcess | None] = {
            WorkerId(host, f"w{i}"): None for i in range(workers)
        }

        self.datasets: set[DatasetId] = set()
        self.heartbeat_watcher = GraceWatcher(grace_ms=heartbeat_grace_ms)

        self.terminating = False
        atexit.register(self.terminate)
        # NOTE following inits are with potential side effects
        self.mlistener = Listener(address_of(portBase))
        self.sender = ReliableSender(self.mlistener.address, resend_grace_ms)
        self.sender.add_host("controller", controller_address)
        # TODO make the shm server params configurable
        shm_port = portBase + 2
        shm_api.publish_client_port(shm_port)
        ctx = get_context("fork")
        self.shm_process = ctx.Process(
            target=shm_server,
            args=(
                shm_port,
                shm_vol_gb * (1024**3) if shm_vol_gb else None,
                logging_config,
                f"sCasc{host}",
            ),
        )
        self.shm_process.start()
        self.daddress = address_of(portBase + 1)
        self.data_server = ctx.Process(
            target=start_data_server,
            args=(
                self.mlistener.address,
                self.daddress,
                self.host,
                shm_port,
                logging_config,
            ),
        )
        self.data_server.start()
        gpus = int(os.environ.get("CASCADE_GPU_COUNT", "0"))
        self.registration = ExecutorRegistration(
            host=self.host,
            maddress=self.mlistener.address,
            daddress=self.daddress,
            workers=[
                Worker(
                    worker_id=worker_id,
                    cpu=1,
                    gpu=1 if idx < gpus else 0,
                    memory_mb=1024,  # TODO better
                )
                for idx, worker_id in enumerate(self.workers.keys())
            ],
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
        if (
            hasattr(self, "shm_process")
            and self.shm_process is not None
            and self.shm_process.is_alive()
        ):
            try:
                shm_client.shutdown()
                self.shm_process.join()
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down shm server")
        if (
            hasattr(self, "data_server")
            and self.data_server is not None
            and self.data_server.is_alive()
        ):
            self.data_server.kill()

    def to_controller(self, m: Message) -> None:
        self.heartbeat_watcher.step()
        self.sender.send("controller", m)

    def start_workers(self, workers: Iterable[WorkerId]) -> None:
        # TODO this method assumes no other message will arrive to mlistener! Thus cannot be used for workers now
        # NOTE a bit risky, but forkserver is too slow. Has unhealthy side effects, like preserving imports caused
        # by JobInstance deser
        ctx = get_context("fork")
        for worker in workers:
            runnerContext = RunnerContext(
                workerId=worker,
                job=self.job_instance,
                param_source=self.param_source,
                callback=self.mlistener.address,
            )
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
        except:
            logger.exception("failed during register")
            self.terminate()
        # NOTE we don't mind this registration message being lost -- if that happens, we send it
        # during next heartbeat. But we may want to introduce a check that if no message,
        # including for-this-purpose introduced & guaranteed controller2worker heartbeat, arrived
        # for a long time, we shut down

    def healthcheck(self) -> None:
        """Checks that no process died, and sends a heartbeat message in case the last message to controller
        was too long ago
        """
        procFail = lambda ex: ex is not None and ex != 0
        for k, e in self.workers.items():
            if e is None:
                ValueError(f"process on {k} is not alive")
            elif procFail(e.exitcode):
                ValueError(
                    f"process on {k} failed to terminate correctly: {e.pid} -> {e.exitcode}"
                )
        if procFail(self.shm_process.exitcode):
            ValueError(
                f"shm server {self.shm_process.pid} failed with {self.shm_process.exitcode}"
            )
        if procFail(self.data_server.exitcode):
            ValueError(
                f"data server {self.data_server.pid} failed with {self.data_server.exitcode}"
            )
        if self.heartbeat_watcher.is_breach() > 0:
            logger.debug(
                f"grace elapsed without message by {self.heartbeat_watcher.elapsed_ms()} -> sending explicit heartbeat at {self.host}"
            )
            # NOTE we send registration in place of heartbeat -- it makes the startup more reliable,
            # and the registration's size overhead is negligible
            self.to_controller(self.registration)

    def recv_loop(self) -> None:
        logger.debug("entering recv loop")
        while not self.terminating:
            try:
                for m in self.mlistener.recv_messages(resend_grace_ms):
                    logger.debug(f"received {type(m)}")
                    # from controller
                    if isinstance(m, TaskSequence):
                        for task in m.tasks:
                            mark(
                                {
                                    "task": task,
                                    "worker": repr(m.worker),
                                    "action": TaskLifecycle.enqueued,
                                }
                            )
                        if (
                            proc := self.workers[m.worker]
                        ) is None or proc.exitcode is not None:
                            raise ValueError(f"worker process {m.worker} is not alive")
                        callback(worker_address(m.worker), m)
                    elif isinstance(m, Ack):
                        self.sender.ack(m.idx)
                    elif isinstance(m, DatasetPurge):
                        if m.ds not in self.datasets:
                            logger.warning(f"unexpected purge of {m.ds}")
                        else:
                            for worker in self.workers:
                                callback(worker_address(worker), m)
                            self.datasets.remove(m.ds)
                            callback(self.daddress, m)
                    elif isinstance(m, ExecutorShutdown):
                        self.to_controller(ExecutorExit(self.host))
                        self.terminate()
                        break
                    # from entrypoint
                    elif isinstance(m, TaskFailure):
                        self.to_controller(m)
                    elif isinstance(m, DatasetPublished):
                        for worker in self.workers:
                            # NOTE if we knew the origin worker, we would exclude it here... but doesn't really matter
                            callback(worker_address(worker), m)
                        self.datasets.add(m.ds)
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
