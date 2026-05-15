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
import tempfile
import uuid
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable

import cascade.executor.platform as platform
import cascade.executor.runner.setup as runner_setup
import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from cascade.deployment.logging import LoggingConfig, as_dict_config, process_log_paths
from cascade.executor.comms import GraceWatcher, Listener, ReliableSender, callback
from cascade.executor.comms import default_message_resend_ms as resend_grace_ms
from cascade.executor.comms import default_timeout_ms as comms_default_timeout_ms
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.data_server import start_data_server
from cascade.executor.msg import (
    Ack,
    BackboneAddress,
    DatasetPersistFailure,
    DatasetPersistSuccess,
    DatasetPublished,
    DatasetPurge,
    DatasetRetrieveFailure,
    DatasetRetrieveSuccess,
    DatasetTransmitFailure,
    ExecutorExit,
    ExecutorFailure,
    ExecutorRegistration,
    ExecutorShutdown,
    Message,
    RunnerRestartRequest,
    TaskFailure,
    TaskSequence,
    Worker,
    WorkerReady,
    WorkerShutdown,
)
from cascade.executor.runner.entrypoint import worker_address
from cascade.executor.runner.setup import RunnerContext, WorkerProcessHandle
from cascade.low.core import DatasetId, HostId, JobInstance, TaskId, WorkerId
from cascade.low.exceptions import CascadeError, CascadeInfrastructureError, CascadeInternalError, CascadeUserError, ser
from cascade.low.func import md5hash24
from cascade.low.tracing import TaskLifecycle, mark
from cascade.low.views import param_source
from cascade.shm.server import entrypoint as shm_server
from cascade.ygg.transport import destroy_context, has_context

logger = logging.getLogger(__name__)
heartbeat_grace_ms = 2 * comms_default_timeout_ms

# messages from the data server which need to go to controller, but have no additional logic here
JustForwardToController = DatasetTransmitFailure | DatasetPersistSuccess | DatasetPersistFailure | DatasetRetrieveFailure


def address_of(port: int) -> BackboneAddress:
    return f"tcp://{platform.get_bindabble_self()}:{port}"


@dataclass
class WorkerHandle:
    process: WorkerProcessHandle
    attempt_cnt: int
    venv_dir: tempfile.TemporaryDirectory[str]


class Executor:
    def __init__(
        self,
        job_instance: JobInstance,
        controller_address: BackboneAddress,
        workers: int,
        host: HostId,
        portBase: int,
        shm_vol_gb: int | None,
        loggingConfig: LoggingConfig,
        url_base: str,
    ) -> None:
        logger.debug(f"init start: {has_context()=}")
        self.job_instance = job_instance
        self.schema_lookup = RunnerContext.build_schema_lookup(self.job_instance)
        self.param_source = param_source(job_instance.edges)
        self.controller_address = controller_address
        self.host = host
        self.workers: dict[WorkerId, WorkerHandle | None] = {WorkerId(host, f"w{i}"): None for i in range(workers)}
        self.worker_awaits: dict[WorkerId, None | TaskSequence] = {}
        self.loggingConfig = loggingConfig
        self.old_workers: list[tuple[WorkerProcessHandle, tempfile.TemporaryDirectory[str]]] = []

        self.datasets: set[DatasetId] = set()
        self.heartbeat_watcher = GraceWatcher(grace_ms=heartbeat_grace_ms)

        self.terminating = False
        logger.debug(f"register terminate function, {has_context()=}")
        atexit.register(self.terminate)
        # NOTE following inits are with potential side effects
        # NOTE shm server is the most dangerous because we fork, therefore imperative to not create zmq context before
        # TODO make shm startable with forkserver, there is some propagation issue in that case -- reproducible with tests
        # TODO make the shm server params configurable
        shm_port = f"/tmp/cascShmSock-{uuid.uuid4()}"  # portBase + 2
        shm_api.publish_socket_addr(shm_port)
        logger.debug(f"about to start an shm process, {has_context()=}")
        self.shm_process = platform.get_mp_ctx("executor-shm").Process(
            target=shm_server,
            kwargs={
                "capacity": shm_vol_gb * (1024**3) if shm_vol_gb else None,
                "logging_config": as_dict_config(loggingConfig, "shm"),
                "shm_pref": f"sCasc{host}",
            },
        )
        self.shm_process.start()
        self.mlistener = Listener(address_of(portBase))
        self.sender = ReliableSender(self.mlistener.address, resend_grace_ms)
        self.sender.add_host(HostId("controller"), controller_address)
        logger.debug(f"started shm process with pid {self.shm_process.pid}")
        self.daddress = address_of(portBase + 1)
        logger.debug(f"about to start a data server process, {has_context()=}")
        self.data_server = platform.get_mp_ctx("executor-dataserver").Process(
            target=start_data_server,
            args=(
                self.mlistener.address,
                self.daddress,
                self.host,
                as_dict_config(loggingConfig, "dsr"),
            ),
        )
        self.data_server.start()
        logger.debug(f"started data server process with pid {self.data_server.pid}")
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
            url_base=url_base,
        )
        # Build the shared RunnerContext and save it to POSIX shared memory.
        # All workers on this executor share this object; only per-worker identity (WorkerSetup)
        # is passed per-process via envvar.
        # The key is host-unique, but quick restarts are problematic on mac, and we cant have too long key for mac
        self.runner_ctx_shm_key = md5hash24(f"sCascRnrCtx{host}" + str(uuid.uuid4()))
        runner_ctx = RunnerContext(
            job=self.job_instance,
            callback=self.mlistener.address,
            param_source=self.param_source,
            loggingConfig=self.loggingConfig,
            schema_lookup=self.schema_lookup,
        )
        self.runner_ctx_shm: SharedMemory = runner_setup.save_runner_ctx_to_shm(runner_ctx, self.runner_ctx_shm_key)
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
                if (handle := self.workers[worker]) is not None:
                    callback(worker_address(worker, handle.attempt_cnt), WorkerShutdown())
                    handle.process.wait()
                    try:
                        handle.venv_dir.cleanup()
                    except Exception as e:
                        logger.warning(f"failed to cleanup venv for {worker}: {repr(e)}")
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down {worker}")
        for proc, venv in self.old_workers:
            logger.debug(f"cleanup old process {proc.pid}")
            try:
                proc.wait()
                venv.cleanup()
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down old worker {proc.pid}")
        if hasattr(self, "runner_ctx_shm") and self.runner_ctx_shm is not None:
            try:
                self.runner_ctx_shm.close()
                self.runner_ctx_shm.unlink()
            except Exception as e:
                logger.warning(f"failed to free runner ctx shm: {repr(e)}")
        if hasattr(self, "shm_process") and self.shm_process is not None and self.shm_process.is_alive():
            try:
                shm_client.shutdown()
                self.shm_process.join()
            except Exception as e:
                logger.warning(f"gotten {repr(e)} when shutting down shm server")
        if hasattr(self, "data_server") and self.data_server is not None and self.data_server.is_alive():
            self.data_server.kill()

    def to_controller(self, m: Message) -> None:
        self.heartbeat_watcher.step()
        self.sender.send(HostId("controller"), m)

    def _start_worker(self, worker: WorkerId, attempt_cnt: int, seq: None | TaskSequence) -> WorkerHandle:
        venv_td, initial_installed = runner_setup.create_venv()
        worker_setup = runner_setup.WorkerSetup(
            workerId=worker,
            workerAttemptCnt=attempt_cnt,
            shm_key=self.runner_ctx_shm_key,
            initial_installed=initial_installed,
        )
        worker_log_paths = process_log_paths(self.loggingConfig, f"worker_{worker.worker}")
        p = runner_setup.launch_in_venv(
            "cascade.executor.runner.entrypoint",
            venv_td.name,
            {runner_setup.WORKER_SETUP_ENVVAR: worker_setup.to_str()},
            stdout_path=worker_log_paths.stdout if worker_log_paths is not None else None,
            stderr_path=worker_log_paths.stderr if worker_log_paths is not None else None,
        )
        if worker in self.worker_awaits:
            raise CascadeInternalError(f"{worker=} was already awaiting")
        self.worker_awaits[worker] = seq
        return WorkerHandle(process=p, attempt_cnt=attempt_cnt, venv_dir=venv_td)

    def start_workers(self, workers: Iterable[WorkerId]) -> None:
        # NOTE fork would be better but causes issues on macos+torch with XPC_ERROR_CONNECTION_INVALID
        initialCnt = 0
        for worker in workers:
            handle = self._start_worker(worker=worker, attempt_cnt=initialCnt, seq=None)
            self.workers[worker] = handle
            logger.debug(f"started process {handle.process.pid} for worker {worker}")

        self.remaining = set(workers)

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
            try:
                self.to_controller(ExecutorFailure(self.host, ser(e)))
            except Exception:
                logger.exception("failed to send ExecutorFailure to controller after register failure")
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
                # this should not really be happening -> InternalError
                raise CascadeInternalError(f"process on {k} is not alive")
            elif procFail(e.process.poll()):
                # we assume low memory setting or callable issue -> UserError
                raise CascadeUserError(f"process on {k} failed to terminate correctly: {e.process.pid} -> {e.process.poll()}")
        if procFail(self.shm_process.exitcode):
            # possibly low memory setting but this is system config -> InfrastructureError
            raise CascadeInfrastructureError(f"shm server {self.shm_process.pid} failed with {self.shm_process.exitcode}")
        if procFail(self.data_server.exitcode):
            # unknown issue, it failed to report its own -> InfrastructureError
            raise CascadeInfrastructureError(f"data server {self.data_server.pid} failed with {self.data_server.exitcode}")
        if self.heartbeat_watcher.is_breach() > 0:
            logger.debug(
                f"grace elapsed without message by {self.heartbeat_watcher.elapsed_ms()} -> sending explicit heartbeat at {self.host}"
            )
            # NOTE we send registration in place of heartbeat -- it makes the startup more reliable,
            # and the registration's size overhead is negligible
            self.to_controller(self.registration)
        if self.old_workers and self.old_workers[0][0].poll() is not None:
            # we check just the first one for simplicity
            proc, venv = self.old_workers.pop(0)
            proc.wait()
            try:
                venv.cleanup()
            except Exception as e:
                logger.warning(f"failed to cleanup old worker venv: {repr(e)}")

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
                        handle = self.workers[m.worker]
                        if handle is None or handle.process.poll() is not None:
                            # unexpected exit -> InfrastructureError
                            raise CascadeInfrastructureError(f"worker process {m.worker} is not alive")
                        if m.worker in self.worker_awaits:
                            if self.worker_awaits[m.worker] is not None:
                                raise CascadeInternalError(f"double enqueue for {m.worker}")
                            else:
                                self.worker_awaits[m.worker] = m
                        else:
                            callback(worker_address(m.worker, handle.attempt_cnt), m)
                    elif isinstance(m, Ack):
                        self.sender.ack(m.idx)
                    elif isinstance(m, DatasetPurge):
                        if m.ds not in self.datasets:
                            logger.warning(f"unexpected purge of {m.ds}")
                        else:
                            for worker in self.workers:
                                handle = self.workers[worker]
                                if handle is not None:
                                    callback(worker_address(worker, handle.attempt_cnt), m)
                            self.datasets.remove(m.ds)
                            callback(self.daddress, m)
                    elif isinstance(m, ExecutorShutdown):
                        self.to_controller(ExecutorExit(self.host))
                        self.terminate()
                        break
                    # from entrypoint
                    elif isinstance(m, WorkerReady):
                        if not m.worker in self.worker_awaits:
                            logger.warning(f"unexpectedly gotten WorkerReady from {m.worker}, assuming double send")
                        else:
                            handle = self.workers[m.worker]
                            if handle is None:
                                raise CascadeInternalError(f"worker {m.worker} is alive but has no handle")
                            for ds in self.datasets:
                                # populate worker with all available datasets
                                dsm = DatasetPublished(ds=ds, origin=self.host, transmit_idx=None)
                                callback(worker_address(m.worker, handle.attempt_cnt), dsm)
                            maybe_seq = self.worker_awaits.pop(m.worker)
                            if maybe_seq is not None:
                                address = worker_address(m.worker, handle.attempt_cnt)
                                logger.debug(f"worker {m.worker} ready, sending task sequence {maybe_seq} to {address}")
                                callback(address, maybe_seq)
                            else:
                                logger.debug(f"worker {m.worker} ready, no work enqueued")
                    elif isinstance(m, RunnerRestartRequest):
                        handle = self.workers[m.worker]
                        if handle is None:
                            raise CascadeInternalError("unexpected restart from worker without handle")
                        callback(worker_address(m.worker, handle.attempt_cnt), WorkerShutdown())
                        self.old_workers.append((handle.process, handle.venv_dir))
                        logger.debug(f"will restart worker {m.worker} with attempt {handle.attempt_cnt + 1}")
                        self.workers[m.worker] = self._start_worker(m.worker, handle.attempt_cnt + 1, m.remainder)
                        self.to_controller(m)
                    elif isinstance(m, TaskFailure):
                        logger.debug(f"Forwarding task failure {m}")
                        self.to_controller(m)
                    elif isinstance(m, DatasetPublished):
                        for worker in self.workers:
                            # NOTE if we knew the origin worker, we would exclude it here... but doesn't really matter
                            handle = self.workers[worker]
                            if handle is not None:
                                callback(worker_address(worker, handle.attempt_cnt), m)
                        self.datasets.add(m.ds)
                        self.to_controller(m)
                    elif isinstance(m, DatasetRetrieveSuccess):
                        availability_notification = DatasetPublished(ds=m.ds, origin=self.host, transmit_idx=None)
                        for worker, handle in self.workers.items():
                            if handle is not None:
                                callback(worker_address(worker, handle.attempt_cnt), availability_notification)
                        self.datasets.add(m.ds)
                        self.to_controller(m)
                    elif isinstance(m, JustForwardToController):
                        self.to_controller(m)
                    else:
                        # NOTE transmit and store are handled in DataServer (which has its own socket)
                        raise CascadeInternalError(f"unexpected message type in executor recv_loop: {type(m)}")
                self.healthcheck()
            except Exception as e:
                logger.exception("executor exited, about to report to controller, propagating")
                self.to_controller(ExecutorFailure(self.host, ser(e)))
                self.terminate()
