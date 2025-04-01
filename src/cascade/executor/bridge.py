# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles communication between controller and remote executors"""

import logging
import time

from cascade.executor.comms import GraceWatcher, Listener, ReliableSender
from cascade.executor.comms import default_message_resend_ms as resend_grace_ms
from cascade.executor.executor import heartbeat_grace_ms as executor_heartbeat_grace_ms
from cascade.executor.msg import (
    Ack,
    DatasetPublished,
    DatasetPurge,
    DatasetTransmitCommand,
    DatasetTransmitFailure,
    DatasetTransmitPayload,
    ExecutorExit,
    ExecutorFailure,
    ExecutorRegistration,
    ExecutorShutdown,
    Message,
    TaskFailure,
    TaskSequence,
)
from cascade.low.core import DatasetId, Environment, HostId, Worker, WorkerId
from cascade.low.func import assert_never

logger = logging.getLogger(__name__)

Event = DatasetPublished | DatasetTransmitPayload
ToShutdown = TaskFailure | ExecutorFailure | DatasetTransmitFailure | ExecutorExit
Unsupported = TaskSequence | DatasetPurge | DatasetTransmitCommand | ExecutorShutdown


class Bridge:
    def __init__(self, controller_url: str, expected_executors: int) -> None:
        self.mlistener = Listener(controller_url)
        self.heartbeat_checker: dict[HostId, GraceWatcher] = {}
        self.transmit_idx_counter = 0
        self.sender = ReliableSender(self.mlistener.address, resend_grace_ms)
        registered = 0
        self.environment = Environment(workers={})
        logger.debug("about to start receiving registrations")
        registration_grace = time.time_ns() + 3 * 60 * 1_000_000_000
        while registered < expected_executors:
            messages = self.mlistener.recv_messages(timeout_ms=10_000)
            logger.debug(f"received {messages=}")
            for message in messages:
                if not isinstance(message, ExecutorRegistration):
                    raise TypeError(type(message))
                if (
                    message.host in self.sender.hosts
                    or "data." + message.host in self.sender.hosts
                ):
                    logger.warning(
                        f"double registration of {message.host}, suggesting network congestion"
                    )
                    continue
                self.sender.add_host(message.host, message.maddress)
                self.sender.add_host("data." + message.host, message.daddress)
                for worker in message.workers:
                    self.environment.workers[worker.worker_id] = Worker(
                        cpu=worker.cpu, gpu=worker.gpu, memory_mb=worker.memory_mb
                    )
                registered += 1
                self.heartbeat_checker[message.host] = GraceWatcher(
                    2 * executor_heartbeat_grace_ms
                )
                self.heartbeat_checker[message.host].step()
            if time.time_ns() > registration_grace:
                self.shutdown()
                raise ValueError("failed to recevied registration in due time")

    def _send(self, hostId: HostId, message: Message) -> None:
        self.sender.send(hostId, message)

    def get_environment(self) -> Environment:
        return self.environment

    def recv_events(self) -> list[Event]:
        try:
            events: list[Event] = []
            shutdown_reason: None | Exception | Message = None
            while (not events) and (not shutdown_reason):
                # timeout ms matches
                for message in self.mlistener.recv_messages(timeout_ms=resend_grace_ms):
                    if hasattr(message, "host") and isinstance(
                        (host := message.host), HostId
                    ):
                        self.heartbeat_checker[host].step()
                    if hasattr(message, "worker") and isinstance(
                        (worker := message.worker), WorkerId
                    ):
                        self.heartbeat_checker[worker.host].step()
                    if isinstance(message, Event):
                        events.append(message)
                    elif isinstance(message, Ack):
                        self.sender.ack(message.idx)
                    elif isinstance(message, ExecutorRegistration):
                        pass
                    elif isinstance(message, ToShutdown):
                        logger.critical(
                            f"received failure {message=}, proceeding with a shutdown"
                        )
                        if (
                            isinstance(message, ExecutorExit | ExecutorFailure)
                            and message.host in self.sender.hosts
                        ):
                            self.sender.hosts.pop(message.host)
                            self.sender.hosts.pop("data." + message.host)
                        shutdown_reason = message
                    elif isinstance(message, Unsupported):
                        logger.critical(
                            f"received unexpected {message=}, proceeding with a shutdown"
                        )
                        shutdown_reason = message
                    else:
                        assert_never(message)
                failed_heartbeats = (
                    e for e in self.heartbeat_checker.items() if e[1].is_breach() == 2
                )
                for host, checker in failed_heartbeats:
                    logger.warning(
                        f"{host=} failed to heartbeat for {checker.elapsed_ms()/1e3:.3f}s"
                    )
                self.sender.maybe_retry()
            if shutdown_reason is None:
                return events
        except Exception as e:
            logger.exception("gotten exception, proceeding with a shutdown")
            shutdown_reason = e
        self.shutdown()
        raise ValueError(shutdown_reason)

    def task_sequence(self, taskSequence: TaskSequence) -> None:
        self._send(taskSequence.worker.host, taskSequence)

    def purge(self, host: HostId, ds: DatasetId) -> None:
        m = DatasetPurge(ds=ds)
        self._send(host, m)

    def transmit(self, ds: DatasetId, source: HostId, target: HostId) -> None:
        m = DatasetTransmitCommand(
            source=source,
            target=target,
            daddress=self.sender.hosts["data." + target][1],
            ds=ds,
            idx=self.transmit_idx_counter,
        )
        self.transmit_idx_counter += 1
        self.sender.send("data." + source, m)

    def fetch(self, ds: DatasetId, source: HostId) -> None:
        m = DatasetTransmitCommand(
            source=source,
            target="controller",
            daddress=self.mlistener.address,
            ds=ds,
            idx=self.transmit_idx_counter,
        )
        self.transmit_idx_counter += 1
        self.sender.send("data." + source, m)

    def shutdown(self) -> None:
        m = ExecutorShutdown()
        for host in self.sender.hosts.keys():
            if not host.startswith("data."):
                self._send(host, m)
        shutdown_grace = time.time_ns() + 3 * 60 * 1_000_000_000
        while self.sender.hosts and time.time_ns() < shutdown_grace:
            # we want to consume all those exit messages
            for message in self.mlistener.recv_messages():
                if isinstance(message, ExecutorExit | ExecutorFailure):
                    if message.host in self.sender.hosts:
                        self.sender.hosts.pop(message.host)
                        self.sender.hosts.pop("data." + message.host)
                else:
                    logger.warning(f"ignoring {type(message)}")
        if self.sender.hosts:
            logger.warning(
                f"not all hosts exited during grace period: {self.sender.hosts.keys()}, quitting anyway"
            )
