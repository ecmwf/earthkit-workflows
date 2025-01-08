"""
Handles communication between controller and remote executors
"""

# TODO handle unresponsive executors

import logging

from cascade.low.core import TaskId, DatasetId, WorkerId, Environment, HostId, Worker
from cascade.low.func import assert_never
from cascade.scheduler.core import DatasetStatus, TaskStatus
from pydantic import BaseModel, Field
from cascade.executor.msg import Message, TaskSequence, TaskFailure, TaskSuccess, DatasetPublished, DatasetPurge, DatasetTransmitCommand, DatasetTransmitPayload, ExecutorFailure, ExecutorExit, ExecutorRegistration, ExecutorShutdown, DatasetTransmitFailure, BackboneAddress
import cascade.executor.serde as serde
from cascade.executor.comms import Listener
import zmq

logger = logging.getLogger(__name__)

Event = DatasetPublished|TaskSuccess|DatasetTransmitPayload
ToShutdown = TaskFailure|ExecutorFailure|DatasetTransmitFailure|ExecutorExit
Unsupported = TaskSequence|DatasetPurge|DatasetTransmitCommand|ExecutorRegistration|ExecutorShutdown

class Bridge:
    def __init__(self, controller_url: str, expected_executors: int) -> None:
        self.mlistener = Listener(controller_url)
        self.hosts: dict[HostId, zmq.Socket] = {}
        self.daddresses: dict[HostId, tuple[BackboneAddress, zmq.Socket]] = {}
        registered = 0
        ctx = zmq.Context()
        self.environment = Environment(workers={})
        while registered < expected_executors:
            for message in self.mlistener.recv_messages():
                if not isinstance(message, ExecutorRegistration):
                    raise TypeError(type(message))
                if message.host in self.hosts or message.host in self.daddresses:
                    raise ValueError(f"double registration of {message.host}")
                msocket = ctx.socket(zmq.PUSH)
                msocket.set(zmq.LINGER, 3000)
                msocket.connect(message.maddress)
                self.hosts[message.host] = msocket
                dsocket = ctx.socket(zmq.PUSH)
                dsocket.set(zmq.LINGER, 3000)
                dsocket.connect(message.daddress)
                self.daddresses[message.host] = (message.daddress, dsocket)
                for worker in message.workers:
                    # TODO proper parameters
                    self.environment.workers[worker] = Worker(cpu=1, gpu=0, memory_mb=1024)
                registered += 1

    def _send(self, hostId: HostId, message: Message) -> None:
        raw = serde.ser_message(message)
        self.hosts[hostId].send(raw)

    def get_environment(self) -> Environment:
        return self.environment

    def recv_events(self) -> list[Event]:
        try:
            events = []
            shutdown_reason: None|Exception|Message = None
            for message in self.mlistener.recv_messages():
                if isinstance(message, Event):
                    events.append(message)
                elif isinstance(message, ToShutdown):
                    logger.critical(f"received failure {message=}, proceeding with a shutdown")
                    if isinstance(message, ExecutorExit|ExecutorFailure) and message.host in self.hosts:
                        self.hosts.pop(message.host)
                    shutdown_reason = message
                elif isinstance(message, Unsupported):
                    logger.critical(f"received unexpected {message=}, proceeding with a shutdown")
                    shutdown_reason = message
                else:
                    assert_never(message)
            if shutdown_reason is None:
                return events
        except Exception as e:
            logger.exception(f"gotten exception, proceeding with a shutdown")
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
            daddress=self.daddresses[target][0],
            ds=ds,
        )
        self.daddresses[source][1].send(serde.ser_message(m))

    def fetch(self, ds: DatasetId, source: HostId) -> None:
        m = DatasetTransmitCommand(
            source=source,
            target="controller",
            daddress=self.mlistener.address,
            ds=ds,
        )
        self.daddresses[source][1].send(serde.ser_message(m))

    def shutdown(self) -> None:
        m = ExecutorShutdown()
        for host in self.hosts.keys():
            self._send(host, m)
        while self.hosts:
            # we want to consume all those exit messages 
            for message in self.mlistener.recv_messages():
                if isinstance(message, ExecutorExit|ExecutorFailure):
                    if message.host in self.hosts:
                        self.hosts.pop(message.host)
                else:
                    logger.warning(f"ignoring {type(message)}")
