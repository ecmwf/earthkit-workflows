"""
Handles communication between controller and remote executors
"""

# TODO handle unresponsive executors

from cascade.low.core import TaskId, DatasetId, WorkerId, Environment
from cascade.scheduler.core import DatasetStatus, TaskStatus
from pydantic import BaseModel, Field

Event = DatasetPublished|TaskSucceeded|DatasetTransmitPayload

class Bridge:
    def __init__(self, controller_url: str, expected_executors: int) -> None:
        self.mlistener = Listener(controler_url)
        self.hosts: dict[HostId, zmq.Socket] = {}
        self.daddresses: dict[HostId, BackboneAddress]
        registered = 0
        ctx = zmq.Context()
        self.environment = Environment(workers={})
        while registered < expected_executors:
            for message in self.mlistener.recv_messages:
                if not isinstance(message, ExecutorRegistration):
                    raise TypeError(type(message))
                if message.host in self.hosts or message.host in self.daddresses:
                    raise ValueError(f"double registration of {message.host}")
                socket = ctx.socket(zmq.PUSH)
                socket.set(zmq.LINGER, 3000)
                socket.connect(message.maddress)
                self.hosts[message.host] = socket
                self.daddresses[message.host] = message.daddress
                for worker in message.workers:
                    # TODO proper data
                    self.environment[worker] = Worker(cpu=1, gpu=0, memory_mb=1024)
                registered += 1

    def _send(self, hostId: HostId, message: Message) -> None:
        raw = serde.ser_message(message)
        self.hosts[hostId].send(raw)

    def get_environment(self) -> Environment:
        return self.environment

    def recv_events(self) -> list[Event]:
        events = []
        for message in self.mlistener.recv_messages():
            if isinstance(message, Event):
                events.append(message)
            elif ! # TODO
        return events

    def task_sequence(self, taskSequence: TaskSequence) -> None:
        self._send(taskSequence.worker.host, taskSequence)

    def purge(self, host: HostId, ds: DatasetId) -> None:
        m = DatasetPurge(ds=ds)
        self._send(host, m)

    def transmit(self, ds: DatasetId, source: HostId, target: HostId) -> None:
        m = DatasetTransmitCommand(
            source=source,
            target=target,
            daddress=self.daddresses[target],
            ds=ds,
        )
        self.daddresses[source].send(serde.ser_message(m))

    def fetch(self, ds: DatasetId, source: HostId) -> None:
        m = DatasetTransmitCommand(
            source=source,
            target="controller",
            daddress=self.mlistener.address,
            ds=ds,
        )
        self.daddresses[source].send(serde.ser_message(m))

    def shutdown(self) -> None:
        m = ExecutorShutdown()
        for host in self.hosts.keys():
            self._send(m)
        while self.hosts:
            # we want to consume all those exit messages 
            for message in self.mlistener.recv_messages():
                if isinstance(message, ExecutorExit|ExecutorFailure):
                    self.hosts.pop(message.host)
                else:
                    logger.warning(f"ignoring {type(message)}")
