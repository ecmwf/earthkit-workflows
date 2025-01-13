"""
An extension of executor that handles its own zmq socket for DatasetTransmit messages.
The reason for the split is to not block the message zmq socket with potentially
large data object.
"""

# NOTE test coverage handled in `test_executor.py` as well

import logging
import logging.config
from concurrent.futures import Future, ThreadPoolExecutor, Executor as PythonExecutor, wait, FIRST_COMPLETED

from cascade.executor.runner import ds2shmid
from cascade.executor.comms import Listener, callback, send_data
from cascade.executor.msg import BackboneAddress, DatasetTransmitCommand, DatasetTransmitPayload, DatasetTransmitFailure, DatasetPublished
from cascade.low.func import assert_never
from cascade.low.tracing import mark, TransmitLifecycle, label
import cascade.shm.client as shm_client
import cascade.shm.api as shm_api

logger = logging.getLogger(__name__)


class DataServer:
    def __init__(self, maddress: BackboneAddress, daddress: BackboneAddress, host: str, shm_port: int, logging_config: dict):
        logging.config.dictConfig(logging_config)
        self.host = host
        label("host", self.host)
        self.maddress = maddress
        self.dlistener = Listener(daddress)
        self.terminating = False
        shm_api.publish_client_port(shm_port)
        self.cap = 2
        self.ds_proc_tp: PythonExecutor = ThreadPoolExecutor(max_workers=self.cap)
        self.futs_queue: list[Future] = []

    def maybe_clean(self) -> None:
        """Cleans out completed futures, waits if too many in progress"""
        while True:
            i = 0
            while i < len(self.futs_queue):
                if self.futs_queue[i].done():
                    ex = self.futs_queue[i].exception()
                    if ex:
                        callback(self.maddress, DatasetTransmitFailure(host=self.host))
                    self.futs_queue.pop(i)
                else:
                    i += 1
            if len(self.futs_queue) < self.cap:
                return
            wait(self.futs_queue, return_when=FIRST_COMPLETED)

    def store_payload(self, payload: DatasetTransmitPayload) -> None:
        try:
            l = len(payload.value)
            try:
                buf = shm_client.allocate(key=ds2shmid(payload.ds), l=l)
            except shm_client.ConflictError as e:
                # NOTE this branch is for situations where the controller issued redundantly two transmits
                logger.warning(f"store of {payload.ds} failed, presumably already computed; continuing")
                mark({"dataset": payload.ds.task, "action": TransmitLifecycle.unloaded, "target": self.host, "mode": "redundant"})
                return
            buf.view()[:l] = payload.value
            buf.close()
            callback(self.maddress, DatasetPublished(ds=payload.ds, host=self.host, from_transmit=True))
            mark({"dataset": payload.ds.task, "action": TransmitLifecycle.unloaded, "target": self.host, "mode": "remote"})
        except Exception as e:
            logger.exception("failed to store payload of {payload.ds}, reporting up")
            callback(self.maddress, DatasetTransmitFailure(host=self.host))

    def send_payload(self, command: DatasetTransmitCommand):
        try:
            if command.target == self.host or command.source != self.host:
                raise ValueError(f"invalid {command=}")
            buf = shm_client.get(key=ds2shmid(command.ds))
            mark({"dataset": command.ds.task, "action": TransmitLifecycle.loaded, "target": command.target, "source": self.host, "mode": "remote"})
            if command.target == "controller":
                # NOTE this is due to controller using single socket for messages/data. We thus
                # use non-optimized send, but it doesnt really matter
                payload = DatasetTransmitPayload(ds=command.ds, value=bytes(buf.view()))
                callback(command.daddress, payload)
            else:
                payload = DatasetTransmitPayload(ds=command.ds, value=buf.view())
                send_data(command.daddress, payload)
                del payload # to enforce deletion of exported pointer
            buf.close()
            logger.debug(f"payload for {command} sent")
        except Exception as e:
            logger.exception("failed to send payload for {command}, reporting up")
            callback(self.maddress, DatasetTransmitFailure(host=self.host))

    def recv_loop(self) -> None:
        # NOTE atm we don't terminate explicitly, rather parent kills us. But we may want to exit cleanly instead
        while not self.terminating:
            try:
                m = self.dlistener.recv_dmessage(timeout_sec=None)
                if isinstance(m, DatasetTransmitCommand):
                    mark({"dataset": m.ds.task, "action": TransmitLifecycle.started, "target": m.target})
                    self.maybe_clean()
                    fut = self.ds_proc_tp.submit(self.send_payload, m)
                    self.futs_queue.append(fut)
                elif isinstance(m, DatasetTransmitPayload):
                    mark({"dataset": m.ds.task, "action": TransmitLifecycle.received, "target": self.host})
                    self.maybe_clean()
                    fut = self.ds_proc_tp.submit(self.store_payload, m)
                    self.futs_queue.append(fut)
                elif m is None:
                    raise ValueError(f"no data despite infinite timeout")
                else:
                    assert_never(m)
            except Exception as e:
                # TODO report some issue and continue? Or just shutdown, since executor is monitoring this proc?
                raise

def start_data_server(maddress: BackboneAddress, daddress: BackboneAddress, host: str, shm_port: int, logging_config: dict):
    server = DataServer(maddress, daddress, host, shm_port, logging_config)
    server.recv_loop()
