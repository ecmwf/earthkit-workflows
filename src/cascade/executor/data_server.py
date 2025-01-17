"""
An extension of executor that handles its own zmq socket for DatasetTransmit messages.
The reason for the split is to not block the message zmq socket with potentially
large data object.
"""

# NOTE test coverage handled in `test_executor.py` as well

import logging
import logging.config
from concurrent.futures import Future, ThreadPoolExecutor, Executor as PythonExecutor, wait, FIRST_COMPLETED
from time import time_ns

from cascade.executor.runner.memory import ds2shmid
from cascade.executor.comms import Listener, callback, send_data
from cascade.executor.msg import BackboneAddress, DatasetTransmitCommand, DatasetTransmitPayloadHeader, DatasetTransmitPayload, DatasetTransmitFailure, DatasetPublished, DatasetTransmitConfirm
from cascade.low.func import assert_never
from cascade.low.tracing import mark, TransmitLifecycle, label
import cascade.shm.client as shm_client
import cascade.shm.api as shm_api

logger = logging.getLogger(__name__)

resend_grace = 2 * 60 * 1_000_000_000 # two minutes

class DataServer:
    def __init__(self, maddress: BackboneAddress, daddress: BackboneAddress, host: str, shm_port: int, logging_config: dict):
        logging.config.dictConfig(logging_config)
        self.host = host
        label("host", self.host)
        self.maddress = maddress
        self.daddress = daddress
        self.dlistener = Listener(daddress)
        self.terminating = False
        shm_api.publish_client_port(shm_port)
        self.cap = 2
        self.ds_proc_tp: PythonExecutor = ThreadPoolExecutor(max_workers=self.cap)
        self.futs_in_progress: dict[DatasetTransmitCommand|DatasetTransmitPayload, Future] = {}
        self.awaiting_confirmation: dict[int, tuple[DatasetTransmitCommand, int]] = {}

    def maybe_clean(self) -> None:
        """Cleans out completed futures, waits if too many in progress"""
        while True:
            keys = list(self.futs_in_progress.keys())
            for key in keys:
                fut = self.futs_in_progress[key]
                if fut.done():
                    ex = fut.exception()
                    if ex:
                        detail = f"{repr(key)} -> {repr(ex)}"
                        callback(self.maddress, DatasetTransmitFailure(host=self.host, detail=detail))
                    self.futs_in_progress.pop(key)
            if len(self.futs_in_progress) < self.cap:
                return
            wait(self.futs_in_progress.values(), return_when=FIRST_COMPLETED)

    def store_payload(self, payload: DatasetTransmitPayload) -> None:
        try:
            l = len(payload.value)
            try:
                buf = shm_client.allocate(key=ds2shmid(payload.header.ds), l=l, deser_fun=payload.header.deser_fun)
            except shm_client.ConflictError as e:
                # NOTE this branch is for situations where the controller issued redundantly two transmits
                logger.warning(f"store of {payload.header.ds} failed, presumably already computed; continuing")
                mark({"dataset": repr(payload.header.ds), "action": TransmitLifecycle.unloaded, "target": self.host, "mode": "redundant"})
                return
            buf.view()[:l] = payload.value
            buf.close()
            callback(payload.header.confirm_address, DatasetTransmitConfirm(idx=payload.header.confirm_idx))
            callback(self.maddress, DatasetPublished(ds=payload.header.ds, host=self.host, transmit_idx=payload.header.confirm_idx))
            mark({"dataset": repr(payload.header.ds), "action": TransmitLifecycle.unloaded, "target": self.host, "mode": "remote"})
        except Exception as e:
            logger.exception("failed to store payload of {payload.header.ds}, reporting up")
            callback(self.maddress, DatasetTransmitFailure(host=self.host, detail=f"{payload.header.confirm_idx}, {payload.header.ds} -> {repr(e)}"))

    def send_payload(self, command: DatasetTransmitCommand):
        buf: None|shm_client.AllocatedBuffer = None
        payload: None|DatasetTransmitPayload = None
        try:
            if command.target == self.host or command.source != self.host:
                raise ValueError(f"invalid {command=}")
            buf = shm_client.get(key=ds2shmid(command.ds))
            mark({"dataset": repr(command.ds), "action": TransmitLifecycle.loaded, "target": command.target, "source": self.host, "mode": "remote"})
            header = DatasetTransmitPayloadHeader(confirm_address=self.daddress, confirm_idx=command.idx, ds=command.ds, deser_fun=buf.deser_fun)
            if command.target == "controller":
                # NOTE this is due to controller using single socket for messages/data. We thus
                # use non-optimized send, but it doesnt really matter since not really used atm
                payload = DatasetTransmitPayload(header, value=bytes(buf.view()))
                callback(command.daddress, payload)
            else:
                payload = DatasetTransmitPayload(header, value=buf.view())
                send_data(command.daddress, payload)
                # TODO await a reply, retry if not come
            logger.debug(f"payload for {command} sent")
        except Exception as e:
            logger.exception("failed to send payload for {command}, reporting up")
            callback(self.maddress, DatasetTransmitFailure(host=self.host, detail=f"{repr(command)} -> {repr(e)}"))
        finally:
            if payload is not None:
                del payload # to enforce deletion of exported pointer, so that buffer can be closed
            if buf is not None:
                buf.close()

    def recv_loop(self) -> None:
        # NOTE atm we don't terminate explicitly, rather parent kills us. But we may want to exit cleanly instead
        while not self.terminating:
            try:
                self.maybe_clean()
                m = self.dlistener.recv_dmessage()
                if m is not None:
                    logger.debug(f"received message {type(m)}")
                if isinstance(m, DatasetTransmitCommand):
                    if m.idx in self.awaiting_confirmation:
                        raise ValueError(f"transmit idx conflict: {m}, {self.awaiting_confirmation[m.idx]}")
                    mark({"dataset": repr(m.ds), "action": TransmitLifecycle.started, "target": m.target})
                    fut = self.ds_proc_tp.submit(self.send_payload, m)
                    self.awaiting_confirmation[m.idx] = (m, time_ns())
                    self.futs_in_progress[m] = fut
                elif isinstance(m, DatasetTransmitPayload):
                    mark({"dataset": repr(m.header.ds), "action": TransmitLifecycle.received, "target": self.host})
                    fut = self.ds_proc_tp.submit(self.store_payload, m)
                    self.futs_in_progress[m] = fut
                elif isinstance(m, DatasetTransmitConfirm):
                    if m.idx not in self.awaiting_confirmation:
                        logger.warning(f"unexpected confirmation: {m.idx}")
                    else:
                        self.awaiting_confirmation.pop(m.idx)
                elif m is None:
                    pass
                else:
                    assert_never(m)

                watermark = time_ns() - resend_grace
                queue = []
                for idx, (_, at) in self.awaiting_confirmation.items():
                    if at < watermark:
                        queue.append(at)
                for e in queue:
                    self.maybe_clean()
                    command = self.awaiting_confirmation[e][0]
                    if command in self.futs_in_progress:
                        logger.warning(f"asked for retry of {command}, but said future still in progress")
                    else:
                        logger.warning(f"submitting a retry of {command}")
                        fut = self.ds_proc_tp.submit(self.send_payload, command)
                        self.futs_in_progress[command] = fut
                        self.awaiting_confirmation[e] = (command, time_ns())
            except Exception as e:
                # NOTE do something more clean here? Not critical since we monitor this process anyway
                raise

def start_data_server(maddress: BackboneAddress, daddress: BackboneAddress, host: str, shm_port: int, logging_config: dict):
    server = DataServer(maddress, daddress, host, shm_port, logging_config)
    server.recv_loop()
