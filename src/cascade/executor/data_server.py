# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""An extension of executor that handles its own zmq socket for DatasetTransmit messages.
The reason for the split is to not block the message zmq socket with potentially
large data object.
"""

# NOTE test coverage handled in `test_executor.py` as well

import logging
import logging.config
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED
from concurrent.futures import Executor as PythonExecutor
from concurrent.futures import Future, ThreadPoolExecutor, wait
from time import time_ns

import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from cascade.executor.comms import Listener, callback, send_data
from cascade.executor.msg import (
    Ack,
    BackboneAddress,
    DatasetPublished,
    DatasetPurge,
    DatasetTransmitCommand,
    DatasetTransmitFailure,
    DatasetTransmitPayload,
    DatasetTransmitPayloadHeader,
    Syn,
)
from cascade.executor.runner.memory import ds2shmid
from cascade.low.core import DatasetId
from cascade.low.func import assert_never
from cascade.low.tracing import TransmitLifecycle, label, mark

logger = logging.getLogger(__name__)


class DataServer:
    def __init__(
        self,
        maddress: BackboneAddress,
        daddress: BackboneAddress,
        host: str,
        shm_port: int,
        logging_config: dict,
    ):
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
        self.futs_in_progress: dict[
            DatasetTransmitCommand | DatasetTransmitPayload, Future
        ] = {}
        self.awaiting_confirmation: dict[int, tuple[DatasetTransmitCommand, int]] = {}
        self.invalid: set[DatasetId] = (
            set()
        )  # for preventing out-of-order stores/transmits for datasets that have already been purged
        self.acks: set[int] = (
            set()
        )  # it could happen that Ack arrives before respective Future finishes, so we need to store separately
        # TODO the two above should be eventually purged, see comms.Listener.acked for a similar concern

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
                        callback(
                            self.maddress,
                            DatasetTransmitFailure(host=self.host, detail=detail),
                        )
                    else:
                        result = fut.result()
                        if isinstance(key, DatasetTransmitCommand):
                            self.awaiting_confirmation[key.idx] = (key, result)
                    self.futs_in_progress.pop(key)
            if len(self.futs_in_progress) < self.cap:
                return
            wait(self.futs_in_progress.values(), return_when=FIRST_COMPLETED)

    def store_payload(self, payload: DatasetTransmitPayload) -> int:
        try:
            l = len(payload.value)
            try:
                buf = shm_client.allocate(
                    key=ds2shmid(payload.header.ds),
                    l=l,
                    deser_fun=payload.header.deser_fun,
                )
            except shm_client.ConflictError as e:
                # NOTE this branch is for situations where the controller issued redundantly two transmits
                logger.warning(
                    f"store of {payload.header.ds} failed with {e}, presumably already computed; continuing"
                )
                mark(
                    {
                        "dataset": repr(payload.header.ds),
                        "action": TransmitLifecycle.unloaded,
                        "target": self.host,
                        "mode": "redundant",
                    }
                )
                return time_ns()
            buf.view()[:l] = payload.value
            buf.close()
            callback(
                self.maddress,
                DatasetPublished(
                    ds=payload.header.ds,
                    origin=self.host,
                    transmit_idx=payload.header.confirm_idx,
                ),
            )
            mark(
                {
                    "dataset": repr(payload.header.ds),
                    "action": TransmitLifecycle.unloaded,
                    "target": self.host,
                    "mode": "remote",
                }
            )
        except Exception as e:
            logger.exception(
                "failed to store payload of {payload.header.ds}, reporting up"
            )
            callback(
                self.maddress,
                DatasetTransmitFailure(
                    host=self.host,
                    detail=f"{payload.header.confirm_idx}, {payload.header.ds} -> {repr(e)}",
                ),
            )
        return (
            time_ns()
        )  # not actually consumed but uniform signature with send_payload simplifies typing

    def send_payload(self, command: DatasetTransmitCommand) -> int:
        buf: None | shm_client.AllocatedBuffer = None
        payload: None | DatasetTransmitPayload = None
        try:
            if command.target == self.host or command.source != self.host:
                raise ValueError(f"invalid {command=}")
            buf = shm_client.get(key=ds2shmid(command.ds))
            mark(
                {
                    "dataset": repr(command.ds),
                    "action": TransmitLifecycle.loaded,
                    "target": command.target,
                    "source": self.host,
                    "mode": "remote",
                }
            )
            header = DatasetTransmitPayloadHeader(
                confirm_address=self.daddress,
                confirm_idx=command.idx,
                ds=command.ds,
                deser_fun=buf.deser_fun,
            )
            payload = DatasetTransmitPayload(header, value=buf.view())
            syn = Syn(command.idx, self.dlistener.address)
            send_data(command.daddress, payload, syn)
            logger.debug(f"payload for {command} sent")
        except Exception as e:
            logger.exception(f"failed to send payload for {command}, reporting up")
            callback(
                self.maddress,
                DatasetTransmitFailure(
                    host=self.host, detail=f"{repr(command)} -> {repr(e)}"
                ),
            )
        finally:
            if payload is not None:
                del payload  # to enforce deletion of exported pointer, so that buffer can be closed
            if buf is not None:
                buf.close()
        return time_ns()

    def recv_loop(self) -> None:
        # NOTE atm we don't terminate explicitly, rather parent kills us. But we may want to exit cleanly instead
        resend_grace_ms = 4_000  # a bit longer than in the regular message case, data messages take longer to receive
        # TODO consider breaking down large transmits into multiple smaller messages
        while not self.terminating:
            try:
                self.maybe_clean()
                for m in self.dlistener.recv_messages(resend_grace_ms):
                    logger.debug(f"received message {type(m)}")
                    if isinstance(m, DatasetTransmitCommand):
                        if m.idx in self.awaiting_confirmation:
                            raise ValueError(
                                f"transmit idx conflict: {m}, {self.awaiting_confirmation[m.idx]}"
                            )
                        if m.ds in self.invalid:
                            raise ValueError(
                                f"unexpected transmit command {m} as the dataset was already purged"
                            )
                        mark(
                            {
                                "dataset": repr(m.ds),
                                "action": TransmitLifecycle.started,
                                "target": m.target,
                            }
                        )
                        self.awaiting_confirmation[m.idx] = (m, -1)
                        fut = self.ds_proc_tp.submit(self.send_payload, m)
                        self.futs_in_progress[m] = fut
                    elif isinstance(m, DatasetTransmitPayload):
                        if m.header.ds in self.invalid:
                            logger.warning(
                                f"ignoring transmit payload {m.header} as the dataset was already purged"
                            )
                            continue
                        mark(
                            {
                                "dataset": repr(m.header.ds),
                                "action": TransmitLifecycle.received,
                                "target": self.host,
                            }
                        )
                        fut = self.ds_proc_tp.submit(self.store_payload, m)
                        self.futs_in_progress[m] = fut
                    elif isinstance(m, Ack):
                        logger.debug(f"confirmed transmit {m.idx}")
                        self.acks.add(m.idx)
                    elif isinstance(m, DatasetPurge):
                        # we need to handle potential commands transmitting this dataset, as otherwise they'd fail
                        to_wait = []
                        for commandProg, fut in self.futs_in_progress.items():
                            if isinstance(commandProg, DatasetTransmitCommand):
                                val = commandProg.ds
                            elif isinstance(commandProg, DatasetTransmitPayload):
                                val = commandProg.header.ds
                            else:
                                assert_never(commandProg)
                            if m.ds == val:
                                logger.debug(
                                    f"waiting for future of {type(commandProg)} of {val}"
                                )
                                to_wait.append(fut)
                        wait(self.futs_in_progress.values(), return_when=ALL_COMPLETED)
                        self.maybe_clean()
                        invalidated = []
                        for idx, (command, _) in self.awaiting_confirmation.items():
                            if command.ds == m.ds:
                                invalidated.append(idx)
                        for idx in invalidated:
                            self.awaiting_confirmation.pop(idx)
                        shm_client.purge(ds2shmid(m.ds))
                        self.invalid.add(m.ds)
                    else:
                        raise NotImplementedError(type(m))

                # TODO ideally, we would be able to re-use the ReliableSender here
                # but we need to be careful because of a/ thread pool b/ opened shm
                # buffers. The current impl has the downside of reading from shm for
                # every retry, instead of keeping it in mem until sent
                watermark = time_ns() - resend_grace_ms * 1_000_000
                queue = []
                for idx, (_, at) in self.awaiting_confirmation.items():
                    if at > 0 and at < watermark:
                        queue.append(idx)
                for e in queue:
                    self.maybe_clean()
                    command = self.awaiting_confirmation[e][0]
                    if command in self.futs_in_progress:
                        raise ValueError(
                            f"asked for retry of {command}, but said future still in progress"
                        )
                    elif command.idx in self.acks:
                        self.awaiting_confirmation.pop(e)

                    elif command.ds in self.invalid:
                        logger.warning(
                            f"{command} won't be retried as the dataset has been purged; assuming lost"
                        )
                        self.awaiting_confirmation.pop(e)
                    else:
                        logger.warning(f"submitting a retry of {command}")
                        fut = self.ds_proc_tp.submit(self.send_payload, command)
                        self.futs_in_progress[command] = fut
                        self.awaiting_confirmation[e] = (command, -1)
            except:
                # NOTE do something more clean here? Not critical since we monitor this process anyway
                raise


def start_data_server(
    maddress: BackboneAddress,
    daddress: BackboneAddress,
    host: str,
    shm_port: int,
    logging_config: dict,
):
    server = DataServer(maddress, daddress, host, shm_port, logging_config)
    server.recv_loop()
