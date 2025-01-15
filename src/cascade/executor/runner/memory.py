"""
Managing datasets in memory -- inputs and outputs of the executed job
Interaction with shm
"""

from contextlib import AbstractContextManager
from typing import Any, Literal
import logging
import hashlib

from cascade.low.tracing import Microtrace, timer
import cascade.shm.client as shm_client
from cascade.low.core import DatasetId, TaskId, WorkerId, NO_OUTPUT_PLACEHOLDER
from cascade.executor.msg import BackboneAddress, DatasetPublished
from cascade.executor.comms import callback
import cascade.executor.serde as serde

logger = logging.getLogger(__name__)

def ds2shmid(ds: DatasetId) -> str:
    # we cant use too long file names for shm, https://trac.macports.org/ticket/64806
    h = hashlib.new("md5", usedforsecurity=False)
    h.update((ds.task + ds.output).encode())
    return h.hexdigest()[:24]

class Memory(AbstractContextManager):
    def __init__(self, callback: BackboneAddress, worker: WorkerId, publish: set[DatasetId]) -> None:
        self.local: dict[DatasetId, Any] = {}
        self.bufs: dict[DatasetId, shm_client.AllocatedBuffer] = {}
        self.publish = publish
        self.callback = callback
        self.worker = worker

    def handle(self, outputId: DatasetId, outputSchema: str, outputValue: Any) -> None:
        if outputId == NO_OUTPUT_PLACEHOLDER:
            if outputValue is not None:
                logger.warning(f"gotten output of type {type(outputValue)} where none was expected, updating annotation")
                outputSchema = "Any"
            else:
                outputValue = "ok"

        # TODO how do we purge from here over time?
        self.local[outputId] = outputValue

        if outputId in self.publish:
            logger.debug(f"publishing {outputId}")
            shmid = ds2shmid(outputId)
            result_ser = timer(serde.ser_output, Microtrace.wrk_ser)(outputValue, outputSchema)
            l = len(result_ser)
            rbuf = shm_client.allocate(shmid, l)
            rbuf.view()[:l] = result_ser
            rbuf.close()
            callback(
                self.callback,
                DatasetPublished(ds=outputId, host=self.worker.host, transmit_idx=None),
            )

    def provide(self, inputId: DatasetId, annotation: str) -> Any:
        if inputId not in self.local:
            if inputId in self.bufs:
                raise ValueError(f"internal data corruption for {inputId}")
            shmid = ds2shmid(inputId)
            logger.debug(f"asking for {inputId} via {shmid}")
            buf = shm_client.get(shmid)
            self.bufs[inputId] = buf
            self.local[inputId] = timer(serde.des_output, Microtrace.wrk_deser)(buf.view(), annotation)

        return self.local[inputId]

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        # TODO allow for purging via ext events -- drop from local, close in bufs

        # this is required so that the Shm can be properly freed, otherwise you get 'pointers cannot be closed'
        del self.local
        for buf in self.bufs.values():
            buf.close()
        return False


