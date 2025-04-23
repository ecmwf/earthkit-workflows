# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Managing datasets in memory -- inputs and outputs of the executed job
Interaction with shm
"""

import hashlib
import logging
from contextlib import AbstractContextManager
from typing import Any, Literal

import cascade.executor.serde as serde
import cascade.shm.client as shm_client
from cascade.executor.comms import callback
from cascade.executor.msg import BackboneAddress, DatasetPublished
from cascade.low.core import NO_OUTPUT_PLACEHOLDER, DatasetId, WorkerId
from cascade.low.tracing import Microtrace, timer

logger = logging.getLogger(__name__)


def ds2shmid(ds: DatasetId) -> str:
    # we cant use too long file names for shm, https://trac.macports.org/ticket/64806
    h = hashlib.new("md5", usedforsecurity=False)
    h.update((ds.task + ds.output).encode())
    return h.hexdigest()[:24]


class Memory(AbstractContextManager):
    def __init__(self, callback: BackboneAddress, worker: WorkerId) -> None:
        self.local: dict[DatasetId, Any] = {}
        self.bufs: dict[DatasetId, shm_client.AllocatedBuffer] = {}
        self.callback = callback
        self.worker = worker

    def handle(
        self, outputId: DatasetId, outputSchema: str, outputValue: Any, isPublish: bool
    ) -> None:
        if outputId == NO_OUTPUT_PLACEHOLDER:
            if outputValue is not None:
                logger.warning(
                    f"gotten output of type {type(outputValue)} where none was expected, updating annotation"
                )
                outputSchema = "Any"
            else:
                outputValue = "ok"

        # TODO how do we purge from here over time?
        self.local[outputId] = outputValue

        if isPublish:
            logger.debug(f"publishing {outputId}")
            shmid = ds2shmid(outputId)
            result_ser, deser_fun = timer(serde.ser_output, Microtrace.wrk_ser)(
                outputValue, outputSchema
            )
            l = len(result_ser)
            rbuf = shm_client.allocate(shmid, l, deser_fun)
            rbuf.view()[:l] = result_ser
            rbuf.close()
            callback(
                self.callback,
                DatasetPublished(ds=outputId, origin=self.worker, transmit_idx=None),
            )

    def provide(self, inputId: DatasetId, annotation: str) -> Any:
        if inputId not in self.local:
            if inputId in self.bufs:
                raise ValueError(f"internal data corruption for {inputId}")
            shmid = ds2shmid(inputId)
            logger.debug(f"asking for {inputId} via {shmid}")
            buf = shm_client.get(shmid)
            self.bufs[inputId] = buf
            self.local[inputId] = timer(serde.des_output, Microtrace.wrk_deser)(
                buf.view(), annotation, buf.deser_fun
            )

        return self.local[inputId]

    def pop(self, ds: DatasetId) -> None:
        if ds in self.local:
            val = self.local.pop(ds)  # noqa: F841
            del val
        if ds in self.bufs:
            buf = self.bufs.pop(ds)
            buf.close()

    def flush(self) -> None:
        # NOTE poor man's memory management -- just drop those locals that weren't published. Called
        # after every taskSequence. In principle, we could purge some locals earlier, and ideally scheduler
        # would invoke some targeted purges to also remove some published ones earlier (eg, they are still
        # needed somewhere but not here)
        purgeable = [inputId for inputId in self.local if inputId not in self.bufs]
        logger.debug(f"will flush {len(purgeable)} datasets")
        for inputId in purgeable:
            self.local.pop(inputId)

        # NOTE poor man's gpu mem management -- currently torch only. Given the task sequence limitation,
        # this may not be the best place to invoke. Additionally, we may want to check first whether
        # the worker is gpu aware, etc
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.debug(f"cuda mem avail: {free/total:.2%}")
                if free / total < 0.8:
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info()
                    logger.debug(f"cuda mem avail post cache empty: {free/total:.2%}")
                    if free / total < 0.8:
                        logger.warning("cuda mem avail low despite cache empty!")
                        logger.debug(torch.cuda.memory_summary())
        except ImportError:
            return
        except Exception:
            logger.exception("failed to free cuda cache")

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        # this is required so that the Shm can be properly freed, otherwise you get 'pointers cannot be closed'
        del self.local
        for buf in self.bufs.values():
            buf.close()
        return False
