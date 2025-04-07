# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles reporting to gateway"""

import logging
import pickle
from dataclasses import dataclass
from time import monotonic_ns

import zmq

from cascade.executor.comms import get_context
from cascade.low.core import DatasetId
from cascade.scheduler.core import State

logger = logging.getLogger(__name__)

JobId = str
JobProgress = str
JobProgressStarted: JobProgress = "0.00%"
JobProgressShutdown: JobProgress = "Shutdown"


@dataclass
class ControllerReport:
    job_id: JobId
    current_status: JobProgress | None
    timestamp: int
    results: list[tuple[DatasetId, bytes]]


def deserialize(raw: bytes) -> ControllerReport:
    maybe = pickle.loads(raw)
    if isinstance(maybe, ControllerReport):
        return maybe
    else:
        raise TypeError(type(maybe))


def serialize(report: ControllerReport) -> bytes:
    return pickle.dumps(report)


def _send(socket: zmq.Socket, report: ControllerReport) -> None:
    # TODO we need to make sure sending is reliable, ie, retries and acks from gateway
    socket.send(serialize(report))


class Reporter:
    def __init__(self, report_address: str | None) -> None:
        if report_address is None:
            self.socket = None
            return
        address, job_id = report_address.split(",", 1)
        logger.debug(f"initialising reporter with {address=} and {job_id=}")
        self.job_id = job_id
        self.socket = get_context().socket(zmq.PUSH)
        self.socket.connect(address)

    def send_progress(self, state: State) -> None:
        if self.socket is None:
            return
        progress = "{:.2%}".format(1.0 - state.remaining / state.total)
        logger.debug(f"reporting {progress=}")
        report = ControllerReport(self.job_id, progress, monotonic_ns(), [])
        _send(self.socket, report)

    def send_result(self, dataset: DatasetId, result: bytes) -> None:
        if self.socket is None:
            return
        logger.debug(f"uploading reuslt {dataset=}")
        report = ControllerReport(
            self.job_id, None, monotonic_ns(), [(dataset, result)]
        )
        _send(self.socket, report)

    def shutdown(self) -> None:
        if self.socket is None:
            return
        logger.debug("reporter shutting down")
        report = ControllerReport(self.job_id, JobProgressShutdown, monotonic_ns(), [])
        _send(self.socket, report)
