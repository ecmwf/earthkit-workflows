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
from typing_extensions import Self

from cascade.executor.comms import get_context
from cascade.low.core import DatasetId
from cascade.low.execution_context import JobExecutionContext

logger = logging.getLogger(__name__)

JobId = str


@dataclass
class JobProgress:
    completed: bool
    pct: (
        str | None
    )  # number in (0, 1) formatted as {:.2%} without the percent sign -- eg 0.10, 23.68
    failure: str | None

    @classmethod
    def failed(cls, failure: str) -> Self:
        return cls(True, None, failure)

    @classmethod
    def progressed(cls, pct: float) -> Self:
        progress = "{:.2%}".format(pct)[:-1]
        return cls(False, progress, None)

    @classmethod
    def succeeded(cls) -> Self:
        return cls(True, None, None)


JobProgressStarted = JobProgress(False, "0.00", None)


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

    def send_progress(self, context: JobExecutionContext) -> None:
        if self.socket is None:
            return
        pct = 1.0 - context.remaining / context.total
        logger.debug(f"reporting progress {pct=}")
        report = ControllerReport(
            self.job_id, JobProgress.progressed(pct), monotonic_ns(), []
        )
        _send(self.socket, report)

    def send_result(self, dataset: DatasetId, result: bytes) -> None:
        if self.socket is None:
            return
        logger.debug(f"uploading result {dataset=}")
        report = ControllerReport(
            self.job_id, None, monotonic_ns(), [(dataset, result)]
        )
        _send(self.socket, report)

    def send_failure(self, failure: str) -> None:
        if self.socket is None:
            return
        logger.debug(f"reporting failure {failure=}")
        report = ControllerReport(
            self.job_id, JobProgress.failed(failure), monotonic_ns(), []
        )
        _send(self.socket, report)

    def success(self) -> None:
        if self.socket is None:
            return
        logger.debug("reporter sending shutdown")
        report = ControllerReport(
            self.job_id, JobProgress.succeeded(), monotonic_ns(), []
        )
        _send(self.socket, report)
