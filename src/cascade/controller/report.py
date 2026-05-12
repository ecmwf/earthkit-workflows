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
from typing import NewType

import zmq
from typing_extensions import Self

import cascade.executor.platform as platform
from cascade.executor.comms import ReliableSender, default_message_resend_ms, get_context
from cascade.executor.msg import Ack
from cascade.executor.serde import des_message
from cascade.low.core import DatasetId, HostId, TaskId
from cascade.low.exceptions import CascadeInternalError
from cascade.low.execution_context import JobExecutionContext

logger = logging.getLogger(__name__)

JobId = NewType("JobId", str)


@dataclass
class JobProgress:
    started: bool
    completed: bool
    pct: str | None  # number in (0, 1) formatted as {:.2%} without the percent sign -- eg 0.10, 23.68
    failure: str | None

    @classmethod
    def failed(cls, failure: str) -> Self:
        return cls(True, True, None, failure)

    @classmethod
    def progressed(cls, pct: float) -> Self:
        progress = "{:.2%}".format(pct)[:-1]
        return cls(True, False, progress, None)

    @classmethod
    def succeeded(cls) -> Self:
        return cls(True, True, None, None)


JobProgressStarted = JobProgress(True, False, "0.00", None)
JobProgressEnqueued = JobProgress(False, False, None, None)


@dataclass
class ControllerReport:
    job_id: JobId
    current_status: JobProgress | None
    timestamp: int
    results: list[tuple[DatasetId, bytes]]
    completed_task: TaskId | None = None
    planned_tasks: set[TaskId] | None = None


def deserialize(raw: bytes) -> ControllerReport:
    maybe = pickle.loads(raw)
    if isinstance(maybe, ControllerReport):
        return maybe
    else:
        raise CascadeInternalError(f"failed to deserialize ControllerReport, got {type(maybe)}")


def serialize(report: ControllerReport) -> bytes:
    return pickle.dumps(report)


class ReporterChannel:
    def __init__(self, report_address: str) -> None:
        address, job_id = report_address.split(",", 1)
        logger.debug(f"initialising reporter with {address=} and {job_id=}")
        self.job_id = JobId(job_id)
        self.ack_socket = get_context().socket(zmq.PULL)
        ack_base = f"tcp://{platform.get_bindabble_self()}"
        ack_port = self.ack_socket.bind_to_random_port(ack_base)
        ack_address = f"{ack_base}:{ack_port}"
        self.ack_poller = zmq.Poller()
        self.ack_poller.register(self.ack_socket, flags=zmq.POLLIN)
        self.sender = ReliableSender(ack_address, default_message_resend_ms)
        self.sender.add_host(HostId("gateway"), address)

    def send(self, report: ControllerReport) -> None:
        self.sender.send_raw(HostId("gateway"), serialize(report), "ControllerReport")
        if self.sender.inflight:
            for socket, _ in self.ack_poller.poll(0):
                if socket is not self.ack_socket:
                    continue
                msg_frames = self.ack_socket.recv_multipart()
                if len(msg_frames) != 1:
                    raise CascadeInternalError(f"expected single-frame Ack on report channel, got {len(msg_frames)=}")
                ack = des_message(msg_frames[0])
                if isinstance(ack, Ack):
                    self.sender.ack(ack.idx)
                else:
                    raise CascadeInternalError(f"expected Ack on report channel, got {type(ack)}")
            self.sender.maybe_retry()


class Reporter:
    def __init__(self, report_address: str | None) -> None:
        self.channel = ReporterChannel(report_address) if report_address is not None else None

    def send_task_completed(self, context: JobExecutionContext, completed_task: TaskId) -> None:
        if self.channel is None:
            return
        pct = 1.0 - context.remaining / context.total
        logger.debug(f"reporting progress {pct=}")
        report = ControllerReport(self.channel.job_id, JobProgress.progressed(pct), monotonic_ns(), [], completed_task)
        self.channel.send(report)

    def send_tasks_planned(self, task_ids: set[TaskId]) -> None:
        if self.channel is None:
            return
        logger.debug(f"reporting planned tasks {task_ids=}")
        report = ControllerReport(self.channel.job_id, None, monotonic_ns(), [], None, task_ids)
        self.channel.send(report)

    def send_result(self, dataset: DatasetId, result: bytes) -> None:
        if self.channel is None:
            return
        logger.debug(f"uploading result {dataset=}")
        report = ControllerReport(self.channel.job_id, None, monotonic_ns(), [(dataset, result)])
        self.channel.send(report)

    def send_failure(self, failure: str) -> None:
        if self.channel is None:
            return
        logger.debug(f"reporting failure {failure=}")
        report = ControllerReport(self.channel.job_id, JobProgress.failed(failure), monotonic_ns(), [])
        self.channel.send(report)

    def success(self) -> None:
        if self.channel is None:
            return
        logger.debug("reporter sending shutdown")
        report = ControllerReport(self.channel.job_id, JobProgress.succeeded(), monotonic_ns(), [])
        self.channel.send(report)
