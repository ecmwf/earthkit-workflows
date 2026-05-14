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

from typing_extensions import Self

import cascade.executor.platform as platform
from cascade.low.core import DatasetId, TaskId
from cascade.low.exceptions import CascadeInternalError
from cascade.low.execution_context import JobExecutionContext
from cascade.ygg.api import YggNode
from cascade.ygg.types import HostEndpoints

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
        bind_base = f"tcp://{platform.get_bindabble_self()}"
        self._ygg = YggNode(f"{bind_base}:*")
        self._ygg.register_host("gateway", HostEndpoints(control=address))

    def send(self, report: ControllerReport) -> None:
        self._ygg.send_message_to_host("gateway", serialize(report), lane="control")
        self._ygg.poll_messages(timeout_ms=0)
        self._ygg.retry_outstanding()

    def close(self) -> None:
        # NOTE we really want to get these acked from gw, otherwise completion is never reported
        self._ygg.close(timeout_ms=5000, wait_for_all_acks=True)


class Reporter:
    def __init__(self, report_address: str | None) -> None:
        self.channel = ReporterChannel(report_address) if report_address is not None else None

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()

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
