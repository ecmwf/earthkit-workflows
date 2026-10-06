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
from cascade.low.exceptions import CascadeError, CascadeInfrastructureError, CascadeInternalError
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
        # NOTE we really want to get these acked from gw, otherwise completion is never reported,
        # we go with 3.1s so that 6 retries with 500ms each fit in, +100ms for some slack
        self._ygg.close(timeout_ms=3100, wait_for_all_acks=True)


class Reporter:
    """Reports to the gateway. Intended to be used around the whole controller lifecycle.

    Sending success or failure finalizes the reporter: the channel is closed (awaiting acks) and set to None,
    and any subsequent report is dropped -- the gateway ignores them anyway. Thus `channel is None` means
    either no gateway to report to, or finalized. Leaving the context without having finalized reports failure.
    """

    def __init__(self, report_address: str | None) -> None:
        self.channel = ReporterChannel(report_address) if report_address is not None else None

    def _close(self) -> None:
        # NOTE idempotent on purpose
        if self.channel is not None:
            channel, self.channel = self.channel, None
            channel.close()

    def _finalize(self, report: ControllerReport) -> None:
        if self.channel is None:
            return
        try:
            self.channel.send(report)
        finally:
            self._close()

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

    def send_failure_and_log(self, ex: BaseException) -> None:
        """Assumed to be called from inside an except block to log trace"""
        # NOTE we log this to get the stacktrace into the logfile
        if self.channel is not None:
            logger.exception(f"reporting a controller crash: {ex!r}")
            if not isinstance(ex, CascadeError):
                ex = CascadeInfrastructureError("crash in controller", parent=ex)
            report = ControllerReport(self.channel.job_id, JobProgress.failed(repr(ex)), monotonic_ns(), [])
            self._finalize(report)
        else:
            logger.warning(f"ignoring a controller crash: {ex!r}")

    def success(self) -> None:
        if self.channel is not None:
            logger.debug("reporter sending success")
            self._finalize(ControllerReport(self.channel.job_id, JobProgress.succeeded(), monotonic_ns(), []))
        else:
            # NOTE this warns even in the no-gateway case where its expected, but we dont care
            logger.warning("reporter ignoring a success due to no channel")
