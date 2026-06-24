# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for gateway.router update semantics."""

import os
from time import monotonic_ns

from cascade.controller.report import JobId, JobProgress, JobProgressStarted
from cascade.deployment.logging import DefaultLoggingConfig
from cascade.gateway.router import Job, JobRouter
from cascade.low.core import TaskId
from cascade.ygg.api import YggNode
from cascade.ygg.types import RetryPolicy, YggConfig


def _make_ygg() -> YggNode:
    config = YggConfig(
        control=RetryPolicy(retry_interval_ms=50, max_retries=3),
    )
    address = f"ipc:///tmp/cascadeTestGatewayRouter.{os.getpid()}.socket"
    return YggNode(address, config=config)


def _make_router(ygg: YggNode) -> JobRouter:
    return JobRouter(
        ygg=ygg,
        loggingConfig=DefaultLoggingConfig,
        troika_config=None,
        shared_path=None,
        install_spec=None,
        max_concurrent_jobs=None,
    )


def _make_job() -> Job:
    return Job(
        progress=JobProgressStarted,
        last_seen=-1,
        results={},
        completed_task_ids=set(),
        planned_task_ids=set(),
    )


def test_maybe_update_ignores_empty_update() -> None:
    with _make_ygg() as ygg:
        router = _make_router(ygg)
        job_id = JobId("job-1")
        router.jobs[job_id] = _make_job()
        initial = router.jobs[job_id]

        router.maybe_update(job_id, progress=None, timestamp=monotonic_ns(), completed_task=None, planned_tasks=None)

        assert router.jobs[job_id] == initial


def test_maybe_update_tracks_planned_and_completed_tasks() -> None:
    with _make_ygg() as ygg:
        router = _make_router(ygg)
        job_id = JobId("job-tasks")
        router.jobs[job_id] = _make_job()
        t1 = TaskId("task-1")
        t2 = TaskId("task-2")

        router.maybe_update(job_id, progress=None, timestamp=monotonic_ns(), planned_tasks={t1, t2})
        assert router.jobs[job_id].planned_task_ids == {t1, t2}
        assert router.jobs[job_id].completed_task_ids == set()

        router.maybe_update(job_id, progress=None, timestamp=monotonic_ns(), completed_task=t1)
        assert router.jobs[job_id].planned_task_ids == {t2}
        assert router.jobs[job_id].completed_task_ids == {t1}


def test_maybe_update_updates_progress_and_timestamp() -> None:
    with _make_ygg() as ygg:
        router = _make_router(ygg)
        job_id = JobId("job-progress")
        router.jobs[job_id] = _make_job()
        ts = monotonic_ns()

        router.maybe_update(job_id, JobProgress.progressed(0.5), ts)
        assert router.jobs[job_id].last_seen == ts
        assert router.jobs[job_id].progress.pct == "50.00"


def test_maybe_update_ignores_stale_timestamp() -> None:
    with _make_ygg() as ygg:
        router = _make_router(ygg)
        job_id = JobId("job-stale")
        router.jobs[job_id] = _make_job()
        ts = monotonic_ns()

        router.maybe_update(job_id, JobProgress.progressed(0.7), ts)
        router.maybe_update(job_id, JobProgress.progressed(0.2), ts - 1)
        assert router.jobs[job_id].progress.pct == "70.00"


def test_maybe_update_active_jobs_decremented_on_completion() -> None:
    with _make_ygg() as ygg:
        router = _make_router(ygg)
        job_id = JobId("job-dec")
        router.jobs[job_id] = _make_job()
        router.active_jobs = 1

        router.maybe_update(job_id, JobProgress.succeeded(), monotonic_ns())
        assert router.active_jobs == 0
