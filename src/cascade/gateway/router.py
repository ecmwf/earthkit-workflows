# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Manages job submissions:
- routes the SubmitJobRequest to the appropriate spawn command
- exposes a port for jobs to report progress, keeps these reports in memory
- exposes a port for jobs to upload outputs, keeps these outputs in memory
- directly responds to JobProgressRequest and ResultRetrievalRequest from memory
"""

import logging
import os
import signal
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import cascade.gateway.api as api
from cascade.controller.report import (
    JobId,
    JobProgress,
    JobProgressEnqueued,
    JobProgressStarted,
    deserialize,
)
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.spawning import EkwInstallSpec, SpawnedJob, spawn_subprocess
from cascade.low.core import DatasetId, TaskId
from cascade.low.exceptions import CascadeUserError
from cascade.low.func import next_uuid
from cascade.ygg.api import YggNode

logger = logging.getLogger(__name__)

# total time given to jobs to terminate gracefully before they get killed
job_termination_grace_s = 10.0
job_termination_poll_s = 0.1


@dataclass
class Job:
    progress: JobProgress
    last_seen: int
    results: dict[DatasetId, bytes]
    completed_task_ids: set[TaskId]
    planned_task_ids: set[TaskId]


class JobRouter:
    def __init__(
        self,
        ygg: YggNode,
        loggingConfig: LoggingConfig,
        troika_config: str | None,
        shared_path: str | None,
        install_spec: EkwInstallSpec | None,
        max_concurrent_jobs: int | None,
        max_jobs_history: int = 20,
        max_queue_length: int = 50,
    ):
        if max_queue_length <= 0:
            raise CascadeUserError(f"{max_queue_length=} must be > 0")
        if max_concurrent_jobs is not None and max_concurrent_jobs <= 0:
            raise CascadeUserError(f"{max_concurrent_jobs=} must be > 0 when set")
        if max_jobs_history < 0:
            raise CascadeUserError(f"{max_jobs_history=} must be >= 0")
        self._ygg = ygg
        self.jobs: dict[JobId, Job] = {}
        self.active_jobs = 0
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_jobs_history = max_jobs_history
        self.max_queue_length = max_queue_length
        self.jobs_queue: OrderedDict[JobId, api.JobSpec] = OrderedDict()
        # NOTE may contain jobs already evicted from `jobs`, if their processes are still running
        self.procs: dict[JobId, SpawnedJob] = {}
        self.job_submission_order: list[JobId] = []
        self.completed_jobs = 0
        self.loggingConfig = loggingConfig
        self.troika_config = troika_config
        self.shared_path = shared_path
        self.install_spec = install_spec

    def maybe_spawn(self) -> None:
        if not self.jobs_queue:
            return
        if self.max_concurrent_jobs is not None and self.active_jobs >= self.max_concurrent_jobs:
            logger.debug(f"already running {self.active_jobs}, no spawn")
            return

        job_id, job_spec = self.jobs_queue.popitem(False)
        full_addr = self._ygg.control_address
        logger.debug(f"will spawn job {job_id} and listen on {full_addr}")
        self.jobs[job_id] = Job(JobProgressStarted, -1, {}, set(), set())
        self.procs[job_id] = spawn_subprocess(
            job_spec,
            full_addr,
            job_id,
            self.loggingConfig,
            self.troika_config,
            self.shared_path,
            self.install_spec,
        )
        self.active_jobs += 1

    def enqueue_job(self, job_spec: api.JobSpec) -> tuple[JobId | None, str | None]:
        if len(self.jobs_queue) >= self.max_queue_length:
            return None, f"queue full: {len(self.jobs_queue)} jobs already queued"
        job_id = next_uuid(
            set(self.jobs.keys()).union(self.jobs_queue.keys()).union(self.job_submission_order),
            lambda: JobId(str(uuid.uuid4())),
        )
        self.jobs_queue[job_id] = job_spec
        self.job_submission_order.append(job_id)
        self.maybe_spawn()
        return job_id, None

    def maybe_evict_old_jobs(self) -> None:
        index = 0
        while self.completed_jobs > self.max_jobs_history and index < len(self.job_submission_order):
            job_id = self.job_submission_order[index]
            job = self.jobs.get(job_id)
            if job is None:
                self.job_submission_order.pop(index)
                continue
            if not job.progress.completed:
                index += 1
                continue
            del self.jobs[job_id]
            spawned = self.procs.get(job_id)
            if spawned is not None and all(proc.poll() is not None for proc in spawned.procs):
                # NOTE otherwise we keep it, to be terminated at shutdown
                self.procs.pop(job_id)
            self.job_submission_order.pop(index)
            self.completed_jobs -= 1
        if self.completed_jobs > self.max_jobs_history:
            logger.warning(
                "unable to evict enough completed jobs: max_jobs_history=%s max_concurrent_jobs=%s job_submission_order_len=%s",
                self.max_jobs_history,
                self.max_concurrent_jobs,
                len(self.job_submission_order),
            )

    def job_became_completed(self) -> None:
        self.active_jobs -= 1
        self.completed_jobs += 1
        self.maybe_spawn()
        self.maybe_evict_old_jobs()

    def progress_of(self, job_ids: Iterable[JobId], detailed_report: bool = False) -> api.JobProgressResponse:
        if not job_ids:
            job_ids = set(self.jobs.keys()).union(self.jobs_queue.keys())
        progresses = {}
        for job_id in job_ids:
            if job_id in self.jobs:
                progresses[job_id] = self.jobs[job_id].progress
            elif job_id in self.jobs_queue:
                progresses[job_id] = JobProgressEnqueued
            else:
                progresses[job_id] = None
        datasets = {job_id: list(self.jobs[job_id].results.keys()) for job_id in job_ids if job_id in self.jobs}
        completed_task_ids: dict[JobId, list[TaskId]] | None = None
        planned_task_ids: dict[JobId, list[TaskId]] | None = None
        if detailed_report:
            completed_task_ids = {job_id: list(self.jobs[job_id].completed_task_ids) for job_id in job_ids if job_id in self.jobs}
            planned_task_ids = {job_id: list(self.jobs[job_id].planned_task_ids) for job_id in job_ids if job_id in self.jobs}
        return api.JobProgressResponse(
            progresses=progresses,
            datasets=datasets,
            queue_length=len(self.jobs_queue),
            error=None,
            completed_task_ids=completed_task_ids,
            planned_task_ids=planned_task_ids,
        )

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> tuple[bytes | None, str | None]:
        if job_id not in self.jobs:
            return None, f"{job_id=} not retained"
        if dataset_id not in self.jobs[job_id].results:
            return None, f"{dataset_id=} not found for {job_id=}"
        return self.jobs[job_id].results[dataset_id], None

    def maybe_update(
        self,
        job_id: JobId,
        progress: JobProgress | None,
        timestamp: int,
        completed_task: TaskId | None = None,
        planned_tasks: set[TaskId] | None = None,
    ) -> None:
        if progress is None and completed_task is None and not planned_tasks:
            return
        if job_id not in self.jobs:
            return
        job = self.jobs[job_id]
        if completed_task is not None:
            job.planned_task_ids.discard(completed_task)
            job.completed_task_ids.add(completed_task)
        if planned_tasks:
            job.planned_task_ids.update(planned_tasks - job.completed_task_ids)
        if progress is None:
            return
        if job.progress.completed:
            # NOTE we dont allow eg a late failure to override success, or a terminated job to be revived
            return
        if timestamp <= job.last_seen:
            return
        job.last_seen = timestamp
        was_completed = job.progress.completed
        if progress.failure is not None and job.progress.failure is None:
            job.progress = progress
        elif job.progress.failure is not None:
            pass
        elif progress.pct is not None:
            job.progress = progress
        if progress.completed and not was_completed:
            if progress.failure is None:
                job.progress = JobProgress(job.progress.started, True, job.progress.pct, job.progress.failure)
            self.job_became_completed()

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
        if job_id not in self.jobs:
            logger.warning(f"result {dataset_id=} for unknown {job_id=}, ignoring")
            return
        if dataset_id not in self.jobs[job_id].results:
            self.jobs[job_id].results[dataset_id] = result

    def delete_results(self, delete_map: dict[JobId, list[DatasetId]]) -> list[str]:
        if not delete_map:
            for job in self.jobs.values():
                job.results = {}
            return []
        errs = []
        for job_id, datasets in delete_map.items():
            if job_id not in self.jobs:
                errs.append(f"{job_id=} not found")
                continue
            if not datasets:
                self.jobs[job_id].results = {}
                continue
            for dataset in datasets:
                if dataset not in self.jobs[job_id].results:
                    errs.append(f"{dataset=} not found for {job_id=}")
                else:
                    del self.jobs[job_id].results[dataset]
        return errs

    def handle_reports(self) -> None:
        """Consumes all controller reports that have arrived so far"""
        while msgs := self._ygg.poll_messages(timeout_ms=0):
            for msg in msgs:
                report = deserialize(msg.payload)
                logger.debug(f"received controller message {report}")
                for dataset_id, result in report.results:
                    self.put_result(report.job_id, dataset_id, result)
                self.maybe_update(report.job_id, report.current_status, report.timestamp, report.completed_task, report.planned_tasks)

    def _mark_terminated(self, job_id: JobId) -> None:
        job = self.jobs.get(job_id)
        if job is None or job.progress.completed:
            return
        job.progress = JobProgress.failed("terminated by gateway")
        self.job_became_completed()

    def _kill_remaining(self, job_id: JobId, spawned: SpawnedJob) -> None:
        for proc in spawned.procs:
            if proc.poll() is None:
                logger.warning(f"{job_id=} process {proc.pid} failed to terminate in time, killing")
                proc.kill()
            proc.wait()
        if spawned.pgid is not None:
            # NOTE whatever remains in the group, eg executors orphaned by a killed controller. If the group is
            # already empty, we get ProcessLookupError. Reuse of the pgid by an unrelated group is very unlikely
            try:
                os.killpg(spawned.pgid, signal.SIGKILL)
                # NOTE usually just lingering helpers such as forkserver or resource tracker
                logger.info(f"{job_id=} had remaining processes in group {spawned.pgid}, killed")
            except (ProcessLookupError, PermissionError):
                pass

    def shutdown(self, only_these: list[JobId] | None) -> list[str]:
        """Terminates the selected jobs, or all jobs (both queued and spawned) if None. Blocks until all
        are terminated -- first signals all, then awaits them for up to `job_termination_grace_s` in total,
        then kills the remaining ones. Idempotent. Returns errors, such as for unknown job ids.
        """
        # TODO we signal the locally spawned processes, which for the local spawn is the controller itself,
        # but for remote spawns (ssh, troika) its only the connection/submission client, and the remote job
        # keeps running. Rework this to send a termination message to the controller via Ygg, once we have
        # a bidirectional channel. That would also allow us to not block here, but keep per-job deadlines
        errors: list[str] = []
        if only_these is None:
            queued = list(self.jobs_queue.keys())
            spawned = list(self.procs.keys())
        else:
            queued = [job_id for job_id in only_these if job_id in self.jobs_queue]
            spawned = [job_id for job_id in only_these if job_id in self.procs]
            for job_id in only_these:
                if job_id not in self.jobs_queue and job_id not in self.jobs and job_id not in self.procs:
                    errors.append(f"{job_id=} not found")

        # NOTE we dequeue first, so that no new job gets spawned as the terminated ones complete
        for job_id in queued:
            self.jobs_queue.pop(job_id)
            self.jobs[job_id] = Job(JobProgress.failed("terminated by gateway before start"), -1, {}, set(), set())
            self.completed_jobs += 1

        to_terminate = {job_id: self.procs[job_id] for job_id in spawned}
        for job_id, spawned_job in to_terminate.items():
            for proc in spawned_job.procs:
                logger.debug(f"terminating {job_id=} process {proc.pid}")
                proc.terminate()  # NOTE no-op if already reaped

        deadline = time.monotonic() + job_termination_grace_s
        is_running = lambda: any(proc.poll() is None for spawned_job in to_terminate.values() for proc in spawned_job.procs)
        while is_running() and time.monotonic() < deadline:
            # NOTE we keep consuming reports, so that the terminating controllers get their final reports acked
            try:
                self.handle_reports()
            except Exception:
                logger.exception("failed to handle reports during shutdown, continuing")
            time.sleep(job_termination_poll_s)

        for job_id, spawned_job in to_terminate.items():
            try:
                self._kill_remaining(job_id, spawned_job)
            except Exception:
                logger.exception(f"failed to kill remaining processes of {job_id=}, continuing")
            self.procs.pop(job_id, None)
            self._mark_terminated(job_id)
        self.maybe_evict_old_jobs()
        return errors
