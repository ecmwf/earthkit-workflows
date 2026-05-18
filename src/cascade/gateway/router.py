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
import subprocess
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
)
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.spawning import spawn_subprocess
from cascade.low.core import DatasetId, TaskId
from cascade.low.func import next_uuid
from cascade.ygg.api import YggNode

logger = logging.getLogger(__name__)


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
        max_concurrent_jobs: int | None,
        max_jobs_history: int = 20,
        max_queue_length: int = 50,
    ):
        self._ygg = ygg
        self.jobs: dict[JobId, Job] = {}
        self.active_jobs = 0
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_jobs_history = max_jobs_history
        self.max_queue_length = max_queue_length
        self.jobs_queue: OrderedDict[JobId, api.JobSpec] = OrderedDict()
        self.procs: dict[JobId, subprocess.Popen] = {}
        self.job_submission_order: list[JobId] = []
        self.completed_jobs = 0
        self.loggingConfig = loggingConfig
        self.troika_config = troika_config

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
        self.procs[job_id] = spawn_subprocess(job_spec, full_addr, job_id, self.loggingConfig, self.troika_config)
        self.active_jobs += 1

    def enqueue_job(self, job_spec: api.JobSpec) -> tuple[JobId | None, str | None]:
        if self.max_queue_length is not None and len(self.jobs_queue) >= self.max_queue_length:
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
        if self.max_jobs_history is None:
            return
        while self.completed_jobs > self.max_jobs_history:
            evicted = False
            index = 0
            while index < len(self.job_submission_order):
                job_id = self.job_submission_order[index]
                job = self.jobs.get(job_id)
                if job is None:
                    self.job_submission_order.pop(index)
                    continue
                if not job.progress.completed:
                    index += 1
                    continue
                del self.jobs[job_id]
                self.procs.pop(job_id, None)
                self.job_submission_order.pop(index)
                self.completed_jobs -= 1
                evicted = True
                break
            if not evicted:
                break

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

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> bytes:
        if job_id not in self.jobs:
            raise KeyError(f"{job_id=} not retained")
        return self.jobs[job_id].results[dataset_id]

    def maybe_update(
        self,
        job_id: JobId,
        progress: JobProgress | None,
        timestamp: int,
        completed_task: TaskId | None = None,
        planned_tasks: set[TaskId] | None = None,
    ) -> bool:
        if progress is None and completed_task is None and not planned_tasks:
            return False
        if job_id not in self.jobs:
            return False
        job = self.jobs[job_id]
        if completed_task is not None:
            job.planned_task_ids.discard(completed_task)
            job.completed_task_ids.add(completed_task)
        if planned_tasks:
            job.planned_task_ids.update(planned_tasks - job.completed_task_ids)
        if progress is None:
            return False
        if timestamp <= job.last_seen:
            return False
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
            self.active_jobs -= 1
            self.completed_jobs += 1
            self.maybe_spawn()
            return True
        return False

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
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

    def shutdown(self):
        for job_id, proc in self.procs.items():
            logger.debug(f"awaiting job {job_id}")
            try:
                proc.terminate()
                proc.wait(2)
            except subprocess.TimeoutExpired:
                logger.error(f"{job_id=} failed to terminate, killing")
                proc.kill()
