# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Represents information about submitted jobs. The main business logic of `cascade.gateway`"""

import base64
import itertools
import logging
import os
import stat
import subprocess
import uuid
from dataclasses import dataclass
from typing import Iterable

import orjson
import zmq

import cascade.executor.platform as platform
from cascade.controller.report import JobId, JobProgress, JobProgressStarted
from cascade.executor.comms import get_context
from cascade.gateway.api import JobSpec, TroikaSpec
from cascade.low.core import DatasetId
from cascade.low.func import next_uuid

logger = logging.getLogger(__name__)


@dataclass
class Job:
    socket: zmq.Socket
    progress: JobProgress
    last_seen: int
    results: dict[DatasetId, bytes]


# TODO this is a hotfix to not port collide on local jobs. There should be way more
# bind-to-random-port overall, but the current code often needs to use the port number
# before the actual bind happens -- this should be inverted
local_job_port = 12345


def _spawn_troika_singlehost(
    job_spec: JobSpec, addr: str, job_id: JobId, troika: TroikaSpec, troika_config: str
) -> subprocess.Popen:
    script = "#!/bin/bash\n"
    script += f"source {troika.venv}\n"
    for k, v in job_spec.envvars.items():
        script += f"export {k}={v}\n"
    if job_spec.benchmark_name is not None:
        if job_spec.job_instance is not None:
            raise TypeError("specified both benchmark name and job instance")
        script += "python -m cascade.benchmarks local"
        script += f" --job {job_spec.benchmark_name}"
    else:
        if job_spec.job_instance is None:
            raise TypeError("specified neither benchmark name nor job instance")
        job_desc_raw = orjson.dumps(job_spec.job_instance.dict())
        job_desc_enc = base64.b64encode(job_desc_raw).decode("ascii")
        script += f'JOB_ENC="{job_desc_enc}"'
        job_json_path = f"/tmp/cascJob.{job_id}.json"
        script += f'echo "$JOB_ENC" | base64 --decode > {job_json_path}'
        script += "python -m cascade.benchmarks local"
        script += f" --instance {job_json_path}"

    script += (
        f" --workers_per_host {job_spec.workers_per_host} --hosts {job_spec.hosts}"
    )
    script += f" --report_address {addr},{job_id}"
    # NOTE technically not needed to be globally unique, but we cant rely on troika environment isolation...
    global local_job_port
    script += f" --port_base {local_job_port}"
    local_job_port += 1 + job_spec.hosts * job_spec.workers_per_host * 10
    script += "\n"
    script_path = f"/tmp/troikascade.{job_id}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(
        script_path,
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH,
    )
    return subprocess.Popen(
        [
            "troika",
            "-c",
            troika_config,
            "submit",
            "-o",
            f"/tmp/output.{job_id}.txt",
            troika.conn,
            script_path,
        ]
    )


def _spawn_local(
    job_spec: JobSpec, addr: str, job_id: JobId, log_base: str | None
) -> subprocess.Popen:
    base = [
        "python",
        "-m",
        "cascade.benchmarks",
        "local",
    ]
    if job_spec.benchmark_name is not None:
        if job_spec.job_instance is not None:
            raise TypeError("specified both benchmark name and job instance")
        base += ["--job", job_spec.benchmark_name]
    else:
        if job_spec.job_instance is None:
            raise TypeError("specified neither benchmark name nor job instance")
        with open(f"/tmp/{job_id}.json", "wb") as f:
            f.write(orjson.dumps(job_spec.job_instance.dict()))
        base += ["--instance", f"/tmp/{job_id}.json"]

    infra = [
        "--workers_per_host",
        f"{job_spec.workers_per_host}",
        "--hosts",
        f"{job_spec.hosts}",
    ]
    report = ["--report_address", f"{addr},{job_id}"]
    if log_base:
        logs = ["--log_base", f"{log_base}/job.{job_id}"]
    else:
        logs = []
    global local_job_port
    portBase = ["--port_base", str(local_job_port)]
    local_job_port += 1 + job_spec.hosts * job_spec.workers_per_host * 10
    return subprocess.Popen(
        base + infra + report + portBase + logs, env={**os.environ, **job_spec.envvars}
    )


def _spawn_slurm(job_spec: JobSpec, addr: str, job_id: JobId) -> subprocess.Popen:
    extra_vars = {
        "EXECUTOR_HOSTS": str(job_spec.hosts),
        "WORKERS_PER_HOST": str(job_spec.workers_per_host),
        # NOTE put to infra specs
        "SHM_VOL_GB": "64",
        "REPORT_ADDRESS": f"{addr},{job_id}",
    }
    if job_spec.benchmark_name is not None:
        if job_spec.job_instance is not None:
            raise TypeError("specified both benchmark name and job instance")
        extra_vars["JOB"] = job_spec.benchmark_name
    else:
        if job_spec.job_instance is None:
            raise TypeError("specified neither benchmark name nor job instance")
        with open(f"./localConfigs/_tmp/{job_id}.json", "wb") as f:
            f.write(orjson.dumps(job_spec.job_instance.dict()))
        extra_vars["INSTANCE"] = f"./localConfigs/_tmp/{job_id}.json"
    subprocess.run(
        ["cp", "localConfigs/gateway.sh", f"localConfigs/_tmp/{job_id}"], check=True
    )
    with open(f"./localConfigs/_tmp/{job_id}", "a") as f:
        for k, v in itertools.chain(job_spec.envvars.items(), extra_vars.items()):
            f.write(f"export {k}={v}\n")
    return subprocess.Popen(
        ["./scripts/launch_slurm.sh", f"localConfigs/_tmp/{job_id}"]
    )


def _spawn_subprocess(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    log_base: str | None,
    troika_config: str | None,
) -> subprocess.Popen:
    if job_spec.troika is not None:
        if log_base is not None:
            raise ValueError(f"unexpected {log_base=}")
        if troika_config is None:
            raise ValueError("cant spawn troika job without troika config")
        if not job_spec.use_slurm:
            return _spawn_troika_singlehost(
                job_spec, addr, job_id, job_spec.troika, troika_config
            )
        else:
            # TODO create a slurm script like in spawn_slurm, but dont refer to any other file
            raise NotImplementedError

    elif job_spec.use_slurm:
        if log_base is not None:
            raise ValueError(f"unexpected {log_base=}")
        return _spawn_slurm(job_spec, addr, job_id)
    else:
        return _spawn_local(job_spec, addr, job_id, log_base)


class JobRouter:
    def __init__(
        self, poller: zmq.Poller, log_base: str | None, troika_config: str | None
    ):
        self.poller = poller
        self.jobs: dict[str, Job] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.log_base = log_base
        self.troika_config = troika_config

    def spawn_job(self, job_spec: JobSpec) -> JobId:
        job_id = next_uuid(self.jobs.keys(), lambda: str(uuid.uuid4()))
        base_addr = f"tcp://{platform.get_bindabble_self()}"
        socket = get_context().socket(zmq.PULL)
        port = socket.bind_to_random_port(base_addr)
        full_addr = f"{base_addr}:{port}"
        logger.debug(f"will spawn job {job_id} and listen on {full_addr}")
        self.poller.register(socket, flags=zmq.POLLIN)
        self.jobs[job_id] = Job(socket, JobProgressStarted, -1, {})
        self.procs[job_id] = _spawn_subprocess(
            job_spec, full_addr, job_id, self.log_base, self.troika_config
        )
        return job_id

    def progress_of(
        self, job_ids: Iterable[JobId]
    ) -> tuple[dict[JobId, JobProgress], dict[JobId, list[DatasetId]]]:
        if not job_ids:
            job_ids = self.jobs.keys()
        progresses = {job_id: self.jobs[job_id].progress for job_id in job_ids}
        datasets = {
            job_id: list(self.jobs[job_id].results.keys()) for job_id in job_ids
        }
        return progresses, datasets

    def get_result(self, job_id: JobId, dataset_id: DatasetId) -> bytes:
        return self.jobs[job_id].results[dataset_id]

    def maybe_update(
        self, job_id: JobId, progress: JobProgress | None, timestamp: int
    ) -> None:
        if progress is None:
            return
        job = self.jobs[job_id]
        if progress.completed:
            self.poller.unregister(job.socket)
        if progress.failure is not None and job.progress.failure is None:
            job.progress = progress
        elif job.last_seen >= timestamp or job.progress.failure is not None:
            pass
        elif progress.pct is not None:
            job.progress = progress

    def put_result(self, job_id: JobId, dataset_id: DatasetId, result: bytes) -> None:
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
