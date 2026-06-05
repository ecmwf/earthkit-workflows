import argparse
import importlib
import logging
import subprocess
import sys
import tempfile
import time
from multiprocessing import Process
from pathlib import Path
from shlex import quote
from types import ModuleType
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import cascade.gateway.api as api
import cascade.gateway.client as client
from cascade.controller.report import JobId
from cascade.deployment.logging import DefaultLoggingConfig, init_from_cliparam
from cascade.gateway.api import SlurmCluster, SshCluster
from cascade.gateway.server import serve
from cascade.low.core import DatasetId, JobInstanceRich
from cascade.main import run_locally
from cascade.ygg.transport import destroy_context
from integration_tests.jobCases.base import JobSpec

logger = logging.getLogger("cascade.main.client")

REPO_ROOT = Path(__file__).parent.parent
GATEWAY_URL = "tcp://localhost:15355"
GATEWAY_BIND_URL = "tcp://0.0.0.0:15355"
PLAIN_CLUSTER_GATEWAY_SSH_URL = "root@127.0.0.1"
PLAIN_CLUSTER_GATEWAY_SSH_PORT = 2221
PLAIN_CLUSTER_GATEWAY_SSH_KEY = str(REPO_ROOT / "integration_tests" / "deployments" / "plain_cluster" / "ssh" / "id_ed25519")
SLURM_CLUSTER_GATEWAY_SSH_URL = "root@127.0.0.1"
SLURM_CLUSTER_GATEWAY_SSH_PORT = 2224
SLURM_CLUSTER_GATEWAY_SSH_KEY = str(REPO_ROOT / "integration_tests" / "deployments" / "slurm_cluster" / "config" / "ssh_cluster_key")
TRIES_LIMIT = 60
POLL_INTERVAL = 3.0
REMOTE_WHEELS_DIR = "/shared/wheels"
DeploymentKind = Literal["local", "plain_cluster", "slurm_cluster"]


def load_job_case(test_case: str) -> ModuleType:
    return importlib.import_module(f"integration_tests.jobCases.{test_case}")


def spawn_local_gateway(shared_path: str | None) -> Process:
    logging_config_ser = DefaultLoggingConfig.ser_cliparam()
    process = Process(
        target=serve,
        args=(GATEWAY_URL,),
        kwargs={
            "loggingConfigSer": logging_config_ser,
            "report_transport": "tcp",
            "shared_path": shared_path,
        },
    )
    process.start()
    return process


def _ssh_args_for_cluster(deployment_kind: DeploymentKind) -> tuple[list[str], str, int]:
    """Get SSH args, URL, and port for a cluster deployment."""
    if deployment_kind == "plain_cluster":
        ssh_key = PLAIN_CLUSTER_GATEWAY_SSH_KEY
        ssh_url = PLAIN_CLUSTER_GATEWAY_SSH_URL
        ssh_port = PLAIN_CLUSTER_GATEWAY_SSH_PORT
    elif deployment_kind == "slurm_cluster":
        ssh_key = SLURM_CLUSTER_GATEWAY_SSH_KEY
        ssh_url = SLURM_CLUSTER_GATEWAY_SSH_URL
        ssh_port = SLURM_CLUSTER_GATEWAY_SSH_PORT
    else:
        raise ValueError(f"Unsupported cluster deployment: {deployment_kind}")

    return (
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-i",
            ssh_key,
        ],
        ssh_url,
        ssh_port,
    )


class RemoteGatewayHelper:
    # NOTE ugly -- refactor better
    ekw_wheel: str
    wheels_dir: str


def spawn_remote_gateway(deployment_kind: DeploymentKind, shared_path: str | None) -> subprocess.Popen[bytes]:
    """Build wheels, copy to remote gateway node, and launch gateway via SSH."""
    with tempfile.TemporaryDirectory(prefix="cascade_remote_wheel_") as wheel_dir:
        for build_root in [str(REPO_ROOT), str(REPO_ROOT / "integration_tests" / "runtime_pkg")]:
            subprocess.run(
                ["uv", "build", "--wheel", "-o", wheel_dir, build_root],
                check=True,
                capture_output=True,
                text=True,
            )
        wheels = sorted(Path(wheel_dir).glob("*.whl"))
        ekw_wheels = [w for w in wheels if w.name.startswith("earthkit")]
        runtime_wheels = [w for w in wheels if w.name.startswith("cascade_integration_runtime")]
        if len(ekw_wheels) != 1 or len(runtime_wheels) != 1:
            raise RuntimeError(f"Expected exactly one ekw and one runtime wheel, got {wheels=}")
        ekw_wheel = ekw_wheels[0]
        runtime_wheel = runtime_wheels[0]

        ssh_args, ssh_url, ssh_port = _ssh_args_for_cluster(deployment_kind)

        subprocess.run(
            ["ssh", *ssh_args, "-p", str(ssh_port), ssh_url, f"mkdir -p {REMOTE_WHEELS_DIR}"],
            check=True,
        )
        for wheel in [ekw_wheel, runtime_wheel]:
            subprocess.run(
                [
                    "scp",
                    *ssh_args,
                    "-P",
                    str(ssh_port),
                    str(wheel),
                    f"{ssh_url}:{REMOTE_WHEELS_DIR}/{wheel.name}",
                ],
                check=True,
            )

    remote_ekw_wheel = f"{REMOTE_WHEELS_DIR}/{ekw_wheel.name}"
    RemoteGatewayHelper.ekw_wheel = remote_ekw_wheel
    RemoteGatewayHelper.wheels_dir = REMOTE_WHEELS_DIR

    # Build gateway command with optional shared_path
    gateway_cmd_parts = [
        "uv",
        "run",
        "--with",
        quote(remote_ekw_wheel),
        "python",
        "-m",
        "cascade.gateway",
        "--url",
        quote(GATEWAY_BIND_URL),
    ]
    if shared_path is not None:
        gateway_cmd_parts.extend(["--shared-path", quote(shared_path)])

    gateway_cmd = " ".join(gateway_cmd_parts)

    return subprocess.Popen(
        [
            "ssh",
            *ssh_args,
            "-p",
            str(ssh_port),
            ssh_url,
            gateway_cmd,
        ],
        shell=False,
    )


def wait_for_gateway(url: str, retries: int = 20) -> None:
    import zmq

    ctx = zmq.Context()
    try:
        for attempt in range(retries):
            sock = ctx.socket(zmq.REQ)
            sock.set(zmq.LINGER, 500)
            try:
                sock.connect(url)
                sock.send(b'{"clazz": "JobProgressRequest", "job_ids": [], "detailed_report": false}')
                if sock.poll(800, zmq.POLLIN):
                    logger.info("Gateway ready after %s attempt(s)", attempt + 1)
                    return
            finally:
                sock.close()
            time.sleep(0.5)
    finally:
        ctx.destroy()
    raise RuntimeError("Gateway did not become ready in time")


def build_job_spec(job: JobInstanceRich, spc: JobSpec, deployment_kind: DeploymentKind) -> api.JobSpec:
    if deployment_kind == "plain_cluster":
        job = job.model_copy(update={"custom_pip_indices": (RemoteGatewayHelper.wheels_dir,)})
        infra = SshCluster(
            controller_url="root@main",
            worker_urls=[f"root@worker{i}" for i in range(1, spc.hosts + 1)],
            workers_per_host=spc.workers,
            wheel_path=RemoteGatewayHelper.ekw_wheel,
        )
        return api.JobSpec(job_instance=job, envvars={}, infra_spec=infra)
    if deployment_kind == "slurm_cluster":
        job = job.model_copy(update={"custom_pip_indices": (RemoteGatewayHelper.wheels_dir,)})
        infra = SlurmCluster(
            workers_per_host=spc.workers,
            hosts=spc.hosts,
            wheel_path=RemoteGatewayHelper.ekw_wheel,
        )
        return api.JobSpec(job_instance=job, envvars={}, infra_spec=infra)
    raise ValueError(f"unsupported deployment kind for cluster submission: {deployment_kind}")


def collect_outputs(job_id: JobId, datasets: list[DatasetId], job: JobInstanceRich, gateway_url: str) -> dict[DatasetId, Any]:
    outputs: dict[DatasetId, Any] = {}
    for ds_id in datasets:
        result_req = api.ResultRetrievalRequest(job_id=job_id, dataset_id=ds_id)
        result_res = client.request_response(result_req, gateway_url, timeout_ms=5000)
        assert isinstance(result_res, api.ResultRetrievalResponse)
        if result_res.error:
            raise RuntimeError(f"could not retrieve {ds_id}: {result_res.error}")
        outputs[ds_id] = api.decoded_result(result_res, job.jobInstance)
    return outputs


def run_cluster(job_mod: ModuleType, deployment_kind: DeploymentKind, shared_path: str | None) -> None:
    destroy_context()
    gw: Process | subprocess.Popen[bytes]

    # For remote deployments, determine the gateway shared_path (inside container)
    gateway_shared_path: str | None = None
    if deployment_kind in ("plain_cluster", "slurm_cluster"):
        if deployment_kind == "slurm_cluster":
            gateway_shared_path = "/shared"  # Path inside container
        gw = spawn_remote_gateway(deployment_kind, gateway_shared_path)
    else:
        gw = spawn_local_gateway(shared_path)

    job = job_mod.job()
    spc = job_mod.spc()
    js = build_job_spec(job, spc, deployment_kind)
    try:
        wait_for_gateway(GATEWAY_URL)

        logger.info("Submitting job to gateway...")
        submit_req = api.SubmitJobRequest(job=js)
        submit_res = client.request_response(submit_req, GATEWAY_URL, timeout_ms=5000)
        assert isinstance(submit_res, api.SubmitJobResponse), f"unexpected response: {submit_res}"
        assert submit_res.error is None, f"submit error: {submit_res.error}"
        job_id = submit_res.job_id
        if job_id is None:
            raise RuntimeError("gateway returned no job id")
        logger.info("Job submitted: %s", job_id)

        last_progress: api.JobProgressResponse | None = None
        for attempt in range(TRIES_LIMIT):
            prog_req = api.JobProgressRequest(job_ids=[job_id])
            prog_res = client.request_response(prog_req, GATEWAY_URL, timeout_ms=5000)
            assert isinstance(prog_res, api.JobProgressResponse)
            assert prog_res.error is None, f"progress error: {prog_res.error}"
            last_progress = prog_res

            prog = prog_res.progresses.get(job_id)
            logger.info("[%s/%s] progress: %s", attempt + 1, TRIES_LIMIT, prog)
            if prog is not None and getattr(prog, "failure", None) is not None:
                raise RuntimeError(f"Job failed: {prog.failure}")
            if prog is not None and getattr(prog, "pct", None) == "100.00":
                break
            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError(f"Job did not complete within {TRIES_LIMIT * POLL_INTERVAL:.0f}s")

        assert last_progress is not None
        outputs = collect_outputs(job_id, last_progress.datasets.get(job_id, []), job, GATEWAY_URL)
        job_mod.outputOk(outputs)

        shutdown_req = api.ShutdownRequest()
        shutdown_res = client.request_response(shutdown_req, GATEWAY_URL, timeout_ms=5000)
        assert isinstance(shutdown_res, api.ShutdownResponse)
        assert shutdown_res.error is None
        if isinstance(gw, Process):
            gw.join(5)
            if gw.exitcode != 0:
                raise RuntimeError(f"gateway exited with {gw.exitcode}")
        else:
            try:
                gw.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                gw.kill()
                raise RuntimeError("gateway ssh session did not exit in time") from exc
            if gw.returncode != 0:
                raise RuntimeError(f"gateway ssh session exited with {gw.returncode}")
        logger.info("Gateway shut down cleanly.")
    except Exception:
        logger.exception("Integration test failed")
        if isinstance(gw, Process):
            if gw.is_alive():
                gw.kill()
        elif gw.poll() is None:
            gw.kill()
        raise


def run_local(job_mod: ModuleType) -> None:
    job = job_mod.job()
    spc = job_mod.spc()
    outputs = run_locally(job=job, hosts=spc.hosts, workers=spc.workers)
    job_mod.outputOk(outputs)
    logger.info("Local integration test passed with outputs: %s", list(outputs.keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case")
    parser.add_argument("deployment_kind", choices=["local", "plain_cluster", "slurm_cluster"])
    parser.add_argument("--shared-path", default=None)
    args = parser.parse_args()

    init_from_cliparam(DefaultLoggingConfig.ser_cliparam(), "client")
    logger.info("client for integration tests starting")

    job_mod = load_job_case(args.test_case)
    deployment_kind = args.deployment_kind
    shared_path = args.shared_path
    if deployment_kind == "slurm_cluster" and shared_path is None:
        shared_path = str(REPO_ROOT / ".cascade-slurm")

    if deployment_kind == "local":
        run_local(job_mod)
    else:
        run_cluster(job_mod, deployment_kind, shared_path)


if __name__ == "__main__":
    main()
