# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning for SshCluster infra spec."""

import logging
import subprocess

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.gateway.api import JobSpec, SshCluster
from cascade.gateway.spawning.common import allocate_port_range, ssh_args
from cascade.gateway.spawning.wheels import EkwInstallSpec, node_install_spec
from cascade.low.exceptions import CascadeUserError

logger = logging.getLogger(__name__)


def get_controller_zmq_hostname(controller_url: str, ssh_key_path: str | None, ssh_config_path: str | None) -> str:
    """SSH to the controller node and ask for its own hostname for ZMQ binding."""
    result = subprocess.run(
        ["ssh", *ssh_args(ssh_key_path, ssh_config_path), controller_url, 'python3 -c "import socket; print(socket.gethostname())"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_instance_to_node(node_url: str, ssh_key_path: str | None, ssh_config_path: str | None, job_json: bytes, remote_path: str) -> None:
    """Pipe job instance JSON to a remote file via SSH stdin."""
    subprocess.run(
        ["ssh", *ssh_args(ssh_key_path, ssh_config_path), node_url, f"cat > {remote_path}"],
        input=job_json,
        check=True,
    )


def spawn_ssh(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: SshCluster,
    install_spec: EkwInstallSpec | None,
) -> subprocess.Popen[bytes]:
    """Spawn controller and executors on remote nodes via SSH.

    The controller is launched on infra.controller_url and executors on each of
    infra.worker_urls. All processes run ``cascade.main dist`` with the appropriate
    index. The job instance is piped to each node via SSH stdin. The earthkit-workflows
    wheel is distributed to each node via the install_spec (shared path or per-node scp).
    """
    if install_spec is None:
        raise CascadeUserError("SSH jobs require a resolved install spec at gateway startup")
    ssh_key = infra.ssh_key_path
    ssh_config = infra.ssh_config_path

    all_nodes = [infra.controller_url] + infra.worker_urls
    n_hosts = len(infra.worker_urls)  # executors only, controller is separate

    # Distribute the wheel (if needed) and job instance to all nodes
    job_json = orjson.dumps(job_spec.job_instance.model_dump())
    remote_instance_path = f"/tmp/cascade_{job_id}.json"
    node_ek_specs: dict[str, str] = {}
    for node_url in all_nodes:
        node_ek_specs[node_url] = node_install_spec(install_spec, node_url, ssh_key, ssh_config)
        write_instance_to_node(node_url, ssh_key, ssh_config, job_json, remote_instance_path)
        logger.debug(f"Prepared node {node_url}: ek_spec={node_ek_specs[node_url]}")

    # Determine the ZMQ URL the controller will bind on (uses controller's own hostname)
    controller_hostname = get_controller_zmq_hostname(infra.controller_url, ssh_key, ssh_config)
    port = allocate_port_range(1 + (n_hosts + 1) * infra.workers_per_host * 10)
    controller_zmq_url = f"tcp://{controller_hostname}:{port}"
    logger.info(f"SSH job {job_id}: controller_zmq_url={controller_zmq_url}, n_executor_hosts={n_hosts}")

    logging_ser = loggingConfig.withContext(f"job_{job_id}").ser_cliparam()

    def _build_dist_cmd(node_url: str, idx: int, report_arg: list[str]) -> list[str]:
        node_ek = node_ek_specs[node_url]
        env_exports = "".join(f"export {k}={v}; " for k, v in job_spec.envvars.items())
        dist_args = [
            "python",
            "-m",
            "cascade.main",
            "dist",
            "--idx",
            str(idx),
            "--controller_url",
            controller_zmq_url,
            "--instance",
            remote_instance_path,
            "--hosts",
            str(n_hosts),
            "--workers_per_host",
            str(infra.workers_per_host),
            "--loggingConfigSer",
            logging_ser,
            *report_arg,
        ]
        # Build the remote shell command: export vars, then uv run --with <ek_spec> <cmd>
        uv_cmd = " ".join(["uv", "run", "--with", node_ek] + dist_args)
        return [env_exports + uv_cmd]

    # Launch controller (idx=0) -- this is the "primary" process we track
    ctrl_cmd = ["ssh", *ssh_args(ssh_key, ssh_config), infra.controller_url] + _build_dist_cmd(
        infra.controller_url, 0, ["--report_address", f"{addr},{job_id}"]
    )
    logger.debug(f"Launching controller: {ctrl_cmd}")
    ctrl_proc = subprocess.Popen(ctrl_cmd, shell=False)

    # Launch executors (idx=1, 2, ...)
    for i, worker_url in enumerate(infra.worker_urls):
        exec_cmd = ["ssh", *ssh_args(ssh_key, ssh_config), worker_url] + _build_dist_cmd(worker_url, i + 1, [])
        logger.debug(f"Launching executor {i + 1} on {worker_url}: {exec_cmd}")
        subprocess.Popen(exec_cmd, shell=False)

    return ctrl_proc
