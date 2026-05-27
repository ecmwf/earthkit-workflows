# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spawning new processes for each SubmitJobRequest, locally or remotely"""

import logging
import os
import shlex
import stat
import subprocess
import tempfile
from importlib import resources
from pathlib import Path

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.executor.runner.packages import _earthkit_install_spec
from cascade.gateway.api import JobSpec, LocalProcesses, SlurmCluster, SshCluster, TroikaSpec
from cascade.low.exceptions import CascadeUserError

logger = logging.getLogger(__name__)
_SLURM_TEMPLATE_PACKAGE = "cascade.gateway.slurm_templates"

# TODO this is a hotfix to not port collide on local jobs. There should be way more
# bind-to-random-port overall, but the current code often needs to use the port number
# before the actual bind happens -- this should be inverted
local_job_port = 12345


def _spawn_troika_singlehost(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    infra: SlurmCluster,
    troika: TroikaSpec,
    troika_config: str,
) -> subprocess.Popen[bytes]:
    script = "#!/bin/bash\n"
    script += f"source {troika.venv}\n"
    for k, v in job_spec.envvars.items():
        script += f"export {k}={v}\n"

    job_json_path = f"/tmp/cascJob.{job_id}.json"
    with open(job_json_path, "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    script += f"python -m cascade.main local"
    script += f" --instance {job_json_path}"

    script += f" --workers_per_host {infra.workers_per_host} --hosts {infra.hosts}"
    script += f" --report_address {addr},{job_id}"
    # NOTE technically not needed to be globally unique, but we cant rely on troika environment isolation...
    global local_job_port
    script += f" --port_base {local_job_port}"
    local_job_port += 1 + infra.hosts * infra.workers_per_host * 10
    script += "\n"
    script_path = f"/tmp/troikascade.{job_id}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(
        script_path,
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
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
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: LocalProcesses,
) -> subprocess.Popen[bytes]:
    base = [
        "python",
        "-m",
        "cascade.main",
        "local",
    ]

    with open(f"/tmp/{job_id}.json", "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))
    base += ["--instance", f"/tmp/{job_id}.json"]

    infra_args = [
        "--workers_per_host",
        f"{infra.workers_per_host}",
        "--hosts",
        f"{infra.hosts}",
    ]
    report = ["--report_address", f"{addr},{job_id}"]
    logs = ["--loggingConfigSer", loggingConfig.withContext(f"job_{job_id}").ser_cliparam()]
    global local_job_port
    portBase = ["--port_base", str(local_job_port)]
    local_job_port += 1 + infra.hosts * infra.workers_per_host * 10
    return subprocess.Popen(base + infra_args + report + portBase + logs, env={**os.environ, **job_spec.envvars}, close_fds=True)


def _stage_text_resource(resource_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = resources.files(_SLURM_TEMPLATE_PACKAGE).joinpath(resource_name).read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp_path.replace(dest)


def _write_slurm_exports(dest: Path, exports: dict[str, str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in exports.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    body = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def _stage_slurm_scripts(shared_path: str) -> Path:
    slurm_root = Path(shared_path) / "cascade-slurm"
    scripts_dir = slurm_root / "scripts"
    _stage_text_resource("launch_slurm.sh", scripts_dir / "launch_slurm.sh")
    _stage_text_resource("slurm_entrypoint.sh", scripts_dir / "slurm_entrypoint.sh")
    return slurm_root


def _spawn_slurm(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: SlurmCluster,
    shared_path: str | None,
) -> subprocess.Popen[bytes]:
    if shared_path is None:
        raise CascadeUserError("Slurm jobs require gateway shared_path")

    slurm_root = _stage_slurm_scripts(shared_path)
    job_root = slurm_root / "jobs" / str(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    scripts_dir = slurm_root / "scripts"

    global local_job_port
    controller_port = local_job_port
    local_job_port += 1 + (infra.hosts + 1) * infra.workers_per_host * 10

    job_instance_path = job_root / "instance.json"
    with open(job_instance_path, "wb") as f:
        f.write(orjson.dumps(job_spec.job_instance.model_dump()))

    logging_ser = loggingConfig.withContext(f"job_{job_id}").ser_cliparam()
    ekw_install_spec = _earthkit_install_spec()
    exports = {
        **job_spec.envvars,
        "EXECUTOR_HOSTS": str(infra.hosts),
        "WORKERS_PER_HOST": str(infra.workers_per_host),
        "SHM_VOL_GB": "64",
        "INSTANCE": str(job_instance_path),
        "REPORT_ADDRESS": f"{addr},{job_id}",
        "LOGGING_CONFIG_SER": logging_ser,
        "EKW_INSTALL_SPEC": ekw_install_spec,
        "CASCADE_EKW_INSTALL_SPEC": ekw_install_spec,
        "CONTROLLER_PORT": str(controller_port),
        "JOB_ROOT": str(job_root),
    }
    config_path = job_root / "config.sh"
    _write_slurm_exports(config_path, exports)

    launcher = scripts_dir / "launch_slurm.sh"
    return subprocess.Popen([str(launcher), str(config_path)])


def _ssh_args(ssh_key_path: str | None) -> list[str]:
    """Common SSH flags: disable host-key prompts, use key if provided."""
    args = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key_path is not None:
        args += ["-i", ssh_key_path]
    return args


def _get_controller_zmq_hostname(controller_url: str, ssh_key_path: str | None) -> str:
    """SSH to the controller node and ask for its own hostname for ZMQ binding."""
    result = subprocess.run(
        ["ssh", *_ssh_args(ssh_key_path), controller_url, 'python3 -c "import socket; print(socket.gethostname())"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _prepare_ekw_on_node(node_url: str, ssh_key_path: str | None, ek_spec: str, wheel_path: str | None) -> str:
    """Ensure earthkit-workflows is available on node, return the install spec to use with uv."""
    if wheel_path is not None:
        remote_wheel = f"/tmp/{os.path.basename(wheel_path)}"
        subprocess.run(
            ["scp", *_ssh_args(ssh_key_path), wheel_path, f"{node_url}:{remote_wheel}"],
            check=True,
        )
        return remote_wheel
    return ek_spec


def _write_instance_to_node(node_url: str, ssh_key_path: str | None, job_json: bytes, remote_path: str) -> None:
    """Pipe job instance JSON to a remote file via SSH stdin."""
    subprocess.run(
        ["ssh", *_ssh_args(ssh_key_path), node_url, f"cat > {remote_path}"],
        input=job_json,
        check=True,
    )


def _spawn_ssh(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    infra: SshCluster,
) -> subprocess.Popen[bytes]:
    """Spawn controller and executors on remote nodes via SSH.

    The controller is launched on infra.controller_url and executors on each of
    infra.worker_urls. All processes run ``cascade.main dist`` with the appropriate
    index. The job instance is piped to each node via SSH stdin. If earthkit-workflows
    is installed as an editable/local package, a wheel is built locally and copied to
    each node before launching.
    """
    ssh_key = infra.ssh_key_path

    # Determine earthkit-workflows install spec for remote nodes
    ek_spec = _earthkit_install_spec()
    wheel_path: str | None = None
    if os.path.isabs(ek_spec):
        # Editable / local install -- build a wheel to copy to remote nodes
        wheel_dir = tempfile.mkdtemp(prefix="cascade_ssh_wheel_")
        logger.info(f"Building wheel from editable install at {ek_spec}")
        subprocess.run(
            ["uv", "build", "--wheel", "-o", wheel_dir, ek_spec],
            check=True,
            capture_output=True,
        )
        wheels = [w for w in os.listdir(wheel_dir) if w.endswith(".whl")]
        if not wheels:
            raise CascadeUserError(f"Failed to build wheel from {ek_spec}")
        wheel_path = os.path.join(wheel_dir, wheels[0])
        logger.info(f"Built wheel: {wheel_path}")

    all_nodes = [infra.controller_url] + infra.worker_urls
    n_hosts = len(infra.worker_urls)  # executors only, controller is separate

    # Distribute the wheel (if needed) and job instance to all nodes
    job_json = orjson.dumps(job_spec.job_instance.model_dump())
    remote_instance_path = f"/tmp/cascade_{job_id}.json"
    node_ek_specs: dict[str, str] = {}
    for node_url in all_nodes:
        node_ek_specs[node_url] = _prepare_ekw_on_node(node_url, ssh_key, ek_spec, wheel_path)
        _write_instance_to_node(node_url, ssh_key, job_json, remote_instance_path)
        logger.debug(f"Prepared node {node_url}: ek_spec={node_ek_specs[node_url]}")

    # Determine the ZMQ URL the controller will bind on (uses controller's own hostname)
    controller_hostname = _get_controller_zmq_hostname(infra.controller_url, ssh_key)
    global local_job_port
    port = local_job_port
    local_job_port += 1 + (n_hosts + 1) * infra.workers_per_host * 10
    controller_zmq_url = f"tcp://{controller_hostname}:{port}"
    logger.info(f"SSH job {job_id}: controller_zmq_url={controller_zmq_url}, n_executor_hosts={n_hosts}")

    logging_ser = loggingConfig.withContext(f"job_{job_id}").ser_cliparam()

    def _build_dist_cmd(node_url: str, idx: int, report_arg: list[str]) -> list[str]:
        node_ek = node_ek_specs[node_url]
        # Propagate envvars from job spec, plus the install spec override so that
        # worker venv creation on the remote node reuses the same wheel we copied.
        all_exports = {**job_spec.envvars, "CASCADE_EKW_INSTALL_SPEC": node_ek}
        env_exports = "".join(f"export {k}={v}; " for k, v in all_exports.items())
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
    ctrl_cmd = ["ssh", *_ssh_args(ssh_key), infra.controller_url] + _build_dist_cmd(
        infra.controller_url, 0, ["--report_address", f"{addr},{job_id}"]
    )
    logger.debug(f"Launching controller: {ctrl_cmd}")
    ctrl_proc = subprocess.Popen(ctrl_cmd, shell=False)

    # Launch executors (idx=1, 2, ...)
    for i, worker_url in enumerate(infra.worker_urls):
        exec_cmd = ["ssh", *_ssh_args(ssh_key), worker_url] + _build_dist_cmd(worker_url, i + 1, [])
        logger.debug(f"Launching executor {i + 1} on {worker_url}: {exec_cmd}")
        subprocess.Popen(exec_cmd, shell=False)

    return ctrl_proc


def spawn_subprocess(
    job_spec: JobSpec,
    addr: str,
    job_id: JobId,
    loggingConfig: LoggingConfig,
    troika_config: str | None,
    shared_path: str | None,
) -> subprocess.Popen[bytes]:
    infra = job_spec.infra_spec
    if isinstance(infra, SlurmCluster):
        if infra.troika is not None:
            # TODO support logging config properly
            if troika_config is None:
                raise CascadeUserError("cant spawn troika job without troika config")
            return _spawn_troika_singlehost(job_spec, addr, job_id, infra, infra.troika, troika_config)
        else:
            # TODO support logging config properly
            return _spawn_slurm(job_spec, addr, job_id, loggingConfig, infra, shared_path)
    elif isinstance(infra, LocalProcesses):
        return _spawn_local(job_spec, addr, job_id, loggingConfig, infra)
    elif isinstance(infra, SshCluster):
        return _spawn_ssh(job_spec, addr, job_id, loggingConfig, infra)
    else:
        raise CascadeUserError(f"unsupported infra_spec type: {type(infra)}")
