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
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from importlib.metadata import distribution
from pathlib import Path

import orjson

from cascade.controller.report import JobId
from cascade.deployment.logging import LoggingConfig
from cascade.executor.runner.packages import _earthkit_install_spec
from cascade.gateway.api import JobSpec, LocalProcesses, SlurmCluster, SshCluster, TroikaSpec
from cascade.low.exceptions import CascadeInfrastructureError, CascadeUserError

logger = logging.getLogger(__name__)
_SLURM_TEMPLATE_PACKAGE = "cascade.gateway.slurm_templates"

# TODO this is a hotfix to not port collide on local jobs. There should be way more
# bind-to-random-port overall, but the current code often needs to use the port number
# before the actual bind happens -- this should be inverted
local_job_port = 12345


@dataclass
class EkwInstallSpec:
    """Gateway-lifetime resolved earthkit-workflows install spec for remote processes.

    Encodes both the install source and the distribution strategy:

    - ``shared_spec``: a PyPI version pin or shared-filesystem .whl path accessible
      to all remote nodes directly -- no copying needed at job submission time.
    - ``local_spec``: a local path on the gateway to distribute via scp: either an
      already-built .whl file or an editable source directory whose wheel is built
      lazily on first use and cached for all subsequent jobs.

    Exactly one of the two fields must be set.
    """

    shared_spec: str | None = None
    local_spec: str | None = None
    _wheel_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.shared_spec is None) == (self.local_spec is None):
            raise ValueError("exactly one of shared_spec or local_spec must be set")

    def get_local_wheel(self) -> str:
        """Return the local wheel path; builds from source on first call if needed."""
        assert self.local_spec is not None, "get_local_wheel() requires local_spec to be set"
        w = self._wheel_cache
        if w is None:
            w = self.local_spec if self.local_spec.endswith(".whl") else _build_wheel(self.local_spec)
            self._wheel_cache = w
        return w


def _build_wheel(source_path: str) -> str:
    """Build a .whl from a source directory; return the local path to the built wheel."""
    wheel_dir = tempfile.mkdtemp(prefix="cascade_wheel_build_")
    logger.info(f"Building wheel from editable install at {source_path}")
    subprocess.run(
        ["uv", "build", "--wheel", "-o", wheel_dir, source_path],
        check=True,
        capture_output=True,
    )
    wheels = [w for w in os.listdir(wheel_dir) if w.endswith(".whl")]
    if not wheels:
        raise CascadeUserError(f"Failed to build wheel from {source_path}")
    result = os.path.join(wheel_dir, wheels[0])
    logger.info(f"Built wheel: {result}")
    return result


def _stage_wheel_to_shared(wheel_path: str, shared_path: str) -> str:
    """Copy a .whl into the shared staging directory; return the staged path."""
    wheels_dir = Path(shared_path) / "cascade-slurm" / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    dest = wheels_dir / os.path.basename(wheel_path)
    tmp = wheels_dir / f".{os.path.basename(wheel_path)}.tmp"
    shutil.copy2(wheel_path, tmp)
    tmp.replace(dest)
    logger.info(f"Staged wheel to shared storage: {dest}")
    return str(dest)


def prepare_install_spec(shared_path: str | None) -> EkwInstallSpec:
    """Resolve the earthkit-workflows install spec for this gateway's lifetime.

    Examines how earthkit-workflows is installed in the running gateway process and
    produces an EkwInstallSpec that all jobs spawned by this gateway will use.

    - PyPI install: returns a version-pin shared_spec; no file distribution needed.
    - Local .whl or editable install with shared_path: the wheel is built (if needed)
      and staged to shared storage once at startup.
    - Local .whl or editable install without shared_path: stores the source path and
      defers wheel building to the first actual SSH job (cached for subsequent jobs).
    """
    ek_spec = _earthkit_install_spec()

    if not os.path.isabs(ek_spec):
        # PyPI install -- version string, no file to distribute
        return EkwInstallSpec(shared_spec=ek_spec)

    # Absolute path: verify if it is an editable source directory (not already a .whl)
    if not ek_spec.endswith(".whl"):
        try:  # ek_spec is a path but not .whl -- verify it's an editable install
            dist = distribution("earthkit-workflows")
            direct_url_text = dist.read_text("direct_url.json") or "{}"
            editable = orjson.loads(direct_url_text)["dir_info"]["editable"]
        except Exception:
            raise CascadeInfrastructureError(f"unparseable installation spec: {ek_spec}")
        if not editable:
            # NOTE maybe its a zip install or another oddity -- lets raise rather than risk
            raise CascadeInfrastructureError(f"unknown installation spec: {ek_spec}")

    if shared_path is not None:
        # Stage the wheel to shared storage now so all Slurm/SSH-shared nodes can access it
        spec = EkwInstallSpec(local_spec=ek_spec)
        staged = _stage_wheel_to_shared(spec.get_local_wheel(), shared_path)
        return EkwInstallSpec(shared_spec=staged)
    else:
        # No shared disk: hold locally, build deferred to first actual use
        return EkwInstallSpec(local_spec=ek_spec)


def _node_install_spec(
    spec: EkwInstallSpec,
    node_url: str,
    ssh_key_path: str | None,
    ssh_config_path: str | None,
) -> str:
    """Ensure the earthkit-workflows wheel is available on a remote node.

    If the spec uses shared storage, returns the shared path immediately.
    If the spec holds a local wheel, scps it to /tmp/ on the node and returns
    that remote path.
    """
    if spec.shared_spec is not None:
        return spec.shared_spec
    remote_path = f"/tmp/{os.path.basename(spec.get_local_wheel())}"
    subprocess.run(
        ["scp", *_ssh_args(ssh_key_path, ssh_config_path), spec.get_local_wheel(), f"{node_url}:{remote_path}"],
        check=True,
    )
    return remote_path


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
    install_spec: EkwInstallSpec | None,
) -> subprocess.Popen[bytes]:
    if shared_path is None:
        raise CascadeUserError("Slurm jobs require gateway shared_path")
    if install_spec is None or install_spec.shared_spec is None:
        raise CascadeUserError("Slurm jobs require a shared-disk install spec; start the gateway with --shared-path")

    install_spec_to_use = install_spec.shared_spec

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
    exports = {
        **job_spec.envvars,
        "EXECUTOR_HOSTS": str(infra.hosts),
        "WORKERS_PER_HOST": str(infra.workers_per_host),
        "SHM_VOL_GB": "64",
        "INSTANCE": str(job_instance_path),
        "REPORT_ADDRESS": f"{addr},{job_id}",
        "LOGGING_CONFIG_SER": logging_ser,
        "UV_RUN_WITH": install_spec_to_use,
        "CONTROLLER_PORT": str(controller_port),
        "JOB_ROOT": str(job_root),
    }
    config_path = job_root / "config.sh"
    _write_slurm_exports(config_path, exports)

    launcher = scripts_dir / "launch_slurm.sh"
    return subprocess.Popen([str(launcher), str(config_path)])


def _ssh_args(ssh_key_path: str | None, ssh_config_path: str | None) -> list[str]:
    """Common SSH flags: disable host-key prompts, use key if provided."""
    args = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key_path is not None:
        args += ["-i", ssh_key_path]
    if ssh_config_path is not None:
        args += ["-F", ssh_config_path]
    return args


def _get_controller_zmq_hostname(controller_url: str, ssh_key_path: str | None, ssh_config_path: str | None) -> str:
    """SSH to the controller node and ask for its own hostname for ZMQ binding."""
    result = subprocess.run(
        ["ssh", *_ssh_args(ssh_key_path, ssh_config_path), controller_url, 'python3 -c "import socket; print(socket.gethostname())"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write_instance_to_node(
    node_url: str, ssh_key_path: str | None, ssh_config_path: str | None, job_json: bytes, remote_path: str
) -> None:
    """Pipe job instance JSON to a remote file via SSH stdin."""
    subprocess.run(
        ["ssh", *_ssh_args(ssh_key_path, ssh_config_path), node_url, f"cat > {remote_path}"],
        input=job_json,
        check=True,
    )


def _spawn_ssh(
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
        node_ek_specs[node_url] = _node_install_spec(install_spec, node_url, ssh_key, ssh_config)
        _write_instance_to_node(node_url, ssh_key, ssh_config, job_json, remote_instance_path)
        logger.debug(f"Prepared node {node_url}: ek_spec={node_ek_specs[node_url]}")

    # Determine the ZMQ URL the controller will bind on (uses controller's own hostname)
    controller_hostname = _get_controller_zmq_hostname(infra.controller_url, ssh_key, ssh_config)
    global local_job_port
    port = local_job_port
    local_job_port += 1 + (n_hosts + 1) * infra.workers_per_host * 10
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
    ctrl_cmd = ["ssh", *_ssh_args(ssh_key, ssh_config), infra.controller_url] + _build_dist_cmd(
        infra.controller_url, 0, ["--report_address", f"{addr},{job_id}"]
    )
    logger.debug(f"Launching controller: {ctrl_cmd}")
    ctrl_proc = subprocess.Popen(ctrl_cmd, shell=False)

    # Launch executors (idx=1, 2, ...)
    for i, worker_url in enumerate(infra.worker_urls):
        exec_cmd = ["ssh", *_ssh_args(ssh_key, ssh_config), worker_url] + _build_dist_cmd(worker_url, i + 1, [])
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
    install_spec: EkwInstallSpec | None,
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
            return _spawn_slurm(job_spec, addr, job_id, loggingConfig, infra, shared_path, install_spec)
    elif isinstance(infra, LocalProcesses):
        return _spawn_local(job_spec, addr, job_id, loggingConfig, infra)
    elif isinstance(infra, SshCluster):
        return _spawn_ssh(job_spec, addr, job_id, loggingConfig, infra, install_spec)
    else:
        raise CascadeUserError(f"unsupported infra_spec type: {type(infra)}")
