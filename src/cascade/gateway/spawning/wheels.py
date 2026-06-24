# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Earthkit-workflows wheel resolution and distribution for remote nodes."""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import distribution
from pathlib import Path

import orjson

from cascade.executor.runner.packages import _earthkit_install_spec
from cascade.gateway.spawning.common import ssh_args
from cascade.low.exceptions import CascadeInfrastructureError, CascadeUserError

logger = logging.getLogger(__name__)


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
            w = self.local_spec if self.local_spec.endswith(".whl") else build_wheel(self.local_spec)
            self._wheel_cache = w
        return w


def build_wheel(source_path: str) -> str:
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


def stage_wheel_to_shared(wheel_path: str, shared_path: str) -> str:
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
        staged = stage_wheel_to_shared(spec.get_local_wheel(), shared_path)
        return EkwInstallSpec(shared_spec=staged)
    else:
        # No shared disk: hold locally, build deferred to first actual use
        return EkwInstallSpec(local_spec=ek_spec)


def node_install_spec(
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
    wheel = spec.get_local_wheel()
    remote_path = f"/tmp/{os.path.basename(wheel)}"
    subprocess.run(
        ["scp", *ssh_args(ssh_key_path, ssh_config_path), wheel, f"{node_url}:{remote_path}"],
        check=True,
    )
    return remote_path
