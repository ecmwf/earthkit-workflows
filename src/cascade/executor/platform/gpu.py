# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gpu capability detection and per-worker assignment.

The public entrypoint is `get_gpu_info`, which is cross-platform and accounts for:
 - partitioned (MIG) gpus, where a single physical gpu line in `nvidia-smi -L` may
   correspond to multiple independently addressable devices
 - unified memory setups (eg macOS, or some linux configurations), where every
   worker may use the gpu regardless of any device indexing
"""

import os
import subprocess
import sys
from dataclasses import dataclass

_NVIDIA_SMI_LIST_GPUS = ["nvidia-smi", "-L"]
_NVIDIA_SMI_QUERY_MEMORY = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"]
_UNIFIED_MEMORY_MARKERS = ("N/A", "Not Supported")
_CUDA_VISIBLE_DEVICES_ENVVAR = "CUDA_VISIBLE_DEVICES"


def _list_gpus_output() -> str:
    """Invokes `nvidia-smi -L`, returning its stdout. Raises on any failure."""
    return subprocess.run(_NVIDIA_SMI_LIST_GPUS, check=True, capture_output=True).stdout.decode("ascii")


def _query_memory_output() -> str:
    """Invokes `nvidia-smi --query-gpu=memory.total`, returning its stdout. Raises on any failure.

    NOTE this call crashes on some driver versions when the gpu is MIG-partitioned -- only
    invoke this once it is known that no partitioning is in place.
    """
    return subprocess.run(_NVIDIA_SMI_QUERY_MEMORY, check=True, capture_output=True).stdout.decode("ascii")


def _parse_gpu_list(output: str) -> tuple[int, bool]:
    """Parses the output of `nvidia-smi -L`.

    Returns (device_count, is_partitioned). A physical gpu without MIG partitions counts
    as a single device. A physical gpu with MIG partitions counts as one device per `MIG ...`
    line (and not for the enclosing `GPU ...` line), and marks the result as partitioned.
    """
    device_count = 0
    is_partitioned = False
    pending_gpu = False
    pending_gpu_partitioned = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("GPU "):
            if pending_gpu and not pending_gpu_partitioned:
                device_count += 1
            pending_gpu = True
            pending_gpu_partitioned = False
        elif line.startswith("MIG "):
            device_count += 1
            pending_gpu_partitioned = True
            is_partitioned = True
    if pending_gpu and not pending_gpu_partitioned:
        device_count += 1
    return device_count, is_partitioned


def _visible_devices_restriction() -> set[int] | None:
    """Reads the `CUDA_VISIBLE_DEVICES` envvar as currently set in this process.

    Returns None if the envvar is unset (no externally imposed restriction). Returns an
    empty set if the envvar is explicitly set to the empty string (meaning: no gpu at all
    is available, regardless of what was detected). Otherwise returns the parsed set of
    allowed device indices.
    """
    if _CUDA_VISIBLE_DEVICES_ENVVAR not in os.environ:
        return None
    value = os.environ[_CUDA_VISIBLE_DEVICES_ENVVAR]
    if value == "":
        return set()
    return {int(v) for v in value.split(",")}


@dataclass(frozen=True)
class GpuInfo:
    """Describes the gpu capabilities of the current physical machine.

    `_gpu_count` is the number of independently addressable devices as detected (0 if no
    gpu at all) -- NOTE this is internal, callers should always go through `count_at_host`
    and `cuda_visible_at`, which additionally honor any `CUDA_VISIBLE_DEVICES` restriction
    already imposed on this process.
    `unified` is None iff `_gpu_count` is 0, otherwise True if any worker on this machine
    may use the gpu regardless of device indexing (eg macOS, or linux unified memory), and
    False if devices need to be exclusively assigned to workers via eg `CUDA_VISIBLE_DEVICES`.
    """

    _gpu_count: int
    unified: bool | None

    def _allowed_devices(self) -> list[int]:
        """Device indices out of the detected `_gpu_count`, restricted by any pre-existing
        `CUDA_VISIBLE_DEVICES` envvar.
        """
        restriction = _visible_devices_restriction()
        if restriction is None:
            return list(range(self._gpu_count))
        return [i for i in range(self._gpu_count) if i in restriction]

    def count_at_host(self, host_idx: int, workers_per_host: int) -> int:
        """Returns how many workers on the host at `host_idx` (relative to other hosts
        sharing this physical machine, see `cascade.low.core.hostId2localIdx`) should be
        reported as gpu-capable, out of `workers_per_host` total workers on that host.
        """
        allowed = self._allowed_devices()
        if not allowed:
            return 0
        if self.unified:
            return workers_per_host
        offset = host_idx * workers_per_host
        return max(0, min(workers_per_host, len(allowed) - offset))

    def cuda_visible_at(self, host_idx: int, workers_per_host: int, worker_num: int) -> str | None:
        """Returns the value `CUDA_VISIBLE_DEVICES` should be set to for the worker at
        `worker_num` (0-indexed within its host), on the host at `host_idx`. Returns None
        when the envvar should not be set at all -- notably for unified memory, where an
        empty string would incorrectly hide the gpu from the worker entirely.
        """
        allowed = self._allowed_devices()
        if not allowed:
            return ""
        if self.unified:
            return None
        idx = host_idx * workers_per_host + worker_num
        return str(allowed[idx]) if idx < len(allowed) else ""


def get_gpu_info() -> GpuInfo:
    """Detects the gpu capabilities of the current physical machine. Cross-platform, and
    aware of partitioned (MIG) gpus as well as unified memory setups.
    """
    if sys.platform == "darwin":
        # unified memory model on macOS -- no per-device enumeration is meaningful/needed
        return GpuInfo(_gpu_count=1, unified=True)
    try:
        list_output = _list_gpus_output()
    except Exception:
        return GpuInfo(_gpu_count=0, unified=None)
    device_count, is_partitioned = _parse_gpu_list(list_output)
    if device_count == 0:
        return GpuInfo(_gpu_count=0, unified=None)
    if is_partitioned:
        # partitions are not shareable/unified, and querying memory would crash anyway
        return GpuInfo(_gpu_count=device_count, unified=False)
    try:
        memory_output = _query_memory_output()
    except Exception:
        return GpuInfo(_gpu_count=device_count, unified=False)
    unified = any(any(marker in line for marker in _UNIFIED_MEMORY_MARKERS) for line in memory_output.splitlines() if line.strip())
    return GpuInfo(_gpu_count=device_count, unified=unified)
