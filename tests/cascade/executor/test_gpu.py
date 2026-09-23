# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for gpu capability detection -- mocks the private nvidia-smi invocations so
that we can exercise all platform/partitioning/unified-memory combinations without any
actual gpu hardware.
"""

import pytest

import cascade.executor.platform.gpu as gpu

NO_PARTITION_OUTPUT = "GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU (UUID: GPU-12345679-abcd-4905-a985-2bdd9950ad63)\n"

ONE_PARTITION_OUTPUT = (
    "GPU 0: NVIDIA H200 NVL (UUID: GPU-12345679-abcd-4905-a985-2bdd9950ad63)\n"
    "  MIG 3g.71gb     Device  0: (UUID: MIG-12345679-abcd-4905-a985-2bdd9950ad63)\n"
)

TWO_PARTITIONS_OUTPUT = (
    "GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-12345679-abcd-4905-a985-2bdd9950ad63)\n"
    "  MIG 2g.48gb     Device  0: (UUID: MIG-12345679-abcd-4905-a985-2bdd9950ad63)\n"
    "  MIG 2g.48gb     Device  1: (UUID: MIG-12345679-abcd-4905-a985-2bdd9950ad63)\n"
)

TWO_GPUS_OUTPUT = (
    "GPU 0: NVIDIA A100 (UUID: GPU-12345679-abcd-4905-a985-2bdd9950ad63)\n"
    "GPU 1: NVIDIA A100 (UUID: GPU-abcdef01-abcd-4905-a985-2bdd9950ad63)\n"
)


@pytest.fixture(autouse=True)
def not_darwin(monkeypatch):
    monkeypatch.setattr(gpu.sys, "platform", "linux")


@pytest.fixture(autouse=True)
def no_visible_devices_restriction(monkeypatch):
    """The global session fixture forces CUDA_VISIBLE_DEVICES="" so that other tests are
    stable regardless of the host's actual gpu setup. Here we test `count_at_host`/
    `cuda_visible_at` themselves, so by default we want no restriction imposed at all;
    individual tests override this via monkeypatch as needed.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_no_gpu(monkeypatch):
    def raiser():
        raise Exception("no nvidia-smi binary")

    monkeypatch.setattr(gpu, "_list_gpus_output", raiser)
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=0, unified=None)


def test_regular_non_unified(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: NO_PARTITION_OUTPUT)
    monkeypatch.setattr(gpu, "_query_memory_output", lambda: "8192 MiB\n")
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=1, unified=False)


def test_regular_unified_memory(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: NO_PARTITION_OUTPUT)
    monkeypatch.setattr(gpu, "_query_memory_output", lambda: "N/A\n")
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=1, unified=True)


def test_regular_unified_memory_not_supported(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: NO_PARTITION_OUTPUT)
    monkeypatch.setattr(gpu, "_query_memory_output", lambda: "Not Supported\n")
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=1, unified=True)


def test_two_gpus(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: TWO_GPUS_OUTPUT)
    monkeypatch.setattr(gpu, "_query_memory_output", lambda: "8192 MiB\n8192 MiB\n")
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=2, unified=False)


def test_one_partition(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: ONE_PARTITION_OUTPUT)

    def query_memory_should_not_be_called():
        raise AssertionError("must not query memory on a partitioned gpu")

    monkeypatch.setattr(gpu, "_query_memory_output", query_memory_should_not_be_called)
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=1, unified=False)


def test_two_partitions(monkeypatch):
    monkeypatch.setattr(gpu, "_list_gpus_output", lambda: TWO_PARTITIONS_OUTPUT)

    def query_memory_should_not_be_called():
        raise AssertionError("must not query memory on a partitioned gpu")

    monkeypatch.setattr(gpu, "_query_memory_output", query_memory_should_not_be_called)
    info = gpu.get_gpu_info()
    assert info == gpu.GpuInfo(_gpu_count=2, unified=False)


def test_darwin(monkeypatch):
    monkeypatch.setattr(gpu.sys, "platform", "darwin")

    def should_not_be_called():
        raise AssertionError("must not invoke nvidia-smi on darwin")

    monkeypatch.setattr(gpu, "_list_gpus_output", should_not_be_called)
    info = gpu.get_gpu_info()
    assert info._gpu_count > 0
    assert info.unified is True


def test_count_at_host_non_unified_offsets_across_local_hosts():
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.count_at_host(0, 3) == 3
    assert info.count_at_host(1, 3) == 1
    assert info.count_at_host(2, 3) == 0


def test_count_at_host_unified_ignores_device_count():
    info = gpu.GpuInfo(_gpu_count=1, unified=True)
    assert info.count_at_host(0, 5) == 5
    assert info.count_at_host(3, 5) == 5


def test_count_at_host_no_gpu():
    info = gpu.GpuInfo(_gpu_count=0, unified=None)
    assert info.count_at_host(0, 5) == 0


def test_cuda_visible_at_non_unified():
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.cuda_visible_at(0, 3, 0) == "0"
    assert info.cuda_visible_at(0, 3, 2) == "2"
    assert info.cuda_visible_at(1, 3, 0) == "3"
    assert info.cuda_visible_at(1, 3, 1) == ""


def test_cuda_visible_at_unified_returns_none():
    info = gpu.GpuInfo(_gpu_count=1, unified=True)
    assert info.cuda_visible_at(0, 5, 4) is None


def test_cuda_visible_at_no_gpu():
    info = gpu.GpuInfo(_gpu_count=0, unified=None)
    assert info.cuda_visible_at(0, 5, 0) == ""


def test_count_at_host_respects_preexisting_visible_devices_restriction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.count_at_host(0, 5) == 2
    assert info.count_at_host(1, 5) == 0


def test_count_at_host_empty_visible_devices_means_no_gpu_even_when_detected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.count_at_host(0, 5) == 0


def test_count_at_host_unified_empty_visible_devices_means_no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    info = gpu.GpuInfo(_gpu_count=1, unified=True)
    assert info.count_at_host(0, 5) == 0


def test_cuda_visible_at_respects_preexisting_visible_devices_restriction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.cuda_visible_at(0, 3, 0) == "1"
    assert info.cuda_visible_at(0, 3, 1) == "3"
    assert info.cuda_visible_at(0, 3, 2) == ""


def test_cuda_visible_at_empty_visible_devices_means_no_gpu_even_when_detected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    info = gpu.GpuInfo(_gpu_count=4, unified=False)
    assert info.cuda_visible_at(0, 3, 0) == ""


def test_cuda_visible_at_unified_empty_visible_devices_means_no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    info = gpu.GpuInfo(_gpu_count=1, unified=True)
    assert info.cuda_visible_at(0, 5, 0) == ""


def test_cuda_visible_at_unified_unrestricted_returns_none(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    info = gpu.GpuInfo(_gpu_count=1, unified=True)
    assert info.cuda_visible_at(0, 5, 0) is None
