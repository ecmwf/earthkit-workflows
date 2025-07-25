import os
from typing import Any

import jax
import jax.numpy as jp
import jax.random as jr

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance


def get_funcs():
    K = int(os.environ["MATMUL_K"])
    size = (2**K, 2**K)
    E = int(os.environ["MATMUL_E"])

    def source() -> Any:
        k0 = jr.key(0)
        m = jr.uniform(key=k0, shape=size)
        return m

    def powr(m: Any) -> Any:
        print(f"powr device is {m.device}")
        return m**E * jp.percentile(m, 0.7)

    return source, powr


def get_job() -> JobInstance:
    L = int(os.environ["MATMUL_L"])
    # D = os.environ["MATMUL_D"]
    # it would be tempting to with jax.default_device(jax.devices(D)):
    # alas, it doesn't work because we can't inject this at deser time

    source, powr = get_funcs()
    source_node = TaskBuilder.from_callable(source)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        source_node.definition.needs_gpu = True
    # currently no need to set True downstream since scheduler prefers no transfer

    job = JobBuilder().with_node("source", source_node)
    prv = "source"
    for i in range(L):
        cur = f"pow{i}"
        node = TaskBuilder.from_callable(powr)
        job = job.with_node(cur, node).with_edge(prv, cur, 0)
        prv = cur

    job = job.build().get_or_raise()
    job.ext_outputs = list(job.outputs_of(cur))
    return job


def execute_locally():
    L = int(os.environ["MATMUL_L"])

    source, powr = get_funcs()

    device = "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    print(f"device is {device}")
    with jax.default_device(jax.devices(device)[0]):
        m0 = source()
        for _ in range(L):
            m0 = powr(m0)

    from multiprocessing.shared_memory import SharedMemory

    mem = SharedMemory("benchmark_tmp", create=True, size=m0.nbytes)
    mem.buf[:] = m0.tobytes()


if __name__ == "__main__":
    execute_locally()
