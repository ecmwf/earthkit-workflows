# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation -- Dask actors entrypoint.

A MatrixGenerator Dask actor maintains the sequential dependency chain (each
matrix is the product of a fresh random matrix and the previous one).  M
tasks are submitted concurrently; they queue at the actor one-by-one, and
after each task exits the actor it computes SVD on a regular worker -- this
overlaps the next task's sleep inside the actor.

Environment variables:
  BENCHMARK_N     -- matrix dimension (N x N)
  BENCHMARK_M     -- number of matrices
  BENCHMARK_T     -- seconds to sleep between generations (default: 0)
  BENCHMARK_W     -- number of Dask worker processes (default: 4)
"""

import os
import sys
import time
import traceback

import numpy as np
from dask.distributed import Client, LocalCluster

from bdg_runtime import svd_nuclear_norm, total_sum



class MatrixGenerator:
    """Dask actor: sequential matrix generator with product dependency.

    Actor method calls are serialised on a single worker.  All M calls can be
    submitted at once; as each task exits the actor it computes SVD on another
    worker while the next task is sleeping inside the actor.
    """

    def __init__(self, n: int, t: float) -> None:
        self._n = n
        self._t = t
        self._last: np.ndarray | None = None

    def next_matrix(self) -> np.ndarray:
        new = np.random.uniform(0.0, 1.0, (self._n, self._n))
        if self._last is not None:
            new = new * self._last
        self._last = new
        if self._t > 0.0:
            time.sleep(self._t)
        return new


def generate_and_svd(generator: MatrixGenerator) -> float:
    """Fetch next matrix from the actor, then compute its SVD nuclear norm.

    SVD runs on a regular worker (not inside the actor), enabling overlap
    between the actor's sleep and downstream computation.
    """
    matrix = generator.next_matrix().result()
    return svd_nuclear_norm(matrix)


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    t = float(os.environ.get("BENCHMARK_T", "0"))
    w = int(os.environ.get("BENCHMARK_W", "4"))
    npt = int(os.environ.get("BENCHMARK_NPTHREAD", "1"))

    with LocalCluster(n_workers=w, threads_per_worker=npt) as cluster, Client(cluster) as client:
        generator = client.submit(MatrixGenerator, n, t, actor=True).result()
        # pure=False prevents Dask from deduplicating tasks that share the same
        # function + arguments -- without it, all M submissions collapse into one.
        futures = [client.submit(generate_and_svd, generator, pure=False) for _ in range(m)]
        sums: list[float] = client.gather(futures)
        result = total_sum(sums)

    print(f"SUCCESS: N={n} M={m} T={t} W={w} total_sum={result:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
