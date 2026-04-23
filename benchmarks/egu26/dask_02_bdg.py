# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation (BDG)

The source generator is inherently sequential: it keeps the last yielded
matrix, and each new matrix is the element-wise product of a fresh random
matrix with that previous one.

Two implementations are provided:

  baseline  -- "wasteful": the source generates all M matrices sequentially
                (sleeping T seconds between each), then passes them all at once
                to the summation stage.  No overlap between generation and
                computation.

  actors    -- uses a Dask actor as the generator.  Actor method calls are
                serialized on a single worker, preserving sequentiality, while
                the downstream sum tasks can start eagerly as each matrix
                becomes available.

Graph shape (both variants):
  source  ->  per_matrix_sums  ->  total_sum

Environment variables:
  BENCHMARK_N  -- matrix dimension (N x N)
  BENCHMARK_M  -- number of matrices to generate
  BENCHMARK_T  -- seconds to sleep between matrix generations (float, default 0)
  BENCHMARK_IMPL -- "baseline" (default) or "actors"
"""

import os
import sys
import time
import traceback

import numpy as np
from dask.distributed import Client, LocalCluster


# --------------------------------------------------------------------------- #
# Task functions
# --------------------------------------------------------------------------- #


def generate_all_matrices(n: int, m: int, t: float) -> list[np.ndarray]:
    """Source node: generate M matrices sequentially with T-second pauses.

    Each matrix is the element-wise product of a fresh random matrix and the
    previously generated one, making the sequence inherently serial.
    """
    matrices: list[np.ndarray] = []
    last: np.ndarray | None = None
    for _ in range(m):
        new = np.random.uniform(0.0, 1.0, (n, n))
        if last is not None:
            new = new * last
        last = new
        matrices.append(new)
        if t > 0.0:
            time.sleep(t)
    return matrices


def per_matrix_svd_sums(matrices: list[np.ndarray]) -> list[float]:
    """Second node: compute SVD of each matrix and sum its singular values (nuclear norm)."""
    results = []
    for mat in matrices:
        _, s, _ = np.linalg.svd(mat)
        results.append(float(np.sum(s)))
    return results


def total_sum(values: list[float]) -> float:
    """Third node: sum all per-matrix floats into one scalar."""
    return sum(values)


# --------------------------------------------------------------------------- #
# Baseline implementation
# --------------------------------------------------------------------------- #


def run_baseline(client: Client, n: int, m: int, t: float) -> float:
    source = client.submit(generate_all_matrices, n, m, t)
    sums = client.submit(per_matrix_svd_sums, source)
    result = client.submit(total_sum, sums)
    return result.result()


# --------------------------------------------------------------------------- #
# Actors implementation
# --------------------------------------------------------------------------- #


class MatrixGenerator:
    """Dask actor whose sole job is sequential matrix generation.

    Calling next_matrix() generates a new random matrix, multiplies it by the
    previous one (maintaining the dependency chain), sleeps T seconds, then
    returns the matrix.  Because actor method calls are serialised on one worker,
    the state is always correct regardless of which task calls in.
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
    """Fetch the next matrix from the actor, then compute its SVD nuclear norm.

    The SVD runs on a regular worker, not inside the actor.  Because M of these
    tasks are submitted concurrently, they queue at the actor one-by-one; as
    each task exits the actor it computes SVD on its worker while the next task
    is already sleeping inside the actor -- that is the concurrency gain.
    """
    matrix = generator.next_matrix().result()
    _, s, _ = np.linalg.svd(matrix)
    return float(np.sum(s))


def run_actors(client: Client, n: int, m: int, t: float) -> float:
    generator = client.submit(MatrixGenerator, n, t, actor=True).result()

    # All M tasks are submitted before any run.  They compete to enter the actor
    # serially (preserving the matrix chain), but SVD runs in parallel with the
    # next task's sleep inside the actor.
    futures = [client.submit(generate_and_svd, generator) for _ in range(m)]

    sums: list[float] = client.gather(futures)
    return total_sum(sums)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    t = float(os.environ.get("BENCHMARK_T", "0"))
    impl = os.environ.get("BENCHMARK_IMPL", "baseline")

    with LocalCluster() as cluster, Client(cluster) as client:
        if impl == "baseline":
            result = run_baseline(client, n, m, t)
        elif impl == "actors":
            result = run_actors(client, n, m, t)
        else:
            raise ValueError(f"Unknown BENCHMARK_IMPL={impl!r}; use 'baseline' or 'actors'")

    print(f"SUCCESS: N={n} M={m} T={t} impl={impl} total_sum={result:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
