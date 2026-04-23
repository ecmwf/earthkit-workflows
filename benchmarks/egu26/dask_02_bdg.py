# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation (BDG)

Two implementations are provided:

  baseline  -- "wasteful": the source generates all M matrices sequentially
                (sleeping T seconds between each), then passes them all at once
                to the summation stage.  No overlap between generation and
                computation.

  actors    -- uses a Dask actor as an accumulator so that each matrix is
                summed eagerly as it is produced, overlapping generation and
                computation.

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
from dask.distributed import Client, LocalCluster, get_client


# --------------------------------------------------------------------------- #
# Task functions
# --------------------------------------------------------------------------- #


def generate_all_matrices(n: int, m: int, t: float) -> list[np.ndarray]:
    """Source node: yield M random N*N matrices with T-second pauses between."""
    matrices = []
    for _ in range(m):
        matrices.append(np.random.uniform(0.0, 1.0, (n, n)))
        if t > 0.0:
            time.sleep(t)
    return matrices


def per_matrix_sums(matrices: list[np.ndarray]) -> list[float]:
    """Second node: compute the element-wise sum of each matrix."""
    return [float(np.sum(mat)) for mat in matrices]


def total_sum(values: list[float]) -> float:
    """Third node: sum all per-matrix floats into one scalar."""
    return sum(values)


# --------------------------------------------------------------------------- #
# Baseline implementation
# --------------------------------------------------------------------------- #


def run_baseline(client: Client, n: int, m: int, t: float) -> float:
    source = client.submit(generate_all_matrices, n, m, t)
    sums = client.submit(per_matrix_sums, source)
    result = client.submit(total_sum, sums)
    return result.result()


# --------------------------------------------------------------------------- #
# Actors implementation
# --------------------------------------------------------------------------- #


class SumAccumulator:
    """Dask actor that eagerly accumulates per-matrix sums."""

    def __init__(self) -> None:
        self._values: list[float] = []

    def add(self, value: float) -> None:
        self._values.append(value)

    def total(self) -> float:
        return sum(self._values)


def generate_and_submit_matrix(n: int, t: float, accumulator: SumAccumulator) -> None:
    """Generate one matrix, compute its sum, and push to the accumulator actor."""
    matrix = np.random.uniform(0.0, 1.0, (n, n))
    if t > 0.0:
        time.sleep(t)
    value = float(np.sum(matrix))
    # Actor method calls return futures; fire-and-forget here is intentional
    accumulator.add(value).result()


def run_actors(client: Client, n: int, m: int, t: float) -> float:
    accumulator = client.submit(SumAccumulator, actor=True).result()

    futures = [
        client.submit(generate_and_submit_matrix, n, t, accumulator) for _ in range(m)
    ]
    client.gather(futures)

    return accumulator.total().result()


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
