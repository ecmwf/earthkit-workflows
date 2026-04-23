# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Single Data, Multiple Instructions (SDMI)

One source node generates an N*N random matrix. M child nodes each consume that
same matrix and compute a different mathematical operation on it.

Environment variables:
  BENCHMARK_N  -- matrix dimension (matrix is N x N)
  BENCHMARK_M  -- number of child tasks
"""

import os
import sys
import traceback

import numpy as np
from dask.distributed import Client, LocalCluster


# --------------------------------------------------------------------------- #
# Source node
# --------------------------------------------------------------------------- #


def generate_matrix(n: int) -> np.ndarray:
    return np.random.uniform(0.0, 1.0, (n, n))


# --------------------------------------------------------------------------- #
# Child operation functions (fixed ordered list)
# --------------------------------------------------------------------------- #


def op_square(matrix: np.ndarray) -> np.ndarray:
    return matrix**2


def op_sqrt(matrix: np.ndarray) -> np.ndarray:
    return np.sqrt(matrix)


def op_sin(matrix: np.ndarray) -> np.ndarray:
    return np.sin(matrix)


def op_cos(matrix: np.ndarray) -> np.ndarray:
    return np.cos(matrix)


def op_exp(matrix: np.ndarray) -> np.ndarray:
    return np.exp(matrix)


def op_log(matrix: np.ndarray) -> np.ndarray:
    # values are in (0, 1) so log is safe (negative results are fine)
    return np.log(matrix)


def op_trace(matrix: np.ndarray) -> float:
    return float(np.trace(matrix))


def op_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix))


def op_det(matrix: np.ndarray) -> float:
    return float(np.linalg.det(matrix))


def op_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(matrix)


OPERATIONS = [
    op_square,
    op_sqrt,
    op_sin,
    op_cos,
    op_exp,
    op_log,
    op_trace,
    op_norm,
    op_det,
    op_eigenvalues,
]


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])

    with LocalCluster() as cluster, Client(cluster) as client:
        source = client.submit(generate_matrix, n)

        children = [
            client.submit(OPERATIONS[i % len(OPERATIONS)], source) for i in range(m)
        ]

        client.gather(children)

    print(f"SUCCESS: N={n} M={m}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
