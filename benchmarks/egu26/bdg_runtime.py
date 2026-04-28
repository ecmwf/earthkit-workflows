# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared runtime functions for the BDG (Batch Data Generation) benchmark.

Imported by bdg_dask_baseline.py, bdg_dask_actors.py, and bdg_cascade.py.
"""

import time
from typing import Iterator

import numpy as np


# --------------------------------------------------------------------------- #
# Shared leaf computation
# --------------------------------------------------------------------------- #


def svd_nuclear_norm(matrix: np.ndarray) -> float:
    """Compute the nuclear norm (sum of singular values) of a matrix."""
    _, s, _ = np.linalg.svd(matrix)
    return float(np.sum(s))


def total_sum(values: list[float]) -> float:
    return sum(values)


# --------------------------------------------------------------------------- #
# Baseline (wasteful) source and intermediate nodes
# --------------------------------------------------------------------------- #


def generate_all_matrices(n: int, m: int, t: float) -> list[np.ndarray]:
    """Generate M matrices sequentially with T-second pauses.

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
    return [svd_nuclear_norm(mat) for mat in matrices]


# --------------------------------------------------------------------------- #
# Cascade source: generator that yields matrices one at a time
# --------------------------------------------------------------------------- #


def matrix_generator(n: int, m: int, t: float) -> Iterator[np.ndarray]:
    """Sequential generator: each matrix is the product of a new random matrix
    and the previous one.  Sleeps T seconds between yields.
    """
    last: np.ndarray | None = None
    for _ in range(m):
        new = np.random.uniform(0.0, 1.0, (n, n))
        if last is not None:
            new = new * last
        last = new
        if t > 0.0:
            time.sleep(t)
        yield new
