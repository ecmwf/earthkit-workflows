# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared runtime functions for the SDMI (Single Data, Multiple Instructions) benchmark.

Imported by both sdmi_dask.py and sdmi_cascade.py.
All operations accept an N*N matrix and return a float summary so that outputs
are trivially serialisable across both runtimes.
"""

import numpy as np


def generate_matrix(n: int) -> np.ndarray:
    return np.random.uniform(0.0, 1.0, (n, n))


def op_square(matrix: np.ndarray) -> float:
    return float(np.sum(matrix**2))


def op_sqrt(matrix: np.ndarray) -> float:
    return float(np.sum(np.sqrt(matrix)))


def op_sin(matrix: np.ndarray) -> float:
    return float(np.sum(np.sin(matrix)))


def op_cos(matrix: np.ndarray) -> float:
    return float(np.sum(np.cos(matrix)))


def op_exp(matrix: np.ndarray) -> float:
    return float(np.sum(np.exp(matrix)))


def op_log(matrix: np.ndarray) -> float:
    return float(np.sum(np.log(matrix)))


def op_trace(matrix: np.ndarray) -> float:
    return float(np.trace(matrix))


def op_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix))


def op_det(matrix: np.ndarray) -> float:
    return float(np.linalg.det(matrix))


def op_eigenvalues(matrix: np.ndarray) -> float:
    return float(np.sum(np.abs(np.linalg.eigvals(matrix))))


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


def multiply_and_svd(i: int, matrix: np.ndarray) -> float:
    """Scale matrix by i then return SVD nuclear norm.  O(n^3), result unique per i."""
    _, s, _ = np.linalg.svd(matrix * (i + 1))
    return float(np.sum(s))


def get_operation(i: int):  # ty:ignore[missing-return-type]
    """Return the child operation for index i.

    Each child multiplies the shared matrix by its own index and then computes
    the SVD nuclear norm -- making the work O(n^3) and numerically distinct per
    child while remaining deterministic.

    The original OPERATIONS list is kept for reference / alternative use.
    """
    return lambda matrix: multiply_and_svd(i, matrix)
