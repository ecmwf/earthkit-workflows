# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation -- vanilla (sequential) entrypoint.

Plain Python implementation: no concurrency, no external runtime.
Matrices are generated and SVD-computed in a simple sequential loop.

Environment variables:
  BENCHMARK_N        -- matrix dimension (N x N)
  BENCHMARK_M        -- number of matrices
  BENCHMARK_T        -- seconds to sleep between generations (default: 0)
  BENCHMARK_NPTHREAD -- numpy/BLAS thread cap (optional, applied before numpy import)
"""

import os
import sys
import traceback

_npt = os.environ.get("BENCHMARK_NPTHREAD", "")
if _npt:
    os.environ.setdefault("OMP_NUM_THREADS", _npt)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _npt)
    os.environ.setdefault("MKL_NUM_THREADS", _npt)

from bdg_runtime import matrix_generator, svd_nuclear_norm, total_sum  # noqa: E402


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    t = float(os.environ.get("BENCHMARK_T", "0"))

    norms = [svd_nuclear_norm(mat) for mat in matrix_generator(n, m, t)]
    result = total_sum(norms)

    print(f"SUCCESS: N={n} M={m} T={t} total={result:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
