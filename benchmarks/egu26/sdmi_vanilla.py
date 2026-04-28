# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Single Data, Multiple Instructions -- vanilla (sequential) entrypoint.

Plain Python implementation: no concurrency, no external runtime.
All M operations are applied in a simple for-loop.

Environment variables:
  BENCHMARK_N        -- matrix dimension (N x N)
  BENCHMARK_M        -- number of child tasks
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

from sdmi_runtime import generate_matrix, get_operation  # noqa: E402


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])

    matrix = generate_matrix(n)
    results = [get_operation(i)(matrix) for i in range(m)]
    total = sum(results)

    print(f"SUCCESS: N={n} M={m} total={total:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
