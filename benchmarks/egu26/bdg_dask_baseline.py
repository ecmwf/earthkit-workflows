# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation -- Dask baseline entrypoint.

Wasteful implementation: source generates all M matrices first (sequential,
with T-second pauses and product dependency), then a second task computes SVD
nuclear norm for each, and a third returns the total.

Environment variables:
  BENCHMARK_N     -- matrix dimension (N x N)
  BENCHMARK_M     -- number of matrices
  BENCHMARK_T     -- seconds to sleep between generations (default: 0)
  BENCHMARK_W     -- number of Dask worker processes (default: 4)
"""

import os
import sys
import traceback

from dask.distributed import Client, LocalCluster

from bdg_runtime import generate_all_matrices, per_matrix_svd_sums, total_sum


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    t = float(os.environ.get("BENCHMARK_T", "0"))
    w = int(os.environ.get("BENCHMARK_W", "4"))

    with LocalCluster(n_workers=w, threads_per_worker=1) as cluster, Client(cluster) as client:
        source = client.submit(generate_all_matrices, n, m, t)
        sums = client.submit(per_matrix_svd_sums, source)
        result = client.submit(total_sum, sums)
        total = result.result()

    print(f"SUCCESS: N={n} M={m} T={t} W={w} total_sum={total:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
