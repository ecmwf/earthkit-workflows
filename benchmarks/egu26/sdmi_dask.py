# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Single Data, Multiple Instructions -- Dask entrypoint.

One source future generates an N*N random matrix.  M child futures each apply
a different mathematical operation (selected by index % num_ops), returning a
float summary.

Environment variables:
  BENCHMARK_N  -- matrix dimension (N x N)
  BENCHMARK_M  -- number of child tasks
  BENCHMARK_W  -- number of Dask worker processes (default: 4)
"""

import os
import sys
import traceback

from dask.distributed import Client, LocalCluster

from sdmi_runtime import OPERATIONS, generate_matrix


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    w = int(os.environ.get("BENCHMARK_W", "4"))

    with LocalCluster(n_workers=w) as cluster, Client(cluster) as client:
        source = client.submit(generate_matrix, n)
        children = [
            client.submit(OPERATIONS[i % len(OPERATIONS)], source) for i in range(m)
        ]
        client.gather(children)

    print(f"SUCCESS: N={n} M={m} W={w}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
