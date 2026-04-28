# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Single Data, Multiple Instructions -- Cascade entrypoint.

Graph shape mirrors the Dask version: one source task produces an N*N matrix,
M child tasks each consume it and return a float summary.  All child outputs
are marked as external so that the controller waits for them (equivalent to
Dask's gather).

Environment variables:
  BENCHMARK_N  -- matrix dimension (N x N)
  BENCHMARK_M  -- number of child tasks
  BENCHMARK_W  -- number of cascade worker processes (default: 4)
"""

import os
import sys

# Set BLAS/OpenMP thread count before numpy is imported (numpy reads these at init time).
_npt = os.environ.get("BENCHMARK_NPTHREAD", "1")
os.environ.setdefault("OMP_NUM_THREADS", _npt)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _npt)
os.environ.setdefault("MKL_NUM_THREADS", _npt)

import traceback

from cascade.deployment.logging import LoggingConfig
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstanceRich
from cascade.main import run_locally

from sdmi_runtime import get_operation, generate_matrix


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    w = int(os.environ.get("BENCHMARK_W", "4"))

    builder = JobBuilder()
    builder = builder.with_node(
        "source", TaskBuilder.from_callable(generate_matrix).with_values(n=n)
    )
    for i in range(m):
        op = get_operation(i)
        builder = builder.with_node(f"child_{i}", TaskBuilder.from_callable(op))
        builder = builder.with_edge("source", f"child_{i}", "matrix")
        builder = builder.with_output(f"child_{i}")

    ji = builder.build().get_or_raise()
    job = JobInstanceRich(jobInstance=ji, checkpointSpec=None)

    run_locally(job=job, hosts=1, workers=w, loggingConfigSer=LoggingConfig(disable=True).ser_cliparam())

    print(f"SUCCESS: N={n} M={m} W={w}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
