# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark: Batch Data Generation -- Cascade entrypoint.

Graph shape:
  generator  ->  svd_0, svd_1, ..., svd_{M-1}  (all marked as ext_outputs)

The generator task yields M matrices using a sequential product dependency.
Each svd_i task computes the SVD nuclear norm of its matrix.  The final sum
is computed in the entrypoint from the collected ext_output values.

Environment variables:
  BENCHMARK_N  -- matrix dimension (N x N)
  BENCHMARK_M  -- number of matrices / generator outputs
  BENCHMARK_T  -- seconds to sleep between generations (default: 0)
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
from cascade.low.core import JobInstanceRich, TaskDefinition, TaskInstance
from cascade.main import run_locally

from bdg_runtime import matrix_generator, svd_nuclear_norm


def main() -> None:
    n = int(os.environ["BENCHMARK_N"])
    m = int(os.environ["BENCHMARK_M"])
    t = float(os.environ.get("BENCHMARK_T", "0"))
    w = int(os.environ.get("BENCHMARK_W", "4"))

    generator_d = TaskDefinition(
        func=TaskDefinition.func_enc(matrix_generator),
        environment=[],
        input_schema={"n": "int", "m": "int", "t": "float"},
        output_schema=[(f"{i}", "numpy.ndarray") for i in range(m)],
    )
    generator_i = TaskInstance(
        definition=generator_d,
        static_input_kw={"n": n, "m": m, "t": t},
        static_input_ps={},
    )

    builder = JobBuilder()
    builder = builder.with_node("generator", generator_i)
    for i in range(m):
        builder = builder.with_node(f"svd_{i}", TaskBuilder.from_callable(svd_nuclear_norm))
        builder = builder.with_edge("generator", f"svd_{i}", "matrix", f"{i}")
        builder = builder.with_output(f"svd_{i}")

    ji = builder.build().get_or_raise()
    job = JobInstanceRich(jobInstance=ji, checkpointSpec=None)

    results = run_locally(job=job, hosts=1, workers=w, loggingConfigSer=LoggingConfig(disable=True).ser_cliparam())
    total = sum(float(v) for v in results.values())

    print(f"SUCCESS: N={n} M={m} T={t} W={w} total_sum={total:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
