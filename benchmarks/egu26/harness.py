# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark harness for EGU26: runs each benchmark as a cold subprocess and reports timing.

Usage:
    python harness.py <scenario> <scale>

Arguments:
    scenario  -- bdg | sdmi | all
    scale     -- 1w | 2w | 4w | all

The harness always sets BENCHMARK_NPTHREAD=1.  Cascade entrypoints translate
this into OMP/OPENBLAS/MKL_NUM_THREADS; Dask entrypoints use it as
threads_per_worker.

BENCHMARK_N, BENCHMARK_M, and BENCHMARK_T are inherited from the caller's
environment; defaults are applied if unset:
    BENCHMARK_N = 100
    BENCHMARK_M = 10
    BENCHMARK_T = 0
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent

SCENARIOS: dict[str, list[str]] = {
    "sdmi": [
        "sdmi_dask.py",
        "sdmi_cascade.py",
    ],
    "bdg": [
        "bdg_dask_baseline.py",
        "bdg_dask_actors.py",
        "bdg_cascade.py",
    ],
}

SCALES: dict[str, int] = {
    "1w": 1,
    "2w": 2,
    "4w": 4,
}

# Per-scenario defaults. Callers can override any of these via env vars.
# sdmi: many small independent ops on one matrix -- larger N, more children.
# bdg: sequential generation with expensive SVD and sleep -- moderate N and M,
#      non-zero T so the actor vs baseline difference is visible.
DEFAULTS: dict[str, dict[str, str]] = {
    "sdmi": {"BENCHMARK_N": "200", "BENCHMARK_M": "20", "BENCHMARK_T": "0"},
    "bdg":  {"BENCHMARK_N": "200", "BENCHMARK_M": "10", "BENCHMARK_T": "0.5"},
}


def run_one(script: str, workers: int, env: dict[str, str]) -> tuple[float, bool, str]:
    run_env = {**env, "BENCHMARK_W": str(workers)}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(BENCH_DIR / script)],
        env=run_env,
        capture_output=True,
        text=True,
        cwd=str(BENCH_DIR),
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    # grab the SUCCESS/FAIL line from stdout for context
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not ok:
        last_line = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    return elapsed, ok, last_line


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <bdg|sdmi|all> <1w|2w|4w|all>")
        sys.exit(1)

    scenario_arg, scale_arg = sys.argv[1], sys.argv[2]

    valid_scenarios = list(SCENARIOS) + ["all"]
    valid_scales = list(SCALES) + ["all"]
    if scenario_arg not in valid_scenarios:
        print(f"Unknown scenario {scenario_arg!r}. Choose from: {valid_scenarios}")
        sys.exit(1)
    if scale_arg not in valid_scales:
        print(f"Unknown scale {scale_arg!r}. Choose from: {valid_scales}")
        sys.exit(1)

    scenarios = list(SCENARIOS) if scenario_arg == "all" else [scenario_arg]
    scales = list(SCALES) if scale_arg == "all" else [scale_arg]

    rows: list[tuple[str, str, int, float, str]] = []

    for scenario in scenarios:
        scenario_env: dict[str, str] = {
            **os.environ,
            "BENCHMARK_NPTHREAD": "1",
        }
        for key, val in DEFAULTS[scenario].items():
            scenario_env.setdefault(key, val)

        for scale in scales:
            workers = SCALES[scale]
            for script in SCENARIOS[scenario]:
                print(f"  running {script} w={workers} ...", flush=True)
                elapsed, ok, detail = run_one(script, workers, scenario_env)
                status = "OK" if ok else "FAIL"
                rows.append((script, scale, workers, elapsed, status, detail))
                print(f"    -> {status} {elapsed:.2f}s", flush=True)

    # print results table
    name_w = max(len(r[0]) for r in rows)
    print()
    print(f"{'script':<{name_w}}  {'scale':<5}  {'workers':<7}  {'time(s)':<9}  result")
    print("-" * (name_w + 5 + 7 + 9 + 10))
    for script, scale, workers, elapsed, status, detail in rows:
        print(f"{script:<{name_w}}  {scale:<5}  {workers:<7}  {elapsed:<9.2f}  {status}")


if __name__ == "__main__":
    main()
