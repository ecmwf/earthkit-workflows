#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$1"
mkdir -p "$JOB_ROOT"

exec srun -J cascade-devel-01 \
    --nodes=$((EXECUTOR_HOSTS+1)) \
    --ntasks-per-node=1 \
    --qos=np \
    --kill-on-bad-exit=1 \
    --output="$JOB_ROOT/slurm-%j-%t.out" \
    --error="$JOB_ROOT/slurm-%j-%t.err" \
    "$SCRIPT_DIR/slurm_entrypoint.sh" "$1"
