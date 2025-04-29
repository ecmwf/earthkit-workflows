#!/bin/bash
set -euo pipefail

EXECUTOR_HOSTS=4
srun -J cascade-devel-01 --nodes=$((EXECUTOR_HOSTS+1)) --ntasks-per-node=1 --qos=np slurm_entrypoint.sh

# TODO:
# - executor hosts should be param
# - the config part of the entrypoint should be sourced
# - correct the entrypoint path

