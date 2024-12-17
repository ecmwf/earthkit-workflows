#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $1

srun -J cascade-devel-01 --nodes=$((EXECUTOR_HOSTS+1)) --ntasks-per-node=1 --qos=np $SCRIPT_DIR/slurm_entrypoint.sh $1
