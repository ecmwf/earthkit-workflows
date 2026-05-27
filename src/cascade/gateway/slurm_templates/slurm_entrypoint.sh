#!/bin/bash

set -euo pipefail

DEBUG=1
source "$1"

CONTROLLER="$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)"
CONTROLLER_URL="tcp://$CONTROLLER:$CONTROLLER_PORT"

if [[ "$DEBUG" == "1" ]]; then
    set -x
    logging_config_ser="${LOGGING_CONFIG_SER:-}"
    {
        echo "=== slurm entrypoint debug ==="
        echo "host=$(hostname)"
        echo "pwd=$(pwd)"
        echo "slurm_job_id=${SLURM_JOB_ID:-}"
        echo "slurm_procid=${SLURM_PROCID:-}"
        echo "slurm_nodelist=${SLURM_JOB_NODELIST:-}"
        echo "command -v uv=$(command -v uv || true)"
        echo "uv --version=$(uv --version || true)"
        echo "command -v python=$(command -v python || true)"
        echo "controller_port=${CONTROLLER_PORT:-}"
        echo "controller=${CONTROLLER}"
        echo "controller_url=${CONTROLLER_URL}"
        echo "instance=${INSTANCE:-}"
        echo "report_address=${REPORT_ADDRESS:-}"
        echo "logging_config_ser_len=${#logging_config_ser}"
        echo "ekw_install_spec=${EKW_INSTALL_SPEC:-}"
        echo "cascade_ekw_install_spec=${CASCADE_EKW_INSTALL_SPEC:-}"
    } >&2
fi

exec uv run --with "$CASCADE_EKW_INSTALL_SPEC" python -m cascade.main dist \
    --idx "$SLURM_PROCID" \
    --controller_url "$CONTROLLER_URL" \
    --instance "$INSTANCE" \
    --hosts "$EXECUTOR_HOSTS" \
    --workers_per_host "$WORKERS_PER_HOST" \
    --loggingConfigSer "$LOGGING_CONFIG_SER" \
    --report_address "$REPORT_ADDRESS"
