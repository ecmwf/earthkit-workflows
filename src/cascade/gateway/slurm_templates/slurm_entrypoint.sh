#!/bin/bash

set -euo pipefail

source "$1"

CONTROLLER="$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)"
CONTROLLER_URL="tcp://$CONTROLLER:$CONTROLLER_PORT"

exec python -m cascade.main dist \
    --idx "$SLURM_PROCID" \
    --controller_url "$CONTROLLER_URL" \
    --instance "$INSTANCE" \
    --hosts "$EXECUTOR_HOSTS" \
    --workers_per_host "$WORKERS_PER_HOST" \
    --loggingConfigSer "$LOGGING_CONFIG_SER" \
    --report_address "$REPORT_ADDRESS"
