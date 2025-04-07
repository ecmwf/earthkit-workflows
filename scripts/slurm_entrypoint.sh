#!/bin/bash

# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

set -eo pipefail

source $1

# *** job prep ***
echo "starting procid.jobid $SLURM_PROCID.$SLURM_JOBID with param $1 at host $(hostname) with nodes $SLURM_NODELIST"
WORKER_NODES=$((SLURM_NTASKS-1))
CONTROLLER="$(scontrol show hostname $SLURM_NODELIST | head -n 1)"
CONTROLLER_URL="tcp://$CONTROLLER:5555"
echo "selected controller $CONTROLLER at $CONTROLLER_URL, expecting $WORKER_NODES worker nodes"
mkdir -p $LOGGING_ROOT/$SLURM_JOBID
# *** -------- ***

# *** job run ***
if [ "$(hostname | cut -f 1 -d.)" == "$CONTROLLER" ] ; then
        LOGDIR=$LOGGING_ROOT/$SLURM_JOBID/ctrl.txt
        echo "$SLURM_PROCID is *CONTROLLER*, will log into $LOGDIR"
        if [ -n "$REPORT_ADDRESS" ] ; then
            EXTRA_ARGS="--report_address $REPORT_ADDRESS"
            echo "will use $EXTRA_ARGS"
        else
            EXTRA_ARGS=""
            echo "no extra args"
        fi
else
        LOGDIR=$LOGGING_ROOT/$SLURM_JOBID/worker.$SLURM_PROCID.txt
        echo "$SLURM_PROCID is *WORKER*, will log into $LOGDIR"
        export CASCADE_GPU_COUNT=$(nvidia-smi --list-gpus | grep -c GPU)
        echo "WORKER $SLURM_PROCID believes it can use $CASCADE_GPU_COUNT gpus with impunity"
        EXTRA_ARGS=""
fi
if [ -n "$JOB" ] ; then EXTRA_ARGS="$EXTRA_ARGS --job $JOB" ;
elif [ -n "$INSTANCE" ]; then EXTRA_ARGS="$EXTRA_ARGS --instance $INSTANCE" ;
else echo "neither job nor instance"; exit 1 ; fi

# TODO check procid == 0 <-> nodelist head -n 1
python -m cascade.benchmarks dist --idx $SLURM_PROCID $EXTRA_ARGS --controller-url $CONTROLLER_URL --hosts $EXECUTOR_HOSTS --workers-per-host $WORKERS_PER_HOST --shm-vol-gb $SHM_VOL_GB 2>&1 | tee $LOGDIR
# *** ------- ***
