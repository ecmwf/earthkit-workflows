#!/bin/bash

set -eo pipefail

# *** CONFIG ***
export LD_LIBRARY_PATH=/usr/local/apps/ecmwf-toolbox/2024.09.0.0/GNU/8.5/lib/
export FDB5_CONFIG=/home/fdbprod/etc/fdb/config.yaml
export PYTHONPATH=/home/ecm6012/src/mir-python
LOGGING_ROOT=/home/ecm6012/logz
JOB='j1.all'
WORKERS_PER_HOST=10
source ~/venv/casc/bin/activate
# TODO expose also:
# data prefix
# shm capacity
# TODO make this externally sourcable config
# *** ------ ***

# *** job prep *** 
echo "starting procid.jobid $SLURM_PROCID.$SLURM_JOBID with param $1 at host $(hostname) with nodes $SLURM_NODELIST"
WORKER_NODES=$((SLURM_NTASKS-1))
CONTROLLER="$(echo $SLURM_NODELIST | sed 's/\(ac.-\)\[\?\([0-9]*\).*/\1\2/')"
CONTROLLER_URL="tcp://$CONTROLLER:5555"
echo "selected controller $CONTROLLER at $CONTROLLER_URL, expecting $WORKER_NODES"
mkdir -p $LOGGING_ROOT/$SLURM_JOBID
# *** -------- *** 

# *** job run ***
if [ "$(hostname | cut -f 1 -d.)" == "$CONTROLLER" ] ; then
	echo "$SLURM_PROCID is *CONTROLLER*"
	python -m cascade.benchmarks --job $JOB --executor zmq --hosts $WORKER_NODES --controller-url $CONTROLLER_URL --workers 1 --dist controller 2>&1 | tee $LOGGING_ROOT/$SLURM_JOBID/ctrl.txt
else
	echo "$SLURM_PROCID is *WORKER*"
	sleep 1 # TODO smarter... wait to establish comms
	python -m cascade.benchmarks --job $JOB --executor zmq --hosts 1 --host_id $SLURM_PROCID --controller-url $CONTROLLER_URL --workers $WORKERS_PER_HOST --dist worker 2>&1 | tee $LOGGING_ROOT/$SLURM_JOBID/worker.$SLURM_PROCID.txt
fi
# *** ------- ***

# TODO remove:
# - hosts in worker & workers in controller... or rather use a new module as the entrypoint, with a cleaner interface
