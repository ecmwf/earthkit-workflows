# controller
EXECUTOR_HOSTS=4

# venvs and libs
export LD_LIBRARY_PATH=/usr/local/apps/ecmwf-toolbox/2024.09.0.0/GNU/8.5/lib/
export FDB5_CONFIG=/home/fdbprod/etc/fdb/config.yaml
source ~/venv/casc/bin/activate

# logging
LOGGING_ROOT=~/logz

# job
JOB='j1.all'
export JOB1_DATA_ROOT="$HPCPERM/gribs/casc_g02/"
export JOB1_END_STEP=60
export JOB1_NUM_ENSEMBLES=10
export JOB1_GRID=O640

# executor
WORKERS_PER_HOST=10
SHM_VOL_GB=64
