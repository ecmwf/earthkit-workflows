# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
