# JOB1_DATA_ROOT=/home/vojta/warehouse/ecmwf/cascEx1
# JOB1_END_STEP=60
# JOB1_NUM_ENSEMBLES=2
# JOB1_GRID=O320
# LD_LIBRARY_PATH="/home/vojta/.local/lib:$LD_LIBRARY_PATH"

JOB1_DATA_ROOT=~/warehouse/ecmwf/cascEx1 JOB1_END_STEP=60 JOB1_NUM_ENSEMBLES=2 JOB1_GRID=O1280 python -m cascade.benchmarks local --job j1.all --workers_per_host 2 --hosts 2 2>&1 | tee /tmp/all.txt
