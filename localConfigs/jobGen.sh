export LD_LIBRARY_PATH="/home/vojta/.local/lib:$LD_LIBRARY_PATH"
GENERATORS_N=8 GENERATORS_K=10 GENERATORS_L=4 python -m cascade.benchmarks local --job generators --workers_per_host 2 --hosts 2 2>&1 | tee /tmp/all.txt
