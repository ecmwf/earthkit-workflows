#!/usr/bin/env bash

set -euo pipefail

echo "Waiting for Slurm workers to become ready..."
for _ in $(seq 1 60); do
  ready_nodes="$(sinfo -h -N -o '%N %t' 2>/dev/null | awk '$2 ~ /idle|mix/ {count++} END {print count+0}')"
  if [[ "${ready_nodes}" -ge 2 ]]; then
    break
  fi
  sleep 2
done

job_script="/shared/slurm-hello.sbatch"
cat > "${job_script}" <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=slurm-hello
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --output=/shared/slurm-hello-%j.out

echo "Hello from Slurm job ${SLURM_JOB_ID}"
echo "Allocated nodes: ${SLURM_JOB_NODELIST}"
srun hostname
EOF

job_id="$(sbatch --parsable "${job_script}")"
echo "Submitted job ${job_id}"

while squeue -h -j "${job_id}" | grep -q .; do
  sleep 1
done

output_file="/shared/slurm-hello-${job_id}.out"
if [[ ! -f "${output_file}" ]]; then
  echo "Job output file was not created: ${output_file}"
  exit 1
fi

echo "=== Slurm hello world output ==="
cat "${output_file}"
