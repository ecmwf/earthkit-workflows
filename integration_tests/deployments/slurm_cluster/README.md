# Minimal Slurm cluster (1 controller, 2 workers)

This folder contains a greenfield Docker Compose setup for:

- `slurm-main` (controller)
- `slurm-worker1` (worker)
- `slurm-worker2` (worker)

All nodes include:

- Slurm + Munge
- OpenSSH server/client on the same Docker network
- `uv` installed at `/usr/local/bin/uv`

## Start the cluster

```bash
cd integration_tests/deployments/slurm_cluster
./start.sh
```

## Run the Slurm hello-world job

```bash
docker compose exec slurm-main /usr/local/bin/slurm-hello-world
```

## Quick SSH check between nodes

```bash
docker compose exec slurm-main ssh slurm-worker1 hostname
docker compose exec slurm-main ssh slurm-worker2 hostname
```
