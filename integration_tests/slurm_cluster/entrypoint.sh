#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${NODE_ROLE:-}" ]]; then
  echo "NODE_ROLE must be set to either controller or worker"
  exit 1
fi

mkdir -p /run/sshd
install -d -m 700 -o munge -g munge /var/lib/munge /var/log/munge
install -d -m 711 -o munge -g munge /var/run/munge
touch /var/log/munge/munged.log
chown munge:munge /var/log/munge/munged.log
chmod 600 /var/log/munge/munged.log

ssh-keygen -A
/usr/sbin/sshd

if ! pgrep -x munged >/dev/null 2>&1; then
  runuser -u munge -- /usr/sbin/munged --syslog --seed-file /var/run/munge/munged.seed
fi

# Force cgroup plugin name to "disabled".
# On some distro builds this is loaded as a plugin name, so a no-op
# cgroup_disabled plugin is installed in the image.
cat > /etc/slurm/cgroup.conf <<'EOF'
CgroupPlugin=disabled
EOF
mkdir -p /etc/slurm-llnl
ln -sf /etc/slurm/cgroup.conf /etc/slurm-llnl/cgroup.conf

# Generate runtime config in /etc/slurm so cgroup.conf is resolved from the same directory.
export SLURM_CONF=/etc/slurm/slurm-runtime.conf
cat > "${SLURM_CONF}" <<'EOF'
ClusterName=slurm-mini
SlurmctldHost=slurm-main

AuthType=auth/munge
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
SlurmdParameters=config_overrides
SlurmctldPort=6817
SlurmdPort=6818
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none

SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

NodeName=slurm-worker1 CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8 ThreadsPerCore=2 RealMemory=1024 State=UNKNOWN
NodeName=slurm-worker2 CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8 ThreadsPerCore=2 RealMemory=1024 State=UNKNOWN
PartitionName=debug Nodes=slurm-worker1,slurm-worker2 Default=YES MaxTime=INFINITE State=UP
EOF

for _ in $(seq 1 20); do
  if munge -n | unmunge >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if [[ "${NODE_ROLE}" == "controller" ]]; then
  exec /usr/sbin/slurmctld -D -vv -f "${SLURM_CONF}"
fi

if [[ "${NODE_ROLE}" == "worker" ]]; then
  exec /usr/sbin/slurmd -D -vv -f "${SLURM_CONF}"
fi

echo "Unknown NODE_ROLE '${NODE_ROLE}', expected controller or worker"
exit 1
