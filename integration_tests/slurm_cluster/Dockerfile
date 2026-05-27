FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --yes --no-install-recommends \
    bash \
    ca-certificates \
    gcc \
    libc6-dev \
    curl \
    munge \
    openssh-client \
    openssh-server \
    slurm-wlm \
    tini \
    util-linux \
    git-all \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN mkdir -p /root/.ssh \
    /run/sshd \
    /var/lib/munge \
    /var/log/munge \
    /var/log/slurm \
    /var/run/munge \
    /var/spool/slurmctld \
    /var/spool/slurmd \
    && chmod 700 /root/.ssh \
    && chown -R munge:munge /var/lib/munge /var/log/munge /var/run/munge \
    && chown -R slurm:slurm /var/log/slurm /var/spool/slurmctld /var/spool/slurmd

COPY config/ssh_cluster_key /root/.ssh/id_ed25519
COPY config/ssh_cluster_key.pub /root/.ssh/id_ed25519.pub
COPY config/ssh_client_config /root/.ssh/config
RUN cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/id_ed25519 /root/.ssh/authorized_keys /root/.ssh/config \
    && chmod 644 /root/.ssh/id_ed25519.pub

COPY config/sshd_cluster.conf /etc/ssh/sshd_config.d/sshd_cluster.conf
COPY config/munge.key /etc/munge/munge.key
COPY config/slurm.conf /etc/slurm/slurm.conf
COPY plugins/cgroup_disabled.c /tmp/cgroup_disabled.c
COPY scripts/hello_world.sh /usr/local/bin/slurm-hello-world
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 400 /etc/munge/munge.key \
    && chown munge:munge /etc/munge/munge.key \
    && mkdir -p /etc/slurm-llnl \
    && ln -sf /etc/slurm/slurm.conf /etc/slurm-llnl/slurm.conf \
    && plugin_dir="$(dirname "$(find /usr -name 'cgroup_v2.so' | head -n 1)")" \
    && test -n "${plugin_dir}" \
    && slurm_ver="$(slurmd -V | awk '{print $2}' | head -n1)" \
    && major="$(echo "${slurm_ver}" | cut -d. -f1)" \
    && minor="$(echo "${slurm_ver}" | cut -d. -f2)" \
    && micro="$(echo "${slurm_ver}" | cut -d. -f3)" \
    && plugin_ver="$(printf '0x%02x%02x%02x' "${major}" "${minor}" "${micro}")" \
    && gcc -fPIC -shared -O2 -DSLURM_PLUGIN_VERSION="${plugin_ver}" /tmp/cgroup_disabled.c -o "${plugin_dir}/disabled.so" \
    && rm -f /tmp/cgroup_disabled.c \
    && chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/slurm-hello-world

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
