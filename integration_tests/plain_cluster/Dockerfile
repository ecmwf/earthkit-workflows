FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN mkdir -p /var/run/sshd /root/.ssh /shared \
    && chmod 700 /root/.ssh

COPY ssh/id_ed25519 /root/.ssh/id_ed25519
COPY ssh/id_ed25519.pub /root/.ssh/id_ed25519.pub

RUN cat /root/.ssh/id_ed25519.pub > /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/id_ed25519 /root/.ssh/authorized_keys \
    && chmod 644 /root/.ssh/id_ed25519.pub \
    && printf "Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n" > /root/.ssh/config \
    && chmod 600 /root/.ssh/config \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY start-sshd.sh /usr/local/bin/start-sshd.sh
RUN chmod +x /usr/local/bin/start-sshd.sh

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/start-sshd.sh"]
