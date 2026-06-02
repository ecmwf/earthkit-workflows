#!/usr/bin/env sh
set -eu

mkdir -p /var/run/sshd
ssh-keygen -A

exec /usr/sbin/sshd -D -e
