# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=triton-claude-code-env:latest
FROM ${BASE_IMAGE}

ARG SANDBOX_USER=claude
ARG SANDBOX_UID=1000
ARG SANDBOX_GID=1000
USER root

COPY template/ /opt/triton-agent-template/
COPY tools/ /opt/triton-agent-tools/

RUN set -eux; \
    command -v getent; \
    command -v groupadd; \
    command -v useradd; \
    getent group "${SANDBOX_GID}" >/dev/null || groupadd -g "${SANDBOX_GID}" "${SANDBOX_USER}"; \
    id -u "${SANDBOX_USER}" >/dev/null 2>&1 \
      || useradd -m -u "${SANDBOX_UID}" -g "${SANDBOX_GID}" -s /bin/bash "${SANDBOX_USER}"; \
    mkdir -p \
      /opt/triton-agent-template/tools \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier; \
    ln -s /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-template/tools/verify_once.sh; \
    ln -s /opt/triton-agent-tools/verifier \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier/scripts; \
    chown -R 0:0 /opt/triton-agent-template /opt/triton-agent-tools; \
    find /opt/triton-agent-template /opt/triton-agent-tools -type d -exec chmod 0555 {} +; \
    find /opt/triton-agent-template /opt/triton-agent-tools -type f -exec chmod 0444 {} +; \
    chmod 0555 \
      /opt/triton-agent-tools/cleanup_task_processes.sh \
      /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-tools/with_npu_lease.py; \
    command -v awk; \
    command -v claude; \
    command -v pkill; \
    command -v ps; \
    command -v python3; \
    command -v timeout; \
    mkdir -p "/home/${SANDBOX_USER}" /workspace; \
    chown -R "${SANDBOX_USER}:${SANDBOX_GID}" "/home/${SANDBOX_USER}" /workspace

ENV HOME=/home/${SANDBOX_USER}
USER ${SANDBOX_USER}
WORKDIR /workspace
