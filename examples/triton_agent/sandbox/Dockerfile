# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=triton-claude-code-env:latest
FROM ${BASE_IMAGE}

ARG SANDBOX_USER=claude
USER root

COPY template/ /opt/triton-agent-template/
COPY tools/ /opt/triton-agent-tools/

RUN set -eux; \
    id -u "${SANDBOX_USER}" >/dev/null; \
    mkdir -p \
      /opt/triton-agent-template/tools \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier \
      /workspace; \
    ln -s /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-template/tools/verify_once.sh; \
    ln -s /opt/triton-agent-tools/verifier \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier/scripts; \
    chmod -R a-w /opt/triton-agent-template /opt/triton-agent-tools; \
    chmod 0555 \
      /opt/triton-agent-tools/cleanup_task_processes.sh \
      /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-tools/with_npu_lease.py; \
    chown "${SANDBOX_USER}:" /workspace

ENV HOME=/home/${SANDBOX_USER}
USER ${SANDBOX_USER}
WORKDIR /workspace
