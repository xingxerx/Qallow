# Devcontainer image for Qallow development on Ubuntu
FROM mcr.microsoft.com/devcontainers/base:ubuntu

ARG DEBIAN_FRONTEND=noninteractive

# Install build essentials and Python tooling.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        cmake \
        git \
        curl \
        wget \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        pkg-config \
        libssl-dev \
        libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Allow pip to install even in externally managed environments (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Create and activate a virtual environment for all Python tooling
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip

# Ensure venv is the default Python/pip on PATH
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspaces/Qallow

# Pre-install Python dependencies into the venv so the container is ready immediately.
# Build context is now parent directory, so requirements.txt is accessible
COPY requirements.txt ./
RUN pip install -r requirements.txt || true
