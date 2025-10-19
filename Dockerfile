# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        cmake \
        git \
        python3 \
        python3-pip \
        clang-format && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip3 install markdownlint-cli cmake-format
RUN ./scripts/build_all.sh --auto --build-type RelWithDebInfo

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libstdc++6 \
        python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qallow
COPY --from=build /app /opt/qallow
ENV QALLOW_ENABLE_CUDA=auto \
    QALLOW_LOG_DIR=/opt/qallow/data/logs

CMD ["/opt/qallow/build/qallow", "--phase=13", "--ticks=400"]
