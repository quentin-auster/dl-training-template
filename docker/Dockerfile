FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

# System deps + Python 3.12 (Ubuntu 22.04 ships 3.10, need deadsnakes for 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# uv (package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# rclone (cloud sync)
RUN curl https://rclone.org/install.sh | bash

WORKDIR /workspace

# Install Python deps — copy lockfiles first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --extra train --frozen

# Project source, configs, scripts
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh scripts/*.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["./scripts/train_gpu.sh"]
