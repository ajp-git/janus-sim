# Janus Cosmological Model — Docker Image
# Base: NVIDIA CUDA + Ubuntu 22.04
# GPU: RTX 3060 (sm_86 architecture)

FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    pkg-config \
    libssl-dev \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    # Pour visualisation légère des résultats CSV
    python3-matplotlib \
    python3-numpy \
    # For cuFFT bindings (cufft_rust, cudarc)
    libclang-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# RTX 3060 = Ampere = sm_86
ENV CUDA_COMPUTE_CAP=86
ENV CUDA_ARCH="sm_86"

# Working directory
WORKDIR /app

# Copy project
COPY . .

# Pre-fetch Rust dependencies (layer cache)
RUN cargo fetch

# Build release binaries
RUN cargo build --release

# Output directory
RUN mkdir -p output

# Default: run the Friedmann simulation
CMD ["cargo", "run", "--release", "--bin", "friedmann"]
