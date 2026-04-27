#!/bin/bash
# Build SPH pressure CUDA kernels to PTX

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
NVCC=${CUDA_PATH}/bin/nvcc

# Compile to PTX for sm_86 (RTX 3060)
echo "Compiling sph_pressure.cu to PTX..."
$NVCC -ptx -arch=sm_86 \
    -O3 \
    --use_fast_math \
    -o sph_pressure.ptx \
    sph_pressure.cu

echo "Done: sph_pressure.ptx"
ls -la sph_pressure.ptx
