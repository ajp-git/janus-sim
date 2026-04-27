#!/bin/bash
# Build Cooling CUDA kernels to PTX

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
NVCC=${CUDA_PATH}/bin/nvcc

# Compile to PTX for sm_86 (RTX 3060)
echo "Compiling cooling_kernel.cu to PTX..."
$NVCC -ptx -arch=sm_86 \
    -O3 \
    --use_fast_math \
    -o cooling_kernel.ptx \
    cooling_kernel.cu

echo "Done: cooling_kernel.ptx"
ls -la cooling_kernel.ptx
