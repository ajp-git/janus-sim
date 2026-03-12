#!/bin/bash
# GPU Limit Detection Script
# Tests progressively larger particle counts to find maximum

OUTPUT_DIR="/app/output/gpu_limit_test"
RESULTS_FILE="/app/output/gpu_limits.txt"

echo "=========================================="
echo "JANUS V9 - GPU LIMIT DETECTION"
echo "=========================================="
echo "Started: $(date)"
echo ""

# Test sequence
PARTICLE_COUNTS=(2000000 4000000 6000000 8000000 10000000 12000000 14000000 16000000 18000000 20000000 24000000 28000000 32000000)

LAST_STABLE=0
MAX_STEPS=100

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Testing particle counts: ${PARTICLE_COUNTS[*]}"
echo ""

for N in "${PARTICLE_COUNTS[@]}"; do
    echo "----------------------------------------"
    echo "Testing N = $N particles..."

    # Create test output directory
    TEST_DIR="${OUTPUT_DIR}/test_${N}"
    mkdir -p "$TEST_DIR"

    # Run short simulation
    timeout 120 cargo run --release --features cuda --bin nbody_overnight -- \
        --n "$N" \
        --eta 1.06 \
        --dt 0.01 \
        --steps $MAX_STEPS \
        --box-size 300.0 \
        --epsilon 0.18 \
        --output "$TEST_DIR" \
        2>&1 | tee "${TEST_DIR}/run.log"

    EXIT_CODE=$?

    # Check for OOM or error
    if [ $EXIT_CODE -eq 0 ]; then
        # Check GPU memory usage
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        GPU_PERCENT=$((GPU_MEM * 100 / GPU_TOTAL))

        echo "  SUCCESS: N=$N completed"
        echo "  GPU Memory: ${GPU_MEM}/${GPU_TOTAL} MiB (${GPU_PERCENT}%)"

        if [ $GPU_PERCENT -lt 95 ]; then
            LAST_STABLE=$N
            echo "  Status: STABLE (< 95% GPU)"
        else
            echo "  Status: MARGINAL (>= 95% GPU)"
            # Still try next, but mark as marginal
            LAST_STABLE=$N
        fi
    else
        echo "  FAILED: N=$N (exit code $EXIT_CODE)"

        # Check for CUDA OOM
        if grep -q "out of memory\|CUDA_ERROR_OUT_OF_MEMORY\|allocation failed" "${TEST_DIR}/run.log" 2>/dev/null; then
            echo "  Cause: CUDA Out of Memory"
        fi

        echo "  Maximum stable: N=$LAST_STABLE"
        break
    fi

    # Clean up intermediate files
    rm -rf "$TEST_DIR"

    echo ""
done

echo "=========================================="
echo "RESULT: Maximum stable particle count = $LAST_STABLE"
echo "=========================================="

# Save result
echo "N_max=$LAST_STABLE" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> "$RESULTS_FILE"

echo ""
echo "Results saved to: $RESULTS_FILE"
