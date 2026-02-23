#!/bin/bash
# Auto-continue script for Run B
# Waits for current run to finish, then resumes for 10000 more steps
# GRID SIZE: 512³ (must match original run)

OUTPUT_DIR="janus-pm/output/pm5_2026-02-22_211736"
RESUME_STEPS=10000
GRID_SIZE=512  # CRITICAL: Must match original run

echo "═══════════════════════════════════════════════════════════════"
echo "  Auto-Continue Monitor for Run B"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Grid size: ${GRID_SIZE}³"
echo "  Additional steps: ${RESUME_STEPS}"
echo "  Source: ${OUTPUT_DIR}"
echo ""
echo "Monitoring Run B completion..."

# Wait for container to finish
while docker ps | grep -q pm5_runB; do
    sleep 60
    # Show latest progress
    docker logs pm5_runB 2>&1 | tail -3
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Run B completed. Launching continuation..."
echo "══════════════════════════════════════════════════════════════"

FINAL_CKPT="${OUTPUT_DIR}/snapshot_final.bin"
if [ ! -f "${FINAL_CKPT}" ]; then
    echo "ERROR: Final checkpoint not found: ${FINAL_CKPT}"
    echo "Trying checkpoint_peak.bin instead..."
    FINAL_CKPT="${OUTPUT_DIR}/checkpoint_peak.bin"
    if [ ! -f "${FINAL_CKPT}" ]; then
        echo "ERROR: No checkpoint found!"
        exit 1
    fi
fi

echo "Checkpoint: ${FINAL_CKPT}"
ls -lh "${FINAL_CKPT}"

CONTINUE_DIR="${OUTPUT_DIR}_continue"
echo ""
echo "Resume parameters:"
echo "  Checkpoint: ${FINAL_CKPT}"
echo "  Grid: ${GRID_SIZE}³"
echo "  Steps: ${RESUME_STEPS}"
echo "  Output: ${CONTINUE_DIR}"
echo ""

# Launch resume
docker compose run --rm -d --name pm5_runB_continue dev bash -c "
    cd /app &&
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$(find /app/target -name 'libpm_kernels.so' -exec dirname {} \; | head -1) \
    ./target/release/pm5_resume \
        ${FINAL_CKPT} \
        ${RESUME_STEPS} \
        ${GRID_SIZE} \
        ${CONTINUE_DIR} \
        2>&1 | tee ${CONTINUE_DIR}.log
"

echo ""
echo "Resume launched: pm5_runB_continue"
echo "Log: ${CONTINUE_DIR}.log"
echo "══════════════════════════════════════════════════════════════"
