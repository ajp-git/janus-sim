#!/bin/bash
# Run all 30 LHS exploration simulations sequentially

set -e
cd /mnt/T2/janus-sim

RESULTS_FILE="output/lhs_exploration/results.csv"
mkdir -p output/lhs_exploration

# Initialize results file
echo "run,eta,lambda_base,r_smooth,seg_final,ke_ratio,time_seconds,status" > "$RESULTS_FILE"

for i in $(seq -w 1 30); do
    CONFIG="optim/lhs_exploration/config_run_${i}.yaml"

    echo "========================================"
    echo "Run $i/30 — $(date)"
    echo "========================================"

    # Extract params from config
    ETA=$(grep "eta:" "$CONFIG" | head -1 | awk '{print $2}')
    LAMBDA=$(grep "lambda_base_mpc:" "$CONFIG" | awk '{print $2}')
    RSMOOTH=$(grep "r_smooth_mpc:" "$CONFIG" | awk '{print $2}')

    echo "eta=$ETA, lambda=$LAMBDA, R_smooth=$RSMOOTH"

    START_TIME=$(date +%s)

    # Run simulation
    if timeout 3600 docker compose run --rm dev \
        cargo run --release --features "cuda cufft" --bin janus_optim -- \
        --config "$CONFIG" 2>&1 | tee "output/lhs_exploration/run_${i}.log"; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Extract results from summary.json
        SUMMARY="output/lhs_exploration/lhs_run_${i}/summary.json"
        if [ -f "$SUMMARY" ]; then
            SEG=$(grep seg_final "$SUMMARY" | grep -oE '[0-9]+\.[0-9]+')
            KE=$(grep ke_ratio_final "$SUMMARY" | grep -oE '[0-9]+\.[0-9]+')
            echo "$i,$ETA,$LAMBDA,$RSMOOTH,$SEG,$KE,$DURATION,OK" >> "$RESULTS_FILE"
            echo "-> S=$SEG, KE=$KE (${DURATION}s)"
        else
            echo "$i,$ETA,$LAMBDA,$RSMOOTH,NA,NA,$DURATION,NO_SUMMARY" >> "$RESULTS_FILE"
            echo "-> NO SUMMARY FILE"
        fi
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "$i,$ETA,$LAMBDA,$RSMOOTH,NA,NA,$DURATION,FAILED" >> "$RESULTS_FILE"
        echo "-> FAILED"
    fi

    echo ""
done

echo "========================================"
echo "LHS EXPLORATION COMPLETE"
echo "Results: $RESULTS_FILE"
echo "========================================"
