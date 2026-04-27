#!/bin/bash
# Run all 10 grid exploration simulations sequentially

set -e
cd /mnt/T2/janus-sim

RESULTS_FILE="output/grid_10/results.csv"
mkdir -p output/grid_10

# Initialize results file
echo "run,eta,r_smooth,lambda_base,seg_final,seg_peak,z_peak,ke_ratio,time_seconds,status" > "$RESULTS_FILE"

for i in $(seq -w 1 10); do
    CONFIG="optim/grid_10/config_run_${i}.yaml"

    echo "========================================"
    echo "Run $i/10 — $(date)"
    echo "========================================"

    # Extract params from config
    ETA=$(grep "eta:" "$CONFIG" | head -1 | awk '{print $2}')
    RSMOOTH=$(grep "r_smooth_mpc:" "$CONFIG" | awk '{print $2}')
    LAMBDA=$(grep "lambda_base_mpc:" "$CONFIG" | awk '{print $2}')

    echo "eta=$ETA, R_smooth=$RSMOOTH, lambda=$LAMBDA"

    START_TIME=$(date +%s)

    # Run simulation
    if timeout 900 docker compose run --rm dev \
        cargo run --release --features "cuda cufft" --bin janus_optim -- \
        --config "$CONFIG" 2>&1 | tee "output/grid_10/run_${i}.log"; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Extract results from summary.json
        SUMMARY="output/grid_10/run_${i}/summary.json"
        if [ -f "$SUMMARY" ]; then
            SEG=$(grep seg_final "$SUMMARY" | grep -oE '[0-9]+\.[0-9]+')
            KE=$(grep ke_ratio_final "$SUMMARY" | grep -oE '[0-9]+\.[0-9]+')

            # Find peak segregation from time_series
            TS="output/grid_10/run_${i}/time_series.csv"
            if [ -f "$TS" ]; then
                PEAK_LINE=$(sort -t, -k6 -rn "$TS" | head -1)
                SEG_PEAK=$(echo "$PEAK_LINE" | cut -d, -f6)
                Z_PEAK=$(echo "$PEAK_LINE" | cut -d, -f2)
            else
                SEG_PEAK="NA"
                Z_PEAK="NA"
            fi

            echo "$i,$ETA,$RSMOOTH,$LAMBDA,$SEG,$SEG_PEAK,$Z_PEAK,$KE,$DURATION,OK" >> "$RESULTS_FILE"
            echo "-> S_final=$SEG, S_peak=$SEG_PEAK (z=$Z_PEAK), KE=$KE (${DURATION}s)"
        else
            echo "$i,$ETA,$RSMOOTH,$LAMBDA,NA,NA,NA,NA,$DURATION,NO_SUMMARY" >> "$RESULTS_FILE"
            echo "-> NO SUMMARY FILE"
        fi
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "$i,$ETA,$RSMOOTH,$LAMBDA,NA,NA,NA,NA,$DURATION,FAILED" >> "$RESULTS_FILE"
        echo "-> FAILED"
    fi

    echo ""
done

echo "========================================"
echo "GRID EXPLORATION COMPLETE"
echo "Results: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
