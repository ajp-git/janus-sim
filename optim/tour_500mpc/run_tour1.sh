#!/bin/bash
# Tour 1 — 20 η exploration runs (500 Mpc)
# Expected time: ~2-3 hours total

set -e

TOUR_DIR="optim/tour_500mpc/tour1"
OUTPUT_BASE="output/tour_500mpc/tour1"
RESULTS_FILE="$OUTPUT_BASE/tour1_results.csv"

mkdir -p "$OUTPUT_BASE"

# Initialize results file
echo "run,eta,steps,z_final,S_max,dcom_max,ke_ratio_final,status" > "$RESULTS_FILE"

for config in $(ls $TOUR_DIR/*.yaml | sort); do
    run_name=$(basename "$config" .yaml)
    eta=$(echo "$run_name" | grep -oP 'eta\K[0-9.]+')

    echo ""
    echo "========================================"
    echo "Running: $run_name (η=$eta)"
    echo "========================================"

    start_time=$(date +%s)

    # Run simulation
    timeout 1800 docker compose run --rm dev \
        cargo run --release --features cuda,cufft --bin janus_optim -- \
        --config /app/$config 2>&1 | tee "$OUTPUT_BASE/${run_name}.log"

    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract metrics from time_series.csv
    ts_file="$OUTPUT_BASE/$run_name/time_series.csv"
    if [ -f "$ts_file" ]; then
        # Get final values
        last_line=$(tail -1 "$ts_file")
        steps=$(echo "$last_line" | cut -d',' -f1)
        z_final=$(echo "$last_line" | cut -d',' -f2)
        S_max=$(awk -F',' 'NR>1 {if($6>max) max=$6} END {print max}' "$ts_file")
        dcom_max=$(awk -F',' 'NR>1 {if($10>max) max=$10} END {print max}' "$ts_file")
        ke_ratio=$(echo "$last_line" | cut -d',' -f5)

        if [ $exit_code -eq 0 ]; then
            status="COMPLETE"
        else
            status="TIMEOUT"
        fi
    else
        steps=0
        z_final=5.0
        S_max=0
        dcom_max=0
        ke_ratio=0
        status="FAILED"
    fi

    echo "$run_name,$eta,$steps,$z_final,$S_max,$dcom_max,$ke_ratio,$status" >> "$RESULTS_FILE"
    echo "→ Completed in ${duration}s: S_max=$S_max, ΔCOM_max=$dcom_max, status=$status"
done

echo ""
echo "========================================"
echo "TOUR 1 COMPLETE"
echo "========================================"
echo "Results: $RESULTS_FILE"
cat "$RESULTS_FILE"
