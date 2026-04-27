#!/bin/bash
# Run CI amplitude scan - 7 values from 0.002% to 50%

DELTAS=(0.002 0.1 1.0 5.0 10.0 20.0 50.0)
LABELS=("baseline" "x50" "moderate" "strong" "near_nl" "very_strong" "quasi_nl")

cd /mnt/T2/janus-sim

for i in "${!DELTAS[@]}"; do
    DELTA=${DELTAS[$i]}
    LABEL=${LABELS[$i]}

    echo ""
    echo "========================================================"
    echo "  RUN $((i+1))/7 : δ_init = ${DELTA}% ($LABEL)"
    echo "========================================================"
    echo ""

    docker compose run --rm dev cargo run --release --features "cuda cufft" \
        --bin scan_ci_amplitude -- --delta ${DELTA}

    echo ""
    echo "Run ${LABEL} (δ=${DELTA}%) DONE"
    echo ""
done

echo ""
echo "========================================================"
echo "  ALL 7 SCANS COMPLETE"
echo "========================================================"

# Show summaries
echo ""
echo "Results:"
for i in "${!DELTAS[@]}"; do
    DELTA=${DELTAS[$i]}
    SUMMARY="/mnt/T2/janus-sim/output/ci_scan_delta${DELTA}/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "δ=${DELTA}%: $(cat $SUMMARY | grep -o '"rho_max_ratio": [0-9.]*' | cut -d: -f2)"
    fi
done
