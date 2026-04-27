#!/bin/bash
# Scan μ fin — 5M particules
# μ ∈ {20, 24, 28, 32, 36, 40, 48}

cd /mnt/T2/janus-sim

MU_VALUES=(20 24 28 32 36 40 48)
N_PARTICLES=5000000

echo "=============================================="
echo "  Scan μ FIN — 5M particules"
echo "=============================================="
echo "  μ values: ${MU_VALUES[*]}"
echo "  N = $N_PARTICLES"
echo "  Estimated time: ~3h (7 × 25 min)"
echo "=============================================="
echo ""

for i in "${!MU_VALUES[@]}"; do
    mu=${MU_VALUES[$i]}
    echo "[$(($i+1))/7] Running μ=$mu (5M)..."
    docker compose run --rm dev cargo run --release --features "cuda cufft" \
        --bin scan_mu -- --mu $mu --n $N_PARTICLES 2>&1 | tee /tmp/scan_mu_${mu}_5M.log
    echo ""
done

echo "=============================================="
echo "  Scan fin COMPLET!"
echo "=============================================="

# Summary
echo ""
echo "Results:"
for mu in "${MU_VALUES[@]}"; do
    if [ -f "/mnt/T2/janus-sim/output/scan_mu_${mu}_5M/summary.json" ]; then
        void=$(grep void_frac /mnt/T2/janus-sim/output/scan_mu_${mu}_5M/summary.json | grep -oP '[\d.]+')
        echo "  μ=$mu: void_frac = ${void}%"
    fi
done
