#!/bin/bash
# Scan serré μ=32-40, Box=1000 Mpc, N=5M
# 9 runs × ~22 min = ~3h30

MU_VALUES=(32 33 34 35 36 37 38 39 40)
N_PARTICLES=5000000
BOX_SIZE=1000

echo "================================================================"
echo "  Tight μ Scan: μ ∈ {32..40}, Box=${BOX_SIZE} Mpc, N=5M"
echo "================================================================"
echo "  9 runs, ~22 min each → ~3h30 total"
echo ""

for mu in "${MU_VALUES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Starting μ=$mu ($(date))"
    echo "================================================================"

    docker compose run --rm dev cargo run --release --features "cuda cufft" \
        --bin scan_mu -- --mu $mu --n $N_PARTICLES --box $BOX_SIZE

    echo ""
    echo "μ=$mu COMPLETE"
    echo ""
done

echo ""
echo "================================================================"
echo "  ALL 9 RUNS COMPLETE ($(date))"
echo "================================================================"
