#!/bin/bash
# Scan μ — Sequential runs
# Order: μ=16, 32, 64, 4 (μ=8 already done as reference)

cd /mnt/T2/janus-sim

echo "=============================================="
echo "  Scan μ — Janus Calibration"
echo "=============================================="
echo ""

# μ=16
echo "[1/4] Running μ=16..."
docker compose run --rm dev cargo run --release --features "cuda cufft" --bin scan_mu -- --mu 16 2>&1 | tee /tmp/scan_mu_16.log
echo ""

# μ=32
echo "[2/4] Running μ=32..."
docker compose run --rm dev cargo run --release --features "cuda cufft" --bin scan_mu -- --mu 32 2>&1 | tee /tmp/scan_mu_32.log
echo ""

# μ=64
echo "[3/4] Running μ=64..."
docker compose run --rm dev cargo run --release --features "cuda cufft" --bin scan_mu -- --mu 64 2>&1 | tee /tmp/scan_mu_64.log
echo ""

# μ=4
echo "[4/4] Running μ=4..."
docker compose run --rm dev cargo run --release --features "cuda cufft" --bin scan_mu -- --mu 4 2>&1 | tee /tmp/scan_mu_4.log
echo ""

echo "=============================================="
echo "  All scans complete!"
echo "=============================================="
echo ""
echo "Results:"
for mu in 4 16 32 64; do
    if [ -f "/mnt/T2/janus-sim/output/scan_mu_$mu/summary.json" ]; then
        echo "μ=$mu: $(cat /mnt/T2/janus-sim/output/scan_mu_$mu/summary.json | grep void_frac)"
    fi
done
