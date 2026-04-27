#!/bin/bash
# Render frames for fine scan runs as they complete

MU_VALUES=(20 24 28 32 36 40 48)
N_MILLIONS=5

echo "Rendering frames for fine scan (5M)..."

for mu in "${MU_VALUES[@]}"; do
    RUN_DIR="/mnt/T2/janus-sim/output/scan_mu_${mu}_${N_MILLIONS}M"

    if [ -f "$RUN_DIR/summary.json" ]; then
        echo "Rendering μ=$mu (5M)..."

        # Fix permissions
        sudo chmod -R 777 "$RUN_DIR" 2>/dev/null

        # Render frames
        /tmp/plotenv/bin/python /mnt/T2/janus-sim/scripts/render/render_scan_mu.py \
            $mu --n $N_MILLIONS 0 500 1000 1500 2000
    else
        echo "μ=$mu not ready yet"
    fi
done

echo "Done!"
