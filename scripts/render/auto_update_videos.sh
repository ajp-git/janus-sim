#!/bin/bash
# Auto-update videos as new frames arrive
# Regenerates videos every 5 minutes

OUTPUT_DIR="/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3"

while true; do
    echo "$(date): Updating videos..."

    # Update velocity video
    ls "$OUTPUT_DIR/frames_velocity"/*.png 2>/dev/null | sort > /tmp/vel_list.txt
    N_VEL=$(wc -l < /tmp/vel_list.txt)
    if [ "$N_VEL" -gt 0 ]; then
        awk '{print "file \047" $0 "\047"; print "duration 0.0333"}' /tmp/vel_list.txt > /tmp/velocity_concat.txt
        ffmpeg -y -f concat -safe 0 -i /tmp/velocity_concat.txt \
            -vf "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$OUTPUT_DIR/petit_pure_20m_velocity_4K.mp4" 2>/dev/null
        echo "  Velocity: $N_VEL frames"
    fi

    # Update composite video
    ls "$OUTPUT_DIR/frames_composite"/*.png 2>/dev/null | sort > /tmp/comp_list.txt
    N_COMP=$(wc -l < /tmp/comp_list.txt)
    if [ "$N_COMP" -gt 0 ]; then
        awk '{print "file \047" $0 "\047"; print "duration 0.0333"}' /tmp/comp_list.txt > /tmp/composite_concat.txt
        ffmpeg -y -f concat -safe 0 -i /tmp/composite_concat.txt \
            -vf "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$OUTPUT_DIR/petit_pure_20m_composite_4K.mp4" 2>/dev/null
        echo "  Composite: $N_COMP frames"
    fi

    # Check if simulation is complete
    if ls "$OUTPUT_DIR/snapshots/snap_02000.bin" 2>/dev/null; then
        echo "Simulation complete! Final video update..."
        sleep 120  # Wait for renderers to finish
        # Final update (same as above)
        ls "$OUTPUT_DIR/frames_velocity"/*.png 2>/dev/null | sort > /tmp/vel_list.txt
        awk '{print "file \047" $0 "\047"; print "duration 0.0333"}' /tmp/vel_list.txt > /tmp/velocity_concat.txt
        ffmpeg -y -f concat -safe 0 -i /tmp/velocity_concat.txt \
            -vf "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$OUTPUT_DIR/petit_pure_20m_velocity_4K.mp4" 2>/dev/null

        ls "$OUTPUT_DIR/frames_composite"/*.png 2>/dev/null | sort > /tmp/comp_list.txt
        awk '{print "file \047" $0 "\047"; print "duration 0.0333"}' /tmp/comp_list.txt > /tmp/composite_concat.txt
        ffmpeg -y -f concat -safe 0 -i /tmp/composite_concat.txt \
            -vf "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$OUTPUT_DIR/petit_pure_20m_composite_4K.mp4" 2>/dev/null
        echo "Done!"
        exit 0
    fi

    echo "  Sleeping 5 minutes..."
    sleep 300
done
