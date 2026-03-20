#!/bin/bash
# make_video.sh — Assemble frames into video
# Usage: ./make_video.sh FRAMES_DIR [FPS] [OUTPUT]

FRAMES_DIR="${1:?Usage: $0 FRAMES_DIR [FPS] [OUTPUT]}"
FPS="${2:-30}"
OUTPUT="${3:-${FRAMES_DIR}/janus_3d.mp4}"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Error: Directory not found: $FRAMES_DIR"
    exit 1
fi

# Count frames
N_FRAMES=$(ls -1 "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
if [ "$N_FRAMES" -eq 0 ]; then
    echo "Error: No frame_*.png files in $FRAMES_DIR"
    exit 1
fi

echo "=== Janus 3D Video Assembly ==="
echo "Frames: $N_FRAMES"
echo "FPS: $FPS"
echo "Output: $OUTPUT"
echo ""

# H.265 encoding for 4K
# -crf 18 = high quality
# -preset slow = better compression
# -pix_fmt yuv420p = compatibility

ffmpeg -y \
    -framerate "$FPS" \
    -i "$FRAMES_DIR/frame_%06d.png" \
    -c:v libx265 \
    -crf 18 \
    -preset slow \
    -pix_fmt yuv420p \
    -tag:v hvc1 \
    "$OUTPUT"

if [ $? -eq 0 ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT" 2>/dev/null)
    echo ""
    echo "=== Done ==="
    echo "Output: $OUTPUT"
    echo "Size: $SIZE"
    echo "Duration: ${DURATION}s"
else
    echo "Error: ffmpeg failed"
    exit 1
fi
