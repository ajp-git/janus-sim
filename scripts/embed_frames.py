#!/usr/bin/env python3
"""
Embed frame PNG into TREEPM_ROADMAP.md

Usage: python scripts/embed_frames.py <chemin_frame.png> <étape> <description>

This script automatically updates the IMAGES section of TREEPM_ROADMAP.md
with a reference to the generated frame.
"""
import sys
from pathlib import Path

def embed_image(frame_path: str, step: str, description: str):
    roadmap_path = Path(__file__).parent.parent / "TREEPM_ROADMAP.md"

    if not roadmap_path.exists():
        print(f"Error: {roadmap_path} not found")
        sys.exit(1)

    frame = Path(frame_path)
    if not frame.exists():
        print(f"Warning: {frame_path} does not exist yet")

    # Compute relative path from roadmap to frame
    try:
        rel_path = frame.relative_to(roadmap_path.parent)
    except ValueError:
        # Frame is outside repo, use absolute path
        rel_path = frame

    roadmap = roadmap_path.read_text()

    # Create entry
    entry = f"\n### Étape {step} — {description}\n![{description}]({rel_path})\n"

    # Insert before "## JOURNAL D'EXÉCUTION"
    marker = "## JOURNAL D'EXÉCUTION"
    if marker in roadmap:
        updated = roadmap.replace(marker, entry + "\n" + marker)
        roadmap_path.write_text(updated)
        print(f"Image embedded: {rel_path}")
    else:
        print(f"Error: Could not find '{marker}' in {roadmap_path}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/embed_frames.py <frame.png> <step> <description>")
        print("Example: python scripts/embed_frames.py outputs/frame_00500.png 4 'TreePM 1M step 500'")
        sys.exit(1)

    embed_image(sys.argv[1], sys.argv[2], sys.argv[3])
