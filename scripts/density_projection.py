#!/usr/bin/env python3
"""
Visualisation en projection de densité pour snapshots Janus binaires.
Génère frames_density/frame_%05d.png en parallèle des runs.
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import sys
import time


def read_snapshot(path):
    """Read binary snapshot file"""
    with open(path, 'rb') as f:
        # Read header
        n_particles = struct.unpack('<Q', f.read(8))[0]
        n_positive = struct.unpack('<Q', f.read(8))[0]
        eta = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        sim_time = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]

        # Read positions
        positions = np.frombuffer(f.read(), dtype=np.float64).reshape(-1, 3)

    return {
        'n_particles': n_particles,
        'n_positive': n_positive,
        'n_negative': n_particles - n_positive,
        'eta': eta,
        'box_size': box_size,
        'step': step,
        'time': sim_time,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'positions': positions
    }


def render_density_frame(
    snapshot_path: str,
    output_path: str,
    grid_size: int = 1024,
    sigma: float = 2.0,
):
    """Render density projection frame from binary snapshot."""
    data = read_snapshot(snapshot_path)

    positions = data['positions']
    n_positive = data['n_positive']
    box_size = data['box_size']
    step = data['step']
    sim_time = data['time']
    seg = data['seg']
    ke_ratio = data['ke_ratio']
    eta = data['eta']
    n_particles = data['n_particles']

    # Split by sign (first n_positive are positive, rest are negative)
    pos_plus = positions[:n_positive]
    pos_minus = positions[n_positive:]

    # Shift positions from [-half_box, half_box] to [0, box_size] for histogram
    half_box = box_size / 2
    pos_plus_shifted = pos_plus + half_box
    pos_minus_shifted = pos_minus + half_box

    def project(p):
        """Projection le long de l'axe Z (somme sur Z)"""
        H, _, _ = np.histogram2d(
            p[:, 0], p[:, 1],
            bins=grid_size,
            range=[[0, box_size], [0, box_size]]
        )
        H = gaussian_filter(H.astype(float), sigma=sigma)
        # Log scale avec protection contre les zéros
        H = np.log1p(H)
        return H

    # Grilles densité
    dens_p = project(pos_plus_shifted) if len(pos_plus) > 0 else np.zeros((grid_size, grid_size))
    dens_m = project(pos_minus_shifted) if len(pos_minus) > 0 else np.zeros((grid_size, grid_size))

    # Normalisation 0-1
    def norm(x):
        xmax = x.max()
        return x / xmax if xmax > 0 else x

    dens_p = norm(dens_p)
    dens_m = norm(dens_m)

    # Composition RGB : bleu = masses+, rouge = masses-
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[:, :, 0] = dens_m          # Rouge   = masses-
    rgb[:, :, 2] = dens_p          # Bleu    = masses+
    rgb[:, :, 1] = np.minimum(dens_p, dens_m) * 0.3  # Légère teinte verte aux intersections
    rgb = np.clip(rgb, 0, 1)

    # Figure 4K
    fig, ax = plt.subplots(figsize=(3840/150, 2160/150), dpi=150)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin='lower',
        extent=[0, box_size, 0, box_size],
        interpolation='bilinear',
    )

    # Overlay
    ax.set_title(
        f"Janus Cosmological Model — Density Projection | N={n_particles/1e6:.1f}M | η={eta:.3f}",
        color='white', fontsize=14, pad=10
    )
    ax.text(
        0.5, 0.02,
        f"Step {step:05d}  |  Time: {sim_time:.3f}  |  Seg: {seg:.4f}  |  KE/KE₀: {ke_ratio:.4f}",
        transform=ax.transAxes,
        color='white', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
    )

    # Legend
    ax.text(0.02, 0.98, "● Positive masses (blue)", transform=ax.transAxes,
            color='#4488FF', fontsize=10, va='top')
    ax.text(0.02, 0.94, "● Negative masses (red)", transform=ax.transAxes,
            color='#FF4444', fontsize=10, va='top')

    ax.axis('off')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', pad_inches=0)
    plt.close()


def watch_and_render(snapshot_dir: str, output_dir: str):
    """
    Mode watch : surveille snapshot_dir et génère un frame
    densité dès qu'un nouveau .bin apparaît.
    """
    snap_dir = Path(snapshot_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = set()

    print(f"Watching {snap_dir} for new snapshots...")
    while True:
        snaps = sorted(snap_dir.glob("snap_*.bin"))
        for snap in snaps:
            if snap.name in processed:
                continue
            step = int(snap.stem.split('_')[1])
            out_path = out_dir / f"frame_{step:05d}.png"
            if out_path.exists():
                processed.add(snap.name)
                continue
            try:
                render_density_frame(str(snap), str(out_path))
                print(f"  Rendered {snap.name} → {out_path.name}")
                processed.add(snap.name)
            except Exception as e:
                print(f"  Error on {snap.name}: {e}")
        time.sleep(10)  # Vérifier toutes les 10 secondes


def render_single(snapshot_path: str, output_path: str):
    """Render a single snapshot."""
    render_density_frame(snapshot_path, output_path)
    print(f"Rendered {snapshot_path} → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Watch mode: python density_projection.py <snapshot_dir> <output_dir>")
        print("  Single:     python density_projection.py --single <snapshot.bin> <output.png>")
        sys.exit(1)

    if sys.argv[1] == "--single":
        render_single(sys.argv[2], sys.argv[3])
    else:
        watch_and_render(sys.argv[1], sys.argv[2])
