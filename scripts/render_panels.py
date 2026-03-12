#!/usr/bin/env python3
"""
Janus Cosmological Simulation — Scientific Panel Renderer

Generates multi-panel visualizations:
- Main: Cosmic web density (Millennium-style)
- Polarization map (+ vs -)
- Web classification (void/filament/node)
- Power spectrum P(k)

Adapted for Janus snapshot format.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import struct
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================

BOX = 492.0  # Mpc
RES = 768    # Grid resolution for projection

N_TRACERS = 2000
tracer_ids = None

# ============================================================
# JANUS SNAPSHOT LOADER
# ============================================================

def load_snapshot(fname):
    """
    Load Janus snapshot binary format:
    Header (24 bytes):
      u64 n_particles
      u64 step_number
      u64 reserved (0)
    Body (n × 16 bytes):
      f32 x, y, z, sign   # sign = +1.0 or -1.0
    """
    with open(fname, 'rb') as f:
        # Read header
        n_particles, step, _ = struct.unpack('<QQQ', f.read(24))

        # Read particle data
        data = np.frombuffer(f.read(n_particles * 16), dtype=np.float32)
        data = data.reshape(n_particles, 4)

    # Separate by sign
    sign = data[:, 3]
    pos_p = data[sign > 0, :3]  # Positive masses
    pos_m = data[sign < 0, :3]  # Negative masses

    return pos_p, pos_m, step


# ============================================================
# DENSITY PROJECTION
# ============================================================

def density_projection(pos, ax1=0, ax2=1, box=BOX, res=RES):
    """Project 3D positions onto 2D density map."""
    H, _, _ = np.histogram2d(
        pos[:, ax1],
        pos[:, ax2],
        bins=res,
        range=[[-box/2, box/2], [-box/2, box/2]]
    )
    return H


def thin_slice(pos, ax1=0, ax2=1, slice_axis=2, thickness=20.0, box=BOX, res=RES):
    """
    Extract thin slice for Millennium-style visualization.
    Shows filaments/nodes/voids instead of smeared blob.

    Args:
        pos: particle positions (N, 3)
        ax1, ax2: axes to project onto (default: X, Y)
        slice_axis: axis perpendicular to slice (default: Z)
        thickness: half-thickness of slice in Mpc (default: 20 Mpc)
    """
    # Select particles within thin slice
    mask = np.abs(pos[:, slice_axis]) < thickness
    slice_pos = pos[mask]

    H, _, _ = np.histogram2d(
        slice_pos[:, ax1],
        slice_pos[:, ax2],
        bins=res,
        range=[[-box/2, box/2], [-box/2, box/2]]
    )

    return H, len(slice_pos)


# ============================================================
# POLARIZATION MAP
# ============================================================

def polarization_map(pos_p, pos_m, ax1=0, ax2=2):
    """Compute polarization P = (ρ+ - ρ-) / (ρ+ + ρ-)"""
    Hp = density_projection(pos_p, ax1, ax2)
    Hm = density_projection(pos_m, ax1, ax2)

    P = (Hp - Hm) / (Hp + Hm + 1e-6)
    return P


# ============================================================
# POWER SPECTRUM
# ============================================================

def power_spectrum(field):
    """Compute 2D power spectrum."""
    F = np.fft.fft2(field)
    P = np.abs(F)**2

    # Radial average
    ny, nx = field.shape
    y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
    r = np.sqrt(x**2 + y**2).astype(int)

    P_shifted = np.fft.fftshift(P)
    r_flat = r.ravel()
    P_flat = P_shifted.ravel()

    r_max = min(nx, ny) // 2
    P_k = np.zeros(r_max)
    counts = np.zeros(r_max)

    for i in range(len(r_flat)):
        if r_flat[i] < r_max:
            P_k[r_flat[i]] += P_flat[i]
            counts[r_flat[i]] += 1

    P_k = P_k / (counts + 1e-10)
    k = np.arange(r_max) * 2 * np.pi / BOX

    return k, P_k


# ============================================================
# TRACERS
# ============================================================

def init_tracers(pos):
    """Initialize tracer particle IDs."""
    global tracer_ids
    n = min(N_TRACERS, len(pos))
    tracer_ids = np.random.choice(len(pos), n, replace=False)


def get_tracers(pos):
    """Get tracer particle positions."""
    if tracer_ids is None:
        return pos[:N_TRACERS]
    valid = tracer_ids[tracer_ids < len(pos)]
    return pos[valid]


# ============================================================
# MILLENNIUM STYLE RENDER
# ============================================================

def millennium_render(density):
    """Apply Millennium-style glow effect."""
    I = np.log1p(density)
    glow = gaussian_filter(I, sigma=2)
    image = I + 0.7 * glow
    return image


# ============================================================
# WEB CLASSIFICATION
# ============================================================

def classify_web(density):
    """Simple void/filament/node classification by density percentiles."""
    d = gaussian_filter(density, 1)

    thresh1 = np.percentile(d, 40)
    thresh2 = np.percentile(d, 75)

    labels = np.zeros_like(d)
    labels[d > thresh1] = 1  # Filaments
    labels[d > thresh2] = 2  # Nodes

    return labels


# ============================================================
# RENDER SCIENTIFIC PANEL
# ============================================================

def render_panels(pos_p, pos_m, step, output_path, slice_thickness=20.0):
    """Render multi-panel scientific visualization."""

    pos = np.vstack([pos_p, pos_m])

    # Compute fields using THIN SLICE (reveals filaments!)
    density, n_slice = thin_slice(pos, 0, 1, slice_axis=2, thickness=slice_thickness)

    # Polarization also on thin slice
    mask_p = np.abs(pos_p[:, 2]) < slice_thickness
    mask_m = np.abs(pos_m[:, 2]) < slice_thickness
    Hp, _ = thin_slice(pos_p, 0, 1, slice_axis=2, thickness=slice_thickness)
    Hm, _ = thin_slice(pos_m, 0, 1, slice_axis=2, thickness=slice_thickness)
    pol = (Hp - Hm) / (Hp + Hm + 1e-6)
    labels = classify_web(density)
    k, Pk = power_spectrum(density)

    # Create figure
    fig = plt.figure(figsize=(16, 9), dpi=200)

    # --------------------------------------------------------
    # MAIN COSMIC WEB (large panel)
    # --------------------------------------------------------
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)

    im = ax.imshow(
        density.T + 1,
        origin="lower",
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=density.max() + 1),
        extent=[-BOX/2, BOX/2, -BOX/2, BOX/2]
    )

    # Tracers
    tracers = get_tracers(pos_p)
    ax.scatter(
        tracers[:, 0],
        tracers[:, 1],
        s=0.5,
        c="cyan",
        alpha=0.4
    )

    ax.set_title(f"Cosmic Web Density — Step {step} (slice ±{slice_thickness} Mpc)", fontsize=14)
    ax.set_xlabel("X (Mpc)")
    ax.set_ylabel("Y (Mpc)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Particle count + 1")

    # --------------------------------------------------------
    # POLARIZATION
    # --------------------------------------------------------
    ax2 = plt.subplot2grid((2, 3), (0, 2))

    im2 = ax2.imshow(
        pol.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        extent=[-BOX/2, BOX/2, -BOX/2, BOX/2]
    )

    ax2.set_title("Polarization P = (ρ+ - ρ-)/(ρ+ + ρ-)", fontsize=11)
    ax2.set_xlabel("X (Mpc)")
    ax2.set_ylabel("Y (Mpc)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # --------------------------------------------------------
    # POWER SPECTRUM
    # --------------------------------------------------------
    ax3 = plt.subplot2grid((2, 3), (1, 2))

    # Plot only positive k values
    valid = (k > 0) & (Pk > 0)
    ax3.loglog(k[valid], Pk[valid], 'b-', linewidth=1.5)

    ax3.set_xlabel("k (Mpc⁻¹)")
    ax3.set_ylabel("P(k)")
    ax3.set_title("Power Spectrum", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Mark key scales
    k_box = 2 * np.pi / BOX
    ax3.axvline(k_box, color='red', linestyle='--', alpha=0.5, label=f'k_box = {k_box:.3f}')
    ax3.legend(fontsize=8)

    # --------------------------------------------------------
    # Stats text
    # --------------------------------------------------------
    n_plus = len(pos_p)
    n_minus = len(pos_m)
    eta = n_minus / n_plus

    stats_text = f"N+ = {n_plus:,}\nN- = {n_minus:,}\nη = {eta:.3f}\nSlice: {n_slice:,} particles"
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(facecolor='white', alpha=0.8))

    # --------------------------------------------------------
    fig.suptitle("JANUS COSMOLOGICAL SIMULATION", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# RENDER MILLENNIUM STYLE IMAGE
# ============================================================

def render_millennium(pos_p, pos_m, step, output_path, slice_thickness=20.0):
    """Render single Millennium-style density image using thin slice."""

    pos = np.vstack([pos_p, pos_m])
    density, n_slice = thin_slice(pos, 0, 1, slice_axis=2, thickness=slice_thickness)
    image = millennium_render(density)

    plt.figure(figsize=(12, 12))

    plt.imshow(
        image.T,
        origin="lower",
        cmap="inferno",
        extent=[-BOX/2, BOX/2, -BOX/2, BOX/2]
    )

    plt.title(f"Janus Cosmic Web — Step {step}", fontsize=14, color='white')
    plt.xlabel("X (Mpc)", color='white')
    plt.ylabel("Y (Mpc)", color='white')
    plt.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='black')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_panels.py <snapshot.bin> [output_dir]")
        print("Example: python render_panels.py snapshots/snap_000500.bin ./frames")
        sys.exit(1)

    snap_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {snap_file}...")
    pos_p, pos_m, step = load_snapshot(snap_file)
    print(f"  N+ = {len(pos_p):,}, N- = {len(pos_m):,}, Step = {step}")

    # Initialize tracers
    init_tracers(pos_p)

    # Render both styles
    panel_path = os.path.join(output_dir, f"panel_{step:06d}.png")
    millennium_path = os.path.join(output_dir, f"millennium_{step:06d}.png")

    render_panels(pos_p, pos_m, step, panel_path)
    render_millennium(pos_p, pos_m, step, millennium_path)

    print("Done!")
