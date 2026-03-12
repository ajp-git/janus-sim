#!/usr/bin/env python3
"""
render_frame_CIC.py — Renderer cosmologique CIC (Cloud In Cell)
Standard N-body projection method (Gadget/Illustris/Millennium style)
"""

import sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

BOX = 492.0
RES = 768
SIGMA = 0.8

BG = "#06060f"
C_TEXT = "#aaaacc"

# ------------------------------------------------------------
# Lecture snapshot
# ------------------------------------------------------------

def read_snapshot(path):
    with open(path, "rb") as f:
        n, step, _ = struct.unpack("<QQQ", f.read(24))
        data = np.frombuffer(f.read(n*16), dtype=np.float32).reshape(n,4)
    sign = data[:,3]
    pos_p = data[sign>0,:3]
    pos_m = data[sign<0,:3]
    return n, step, pos_p, pos_m

# ------------------------------------------------------------
# CIC deposition
# ------------------------------------------------------------

def cic_projection(pos, ax0, ax1, debug=False):
    half = BOX/2
    # Décalage demi-pixel pour éviter l'artefact de bord
    x = (pos[:,ax0] + half) / BOX * RES - 0.5
    y = (pos[:,ax1] + half) / BOX * RES - 0.5

    i = np.floor(x).astype(int)
    j = np.floor(y).astype(int)

    if debug:
        print(f"  ax{ax0},ax{ax1}: i range [{i.min()}, {i.max()}], j range [{j.min()}, {j.max()}]")

    dx = x - i
    dy = y - j

    grid = np.zeros((RES,RES), dtype=np.float64)

    i0 = i % RES
    j0 = j % RES
    i1 = (i+1) % RES
    j1 = (j+1) % RES

    w00 = (1-dx)*(1-dy)
    w10 = dx*(1-dy)
    w01 = (1-dx)*dy
    w11 = dx*dy

    np.add.at(grid,(j0,i0),w00)
    np.add.at(grid,(j0,i1),w10)
    np.add.at(grid,(j1,i0),w01)
    np.add.at(grid,(j1,i1),w11)

    return grid

# ------------------------------------------------------------
# projections
# ------------------------------------------------------------

def compute_maps(pos, debug=False):
    xy = cic_projection(pos, 0, 1, debug)
    xz = cic_projection(pos, 0, 2, debug)
    yz = cic_projection(pos, 1, 2, debug)

    if SIGMA>0:
        xy = gaussian_filter(xy,SIGMA)
        xz = gaussian_filter(xz,SIGMA)
        yz = gaussian_filter(yz,SIGMA)

    return xy,xz,yz

# ------------------------------------------------------------
# rendering
# ------------------------------------------------------------

def render(snap, out, z_cosmo=None, seg=None, ke=None):
    n, step, pos_p, pos_m = read_snapshot(snap)

    print(f"Particles + : {len(pos_p):,}")
    print(f"Particles - : {len(pos_m):,}")

    print("pos_p ranges:")
    print(f"  X: {pos_p[:,0].min():.1f} → {pos_p[:,0].max():.1f}")
    print(f"  Y: {pos_p[:,1].min():.1f} → {pos_p[:,1].max():.1f}")
    print(f"  Z: {pos_p[:,2].min():.1f} → {pos_p[:,2].max():.1f}")
    print("pos_m ranges:")
    print(f"  X: {pos_m[:,0].min():.1f} → {pos_m[:,0].max():.1f}")
    print(f"  Y: {pos_m[:,1].min():.1f} → {pos_m[:,1].max():.1f}")
    print(f"  Z: {pos_m[:,2].min():.1f} → {pos_m[:,2].max():.1f}")

    print("Computing CIC density maps...")
    print("pos_p:")
    p_xy, p_xz, p_yz = compute_maps(pos_p, debug=True)
    print("pos_m:")
    m_xy, m_xz, m_yz = compute_maps(pos_m, debug=True)

    vmax = max(
        p_xy.max(), p_xz.max(), p_yz.max(),
        m_xy.max(), m_xz.max(), m_yz.max()
    )

    # Log transform with epsilon (no hard cutoff)
    eps = 0.1
    log_vmax = np.log10(vmax)

    extent = [-BOX/2, BOX/2, -BOX/2, BOX/2]

    fig, axs = plt.subplots(2, 3, figsize=(19.2, 10.8), facecolor=BG)

    # Row labels
    titles_top = ['Projection XY', 'Projection XZ', 'Projection YZ']
    titles_row = ['Masses +', 'Masses −']

    # Custom colormaps: black → color
    from matplotlib.colors import LinearSegmentedColormap
    cmap_blue = LinearSegmentedColormap.from_list('', [BG, '#1a3a5c', '#3080c0', '#60b0ff', '#ffffff'])
    cmap_red  = LinearSegmentedColormap.from_list('', [BG, '#5c1a1a', '#c03030', '#ff6040', '#ffffff'])

    maps = [
        (p_xy, cmap_blue),
        (p_xz, cmap_blue),
        (p_yz, cmap_blue),
        (m_xy, cmap_red),
        (m_xz, cmap_red),
        (m_yz, cmap_red),
    ]

    for idx, (ax, (g, cmap)) in enumerate(zip(axs.flat, maps)):
        # Mask truly empty regions with NaN → transparent
        g_log = np.log10(g + eps)
        g_log[g < 0.5] = np.nan  # cells with <0.5 particles → invisible

        ax.imshow(
            g_log,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-1,
            vmax=log_vmax,
            interpolation="nearest"
        )
        ax.set_facecolor(BG)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

        # Column titles (top row only)
        if idx < 3:
            ax.set_title(titles_top[idx], color=C_TEXT, fontsize=11, pad=8)

        # Row labels (left column only)
        if idx % 3 == 0:
            ax.set_ylabel(titles_row[idx // 3] + '\n\nMpc', color=C_TEXT, fontsize=10)

        ax.set_xlabel('Mpc', color=C_TEXT, fontsize=9)

    # Suptitle
    title = f'JANUS COSMOLOGICAL SIMULATION — N-Body Gravitational Segregation\n'
    if z_cosmo is not None:
        title += f'z = {z_cosmo:.3f}   '
    if seg is not None:
        title += f'Seg = {seg:.4f}   '
    if ke is not None:
        title += f'KE/KE₀ = {ke:.3f}   '
    title += f'Step {step:06d}   N = {n/1e6:.2f}M   Box = {BOX:.0f} Mpc'

    fig.suptitle(title, color='white', fontsize=13, fontfamily='monospace', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Done → {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 render_frame_CIC.py <snap.bin> [out.png] [z] [seg] [ke]")
        sys.exit(1)

    snap = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else snap.replace('.bin', '.png')
    z_c = float(sys.argv[3]) if len(sys.argv) > 3 else None
    seg = float(sys.argv[4]) if len(sys.argv) > 4 else None
    ke = float(sys.argv[5]) if len(sys.argv) > 5 else None

    render(snap, out, z_c, seg, ke)
