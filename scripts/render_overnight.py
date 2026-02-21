#!/usr/bin/env python3
"""
Janus N-body visualization — 3-panel layout renderer
Layout:
  ┌─────────────────────────┬───────────┐
  │                         │  Masses+  │
  │   Vue combinée 2D XY    │  (bleu)   │
  │   density (imshow)      ├───────────┤
  │   rouge + bleu          │  Masses−  │
  │                         │  (rouge)  │
  └─────────────────────────┴───────────┘
"""
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def read_snapshot(path):
    """Read binary snapshot file"""
    with open(path, 'rb') as f:
        n_particles = struct.unpack('<Q', f.read(8))[0]
        n_positive = struct.unpack('<Q', f.read(8))[0]
        eta = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        sim_time = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
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

def auto_zoom(positions, percentile_low=2, percentile_high=98, margin=0.1):
    """Compute auto-zoom bounds using percentile + margin"""
    if len(positions) == 0:
        return -1, 1, -1, 1

    x_lo = np.percentile(positions[:, 0], percentile_low)
    x_hi = np.percentile(positions[:, 0], percentile_high)
    y_lo = np.percentile(positions[:, 1], percentile_low)
    y_hi = np.percentile(positions[:, 1], percentile_high)

    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    x_lo -= x_range * margin
    x_hi += x_range * margin
    y_lo -= y_range * margin
    y_hi += y_range * margin

    return x_lo, x_hi, y_lo, y_hi

def render_frame(snapshot_path, output_path, eta, n_particles, step, sim_time, seg, ke_ratio):
    """Render a single 1080p frame with 3-panel layout"""

    data = read_snapshot(snapshot_path)
    positions = data['positions']
    n_positive = data['n_positive']
    n_total = data['n_particles']
    n_negative = n_total - n_positive

    # Split by sign
    pos_plus = positions[:n_positive]
    pos_minus = positions[n_positive:]

    print(f"[DEBUG] N+ = {len(pos_plus):,}, N- = {len(pos_minus):,}, Total = {n_total:,}")

    # Colors
    color_plus = '#00aaff'   # Bright blue
    color_minus = '#ff4444'  # Bright red

    # Create figure 1920x1080
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='black')

    # Layout: left panel 2/3, right panels 1/3 (tight spacing)
    ax_combined = fig.add_axes([0.01, 0.08, 0.64, 0.84], facecolor='black')
    ax_plus = fig.add_axes([0.66, 0.50, 0.33, 0.42], facecolor='black')
    ax_minus = fig.add_axes([0.66, 0.08, 0.33, 0.42], facecolor='black')

    # === LEFT PANEL: Density map with imshow ===
    x_lo, x_hi, y_lo, y_hi = auto_zoom(positions)
    print(f"[DEBUG] Combined panel: x=[{x_lo:.2f}, {x_hi:.2f}], y=[{y_lo:.2f}, {y_hi:.2f}]")

    grid_size = 512
    sigma = 0.8

    # Histogram for positive masses (blue)
    H_plus, xedges, yedges = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size,
        range=[[x_lo, x_hi], [y_lo, y_hi]]
    )
    H_plus = gaussian_filter(H_plus.astype(float), sigma=sigma)
    H_plus = np.log1p(H_plus)
    # Normalize by percentile 99 to avoid saturation from outliers
    p99_plus = np.percentile(H_plus, 99)
    if p99_plus > 0:
        H_plus = np.clip(H_plus / p99_plus, 0, 1)

    # Histogram for negative masses (red)
    H_minus, _, _ = np.histogram2d(
        pos_minus[:, 0], pos_minus[:, 1],
        bins=grid_size,
        range=[[x_lo, x_hi], [y_lo, y_hi]]
    )
    H_minus = gaussian_filter(H_minus.astype(float), sigma=sigma)
    H_minus = np.log1p(H_minus)
    # Normalize by percentile 99
    p99_minus = np.percentile(H_minus, 99)
    if p99_minus > 0:
        H_minus = np.clip(H_minus / p99_minus, 0, 1)

    # Compose RGB: R=minus, G=min*0.3, B=plus
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[:, :, 0] = H_minus          # Red = masses-
    rgb[:, :, 2] = H_plus           # Blue = masses+
    rgb[:, :, 1] = np.minimum(H_plus, H_minus) * 0.3  # Green = overlap
    rgb = np.clip(rgb, 0, 1)

    # imshow expects (rows, cols, 3) - transpose first two axes only
    ax_combined.imshow(
        np.transpose(rgb, (1, 0, 2)),
        origin='lower',
        extent=[x_lo, x_hi, y_lo, y_hi],
        aspect='equal',
        interpolation='bilinear'
    )
    ax_combined.axis('off')
    ax_combined.set_title('Combined (XY density)', color='white', fontsize=14, pad=5)

    # === RIGHT PANELS: Scatter plots ===
    N_max_display = 50000
    point_size = 1.5
    alpha = 0.6

    # Subsample for right panels
    if len(pos_plus) > N_max_display:
        idx = np.random.choice(len(pos_plus), N_max_display, replace=False)
        pos_plus_disp = pos_plus[idx]
    else:
        pos_plus_disp = pos_plus

    if len(pos_minus) > N_max_display:
        idx = np.random.choice(len(pos_minus), N_max_display, replace=False)
        pos_minus_disp = pos_minus[idx]
    else:
        pos_minus_disp = pos_minus

    print(f"[DEBUG] Right panels display: N+={len(pos_plus_disp):,}, N-={len(pos_minus_disp):,}")

    # === RIGHT TOP: Masses+ ===
    x_lo_p, x_hi_p, y_lo_p, y_hi_p = auto_zoom(pos_plus)

    ax_plus.scatter(pos_plus_disp[:, 0], pos_plus_disp[:, 1],
                    c=color_plus, alpha=alpha, s=point_size, marker='.', linewidths=0)
    ax_plus.set_xlim(x_lo_p, x_hi_p)
    ax_plus.set_ylim(y_lo_p, y_hi_p)
    ax_plus.set_aspect('equal', adjustable='box')
    ax_plus.axis('off')
    ax_plus.set_title(f'Masses+ ({n_positive:,})', color=color_plus, fontsize=12, pad=5)

    # === RIGHT BOTTOM: Masses- ===
    x_lo_m, x_hi_m, y_lo_m, y_hi_m = auto_zoom(pos_minus)

    ax_minus.scatter(pos_minus_disp[:, 0], pos_minus_disp[:, 1],
                     c=color_minus, alpha=alpha, s=point_size, marker='.', linewidths=0)
    ax_minus.set_xlim(x_lo_m, x_hi_m)
    ax_minus.set_ylim(y_lo_m, y_hi_m)
    ax_minus.set_aspect('equal', adjustable='box')
    ax_minus.axis('off')
    ax_minus.set_title(f'Masses− ({n_negative:,})', color=color_minus, fontsize=12, pad=5)

    # === Title ===
    title = f"Janus Cosmological Model — {n_total/1e3:.0f}K particles | η={eta:.3f}"
    fig.text(0.5, 0.97, title, ha='center', va='top', fontsize=20,
             color='white', fontweight='bold')

    # === Stats bar ===
    stats = f"Step {step:05d} | Time: {sim_time:.3f} | Seg: {seg:.4f} | KE/KE₀: {ke_ratio:.2f}"
    fig.text(0.5, 0.02, stats, ha='center', va='bottom', fontsize=14,
             color='white', family='monospace')

    plt.savefig(output_path, facecolor='black', edgecolor='none', dpi=100)
    plt.close()
    print(f"[DEBUG] Saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Usage: render_overnight.py <snapshot> <output> <eta> <n_particles> <step> <time> <seg> <ke_ratio>")
        sys.exit(1)

    snapshot_path = sys.argv[1]
    output_path = sys.argv[2]
    eta = float(sys.argv[3])
    n_particles = int(sys.argv[4])
    step = int(sys.argv[5])
    sim_time = float(sys.argv[6])
    seg = float(sys.argv[7])
    ke_ratio = float(sys.argv[8])

    render_frame(snapshot_path, output_path, eta, n_particles, step, sim_time, seg, ke_ratio)
