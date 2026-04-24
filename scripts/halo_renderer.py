#!/usr/bin/env python3
"""
Halo Renderer Daemon — Generates detailed halo analysis frames
Layout 2×3 :
  [0,0] XY projection m+  (split_level)
  [0,1] XZ projection m+  (split_level)
  [0,2] XY velocity map m+ (radial velocity v_r)
  [1,0] Radial density profile ρ(r)  — m+ and m-
  [1,1] XY tangential velocity m+    (|v_tangential|)
  [1,2] Radial velocity profile v_r(r) — m+ and m-
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import struct
import time
import argparse
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
MARGIN            = 50.0    # Mpc — forbidden zone on each edge
MPC_GYR_TO_KMS   = 977.8   # 1 Mpc/Gyr = 977.8 km/s
R_EXTRACT_DEFAULT = 25.0    # Mpc


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT READER  (format V3)
# ─────────────────────────────────────────────────────────────────────────────
def read_snapshot_v3(path):
    with open(path, 'rb') as f:
        header = f.read(408)
        n     = struct.unpack('<Q', header[16:24])[0]
        a     = struct.unpack('<d', header[24:32])[0]
        l_box = struct.unpack('<d', header[40:48])[0]
        z     = 1.0 / a - 1.0
        dt = np.dtype([
            ('x',  '<f4'), ('y',  '<f4'), ('z',  '<f4'),
            ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.fromfile(f, dtype=dt, count=n)
    pos = np.column_stack([particles['x'], particles['y'], particles['z']])
    vel = np.column_stack([particles['vx'], particles['vy'], particles['vz']])
    return {
        'pos': pos, 'vel': vel,
        'sign': particles['sign'], 'split_level': particles['split_level'],
        'mass': particles['mass'], 'z': z, 'a': a, 'l_box': l_box, 'n': n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS CSV
# ─────────────────────────────────────────────────────────────────────────────
def read_analysis_csv(path):
    halos_by_step = {}
    try:
        with open(path, 'r') as f:
            header   = f.readline().strip().split(',')
            step_idx = header.index('step') if 'step' in header else 0
            halo_cols = {}
            for i in range(10):
                xc, yc, zc = f'halo{i}_x', f'halo{i}_y', f'halo{i}_z'
                if xc in header and yc in header and zc in header:
                    halo_cols[i] = (header.index(xc), header.index(yc), header.index(zc))
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                try:
                    step  = int(parts[step_idx])
                    halos = []
                    for i in sorted(halo_cols.keys()):
                        xi, yi, zi = halo_cols[i]
                        x = float(parts[xi]) if parts[xi] else np.nan
                        y = float(parts[yi]) if parts[yi] else np.nan
                        z = float(parts[zi]) if parts[zi] else np.nan
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            halos.append(np.array([x, y, z]))
                    if halos:
                        halos_by_step[step] = halos
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return halos_by_step


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY PEAK FINDER  (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def find_density_peaks(pos, signs, l_box, n_halos=4, grid_size=64, border=20.0):
    from scipy.ndimage import maximum_filter, gaussian_filter
    half     = l_box / 2
    is_plus  = signs == 1
    pos_plus = pos[is_plus]
    if len(pos_plus) < 100:
        return []
    cell = l_box / grid_size
    grid = np.zeros((grid_size, grid_size, grid_size))
    ix = ((pos_plus[:, 0] + half) / cell).astype(int) % grid_size
    iy = ((pos_plus[:, 1] + half) / cell).astype(int) % grid_size
    iz = ((pos_plus[:, 2] + half) / cell).astype(int) % grid_size
    np.add.at(grid, (ix, iy, iz), 1)
    gs        = gaussian_filter(grid, sigma=1.5)
    local_max = maximum_filter(gs, size=5)
    peaks     = (gs == local_max) & (gs > np.percentile(gs, 99))
    pcoords   = np.argwhere(peaks)
    pvals     = gs[peaks]
    order     = np.argsort(pvals)[::-1]
    pcoords   = pcoords[order]
    halos = []
    for pc in pcoords:
        if len(halos) >= n_halos:
            break
        cx = (pc[0] + 0.5) * cell - half
        cy = (pc[1] + 0.5) * cell - half
        cz = (pc[2] + 0.5) * cell - half
        if abs(cx) > half - border or abs(cy) > half - border or abs(cz) > half - border:
            continue
        halos.append(np.array([cx, cy, cz]))
    return halos


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
def decompose_velocities(ldx, ldy, ldz, lr, lvel):
    """
    Returns (v_r, v_t) in km/s for every particle in the local array.

    v_r = v · r̂              positive = outflow, negative = infall
    v_t = sqrt(|v|² - v_r²)  always ≥ 0, measures rotation / turbulence
    """
    r_safe = np.maximum(lr, 1e-10)
    r_hat  = np.column_stack([ldx / r_safe, ldy / r_safe, ldz / r_safe])

    vr_mpc   = np.sum(lvel * r_hat, axis=1)          # Mpc/Gyr
    vr_kms   = vr_mpc * MPC_GYR_TO_KMS

    v2_kms2  = np.sum((lvel * MPC_GYR_TO_KMS) ** 2, axis=1)
    vt2_kms2 = np.maximum(v2_kms2 - vr_kms ** 2, 0.0)
    vt_kms   = np.sqrt(vt2_kms2)

    return vr_kms, vt_kms


# ─────────────────────────────────────────────────────────────────────────────
# AXIS STYLE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def style_scatter(ax, r_extract, xlabel, ylabel, title, tc='white'):
    ax.set_facecolor('black')
    ax.set_xlim(-r_extract, r_extract)
    ax.set_ylim(-r_extract, r_extract)
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color=tc)
    ax.tick_params(colors='gray')
    ax.set_aspect('equal')

def add_cb(sc, ax, label):
    cb = plt.colorbar(sc, ax=ax, label=label)
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='gray')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_halo(data, halo_center, halo_idx, step, out_dir, r_extract=R_EXTRACT_DEFAULT):
    pos   = data['pos'];  vel = data['vel']
    sign  = data['sign']; spl = data['split_level']
    mass  = data['mass']; z   = data['z']
    l_box = data['l_box']; half = l_box / 2

    # reject halos near box edge
    safe_limit = half - MARGIN
    if (abs(halo_center[0]) > safe_limit or
        abs(halo_center[1]) > safe_limit or
        abs(halo_center[2]) > safe_limit):
        return None

    # periodic displacement
    dx = pos[:, 0] - halo_center[0]
    dy = pos[:, 1] - halo_center[1]
    dz = pos[:, 2] - halo_center[2]
    for d in [dx, dy, dz]:
        pass
    dx = np.where(dx >  half, dx - l_box, np.where(dx < -half, dx + l_box, dx))
    dy = np.where(dy >  half, dy - l_box, np.where(dy < -half, dy + l_box, dy))
    dz = np.where(dz >  half, dz - l_box, np.where(dz < -half, dz + l_box, dz))

    r    = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = r < r_extract
    if np.sum(mask) < 10 or np.sum(sign[mask] == 1) < 100:
        return None

    # local arrays
    ldx  = dx[mask];  ldy  = dy[mask];  ldz  = dz[mask]
    lr   = r[mask];   lspl = spl[mask]; lmass = mass[mask]
    lsign = sign[mask]; lvel = vel[mask]

    is_plus  = (lsign == 1)
    is_minus = ~is_plus

    # Subtract centre-of-mass velocity (mass-weighted) to reveal internal dynamics
    total_mass = np.sum(lmass)
    v_com      = np.sum(lvel * lmass[:, np.newaxis], axis=0) / total_mass  # Mpc/Gyr
    lvel_cm    = lvel - v_com[np.newaxis, :]  # velocities in halo rest frame

    vr_all, vt_all = decompose_velocities(ldx, ldy, ldz, lr, lvel_cm)
    vr_plus  = vr_all[is_plus]
    vt_plus  = vt_all[is_plus]
    vr_minus = vr_all[is_minus]

    # colour limits
    max_split  = int(max(6, lspl.max() + 1))
    cmap_split = ListedColormap(plt.cm.viridis(np.linspace(0, 1, max_split)))

    vr_abs_max = max(np.percentile(np.abs(vr_plus), 97) if len(vr_plus) else 1000, 100.0)
    norm_vr    = TwoSlopeNorm(vmin=-vr_abs_max, vcenter=0, vmax=vr_abs_max)

    vt_max = max(np.percentile(vt_plus, 97) if len(vt_plus) else 1000, 100.0)

    # radial bins
    r_bins    = np.linspace(0, r_extract, 35)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # ── figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='black')
    fig.suptitle(
        f'Halo {halo_idx+1} — Step {step} | z = {z:.2f}\n'
        f'Center: ({halo_center[0]:.1f}, {halo_center[1]:.1f}, {halo_center[2]:.1f}) Mpc',
        color='white', fontsize=14)

    # ── [0,0]  XY m+  split level ──
    ax = axes[0, 0]
    if np.sum(is_plus) > 0:
        o = np.argsort(lspl[is_plus])
        sc = ax.scatter(ldx[is_plus][o], ldy[is_plus][o],
                        c=lspl[is_plus][o], cmap=cmap_split,
                        vmin=0, vmax=max_split-1, s=2, alpha=0.75, rasterized=True)
        add_cb(sc, ax, 'Split Level')
    style_scatter(ax, r_extract, 'ΔX [Mpc]', 'ΔY [Mpc]', 'XY Projection (m+)', '#44aaff')

    # ── [0,1]  XZ m+  split level ──
    ax = axes[0, 1]
    if np.sum(is_plus) > 0:
        o = np.argsort(lspl[is_plus])
        sc = ax.scatter(ldx[is_plus][o], ldz[is_plus][o],
                        c=lspl[is_plus][o], cmap=cmap_split,
                        vmin=0, vmax=max_split-1, s=2, alpha=0.75, rasterized=True)
        add_cb(sc, ax, 'Split Level')
    style_scatter(ax, r_extract, 'ΔX [Mpc]', 'ΔZ [Mpc]', 'XZ Projection (m+)', '#44aaff')

    # ── [0,2]  XY m+  radial velocity v_r ──
    ax = axes[0, 2]
    if np.sum(is_plus) > 0:
        o = np.argsort(np.abs(vr_plus))
        sc = ax.scatter(ldx[is_plus][o], ldy[is_plus][o],
                        c=vr_plus[o], cmap=plt.cm.RdBu_r, norm=norm_vr,
                        s=2, alpha=0.80, rasterized=True)
        add_cb(sc, ax, 'v_r [km/s]')
    style_scatter(ax, r_extract, 'ΔX [Mpc]', 'ΔY [Mpc]', 'Radial Velocity v_r (m+)', '#ffaa44')
    ax.text(0.02, 0.97, 'Blue = infall  |  Red = outflow',
            transform=ax.transAxes, color='white', fontsize=8, va='top', alpha=0.7)

    # ── [1,0]  Radial density profile  m+ and m- ──
    ax = axes[1, 0]
    ax.set_facecolor('black')
    rho_plus  = np.zeros(len(r_centers))
    rho_minus = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        shell = (lr >= r_bins[i]) & (lr < r_bins[i+1])
        vol   = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)
        if vol > 0:
            rho_plus[i]  = np.sum(lmass[shell & is_plus])  / vol
            rho_minus[i] = np.sum(lmass[shell & is_minus]) / vol
    ax.semilogy(r_centers, rho_plus  + 1e-10, 'b-',  lw=2, label='m+')
    ax.semilogy(r_centers, rho_minus + 1e-10, 'r--', lw=2, label='m-')
    ax.set_xlabel('r [Mpc]',      color='white')
    ax.set_ylabel('ρ [M☉/Mpc³]', color='white')
    ax.set_title('Radial Density Profile', color='white')
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2); ax.set_xlim(0, r_extract)
    # stats
    stats = (f'N_total: {len(lr):,}\nN_m+: {np.sum(is_plus):,}\n'
             f'N_m-: {np.sum(is_minus):,}\nM_total: {np.sum(lmass):.2e} M☉\n'
             f'split_max: {lspl.max()}')
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            color='white', fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # ── [1,1]  XY m+  tangential velocity v_t ──
    ax = axes[1, 1]
    if np.sum(is_plus) > 0:
        o = np.argsort(vt_plus)
        sc = ax.scatter(ldx[is_plus][o], ldy[is_plus][o],
                        c=vt_plus[o], cmap=plt.cm.plasma,
                        vmin=0, vmax=vt_max,
                        s=2, alpha=0.80, rasterized=True)
        add_cb(sc, ax, 'v_t [km/s]')
    style_scatter(ax, r_extract, 'ΔX [Mpc]', 'ΔY [Mpc]',
                  'Tangential Velocity v_t (m+)', '#cc88ff')
    ax.text(0.02, 0.97, 'Bright = fast rotation / turbulence',
            transform=ax.transAxes, color='white', fontsize=8, va='top', alpha=0.7)

    # ── [1,2]  Radial velocity profile  m+ and m- ──
    ax = axes[1, 2]
    ax.set_facecolor('black')
    vr_p_mean = np.zeros(len(r_centers)); vr_p_std = np.zeros(len(r_centers))
    vr_m_mean = np.zeros(len(r_centers)); vr_m_std = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        shell = (lr >= r_bins[i]) & (lr < r_bins[i+1])
        sp = shell & is_plus;  sm = shell & is_minus
        if np.sum(sp) > 0:
            vr_p_mean[i] = np.mean(vr_all[sp]); vr_p_std[i] = np.std(vr_all[sp])
        if np.sum(sm) > 0:
            vr_m_mean[i] = np.mean(vr_all[sm]); vr_m_std[i] = np.std(vr_all[sm])
    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.plot(r_centers, vr_p_mean, 'b-',  lw=2, label='m+')
    ax.fill_between(r_centers, vr_p_mean - vr_p_std, vr_p_mean + vr_p_std,
                    color='blue', alpha=0.15)
    ax.plot(r_centers, vr_m_mean, 'r--', lw=2, label='m-')
    ax.fill_between(r_centers, vr_m_mean - vr_m_std, vr_m_mean + vr_m_std,
                    color='red', alpha=0.15)
    ax.set_xlabel('r [Mpc]',       color='white')
    ax.set_ylabel('⟨v_r⟩ [km/s]', color='white')
    ax.set_title('Radial Velocity Profile', color='#ffaa44')
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2); ax.set_xlim(0, r_extract)

    plt.tight_layout()
    out_path = out_dir / f'frame_halo{halo_idx+1}_step{step:05d}.png'
    fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# DAEMON
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Halo Renderer Daemon')
    parser.add_argument('--snap-dir',  required=True)
    parser.add_argument('--analysis',  default=None)
    parser.add_argument('--out-dir',   required=True)
    parser.add_argument('--n-halos',   type=int,   default=4)
    parser.add_argument('--r-extract', type=float, default=R_EXTRACT_DEFAULT)
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("=== Halo Renderer Daemon ===")
    print(f"Snap dir : {snap_dir}\nAnalysis : {args.analysis}")
    print(f"Out dir  : {out_dir}\nN halos  : {args.n_halos},  R extract : {args.r_extract} Mpc\n")

    rendered = set()
    for f in out_dir.glob('frame_halo*_step*.png'):
        try:
            step = int(f.stem.split('step')[1])
            halo = int(f.stem.split('halo')[1].split('_')[0])
            rendered.add((halo, step))
        except Exception:
            pass
    print(f"Already rendered: {len(rendered)} halo frames")

    attempted_steps = set()

    while True:
        halos_by_step = {}
        if args.analysis and Path(args.analysis).exists():
            halos_by_step = read_analysis_csv(args.analysis)

        for snap_path in sorted(snap_dir.glob('snap_*.bin')):
            try:
                step = int(snap_path.stem.split('_')[1])
            except Exception:
                continue
            if step in attempted_steps:
                continue
            if all((h+1, step) in rendered for h in range(args.n_halos)):
                attempted_steps.add(step); continue

            try:
                data = read_snapshot_v3(str(snap_path))
            except Exception as e:
                print(f"[ERROR] {snap_path.name}: {e}"); continue

            halos = (halos_by_step.get(step, [])[:args.n_halos]
                     or find_density_peaks(data['pos'], data['sign'],
                                           data['l_box'], args.n_halos))
            if not halos:
                attempted_steps.add(step); continue

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Step {step}: {len(halos)} halos...", end=" ", flush=True)

            for i, center in enumerate(halos):
                if (i+1, step) in rendered:
                    continue
                try:
                    result = render_halo(data, center, i, step, out_dir, args.r_extract)
                    if result:
                        print(f"H{i+1}:OK", end=" ", flush=True)
                        rendered.add((i+1, step))
                except Exception as e:
                    print(f"H{i+1}:ERR({e})", end=" ", flush=True)

            print()
            attempted_steps.add(step)

        time.sleep(30)


if __name__ == '__main__':
    main()
