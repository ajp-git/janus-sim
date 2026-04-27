#!/usr/bin/env python3
"""
Étape 3a - Analyse P(k) préliminaire
Tests BAO sur snapshots existants

Usage:
    python analyse_pk_etape3a.py <snapshot.bin>
"""

import struct
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_snapshot_v2(path):
    """Load snapshot with format: u32 n, f32 box, u32 step, f32 z, N×(f32 x,y,z,sign)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)

    return {
        'n': n,
        'box': box,
        'step': step,
        'z': z,
        'pos': data[:, :3].astype(np.float64),
        'sign': data[:, 3].astype(np.float64)
    }

def load_snapshot_v1(path):
    """Load snapshot with format: u64 n, N×28 bytes (7×f32: x,y,z,vx,vy,vz,sign)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)

    return {
        'n': n,
        'box': 500.0,  # Default
        'step': 0,
        'z': 0.0,
        'pos': data[:, :3].astype(np.float64),
        'sign': data[:, 6].astype(np.float64)
    }

def load_snapshot_jsnp(path, max_particles=1_000_000):
    """Load JSNP format: magic + version + n + z + box + particles

    For large files, randomly sample max_particles to save memory.
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]  # f64
        box = struct.unpack('<d', f.read(8))[0]  # f64

        header_size = 4 + 4 + 8 + 8 + 8  # = 32
        bytes_per_particle = 26  # 3*8 + 2*1

        # Random sampling for large files
        if n > max_particles:
            print(f"  Sampling {max_particles:,} / {n:,} particles...")
            indices = np.random.choice(n, max_particles, replace=False)
            indices = np.sort(indices)

            pos = np.zeros((max_particles, 3), dtype=np.float64)
            sign = np.zeros(max_particles, dtype=np.float64)

            for i, idx in enumerate(indices):
                f.seek(header_size + idx * bytes_per_particle)
                particle = f.read(bytes_per_particle)
                pos[i, 0] = struct.unpack('<d', particle[0:8])[0]
                pos[i, 1] = struct.unpack('<d', particle[8:16])[0]
                pos[i, 2] = struct.unpack('<d', particle[16:24])[0]
                s = struct.unpack('<b', particle[24:25])[0]
                sign[i] = 1.0 if s > 0 else -1.0

            n_actual = max_particles
        else:
            # Load all particles
            remaining = f.read()
            n_actual = len(remaining) // bytes_per_particle

            pos = np.zeros((n_actual, 3), dtype=np.float64)
            sign = np.zeros(n_actual, dtype=np.float64)

            for i in range(n_actual):
                offset = i * bytes_per_particle
                pos[i, 0] = struct.unpack('<d', remaining[offset:offset+8])[0]
                pos[i, 1] = struct.unpack('<d', remaining[offset+8:offset+16])[0]
                pos[i, 2] = struct.unpack('<d', remaining[offset+16:offset+24])[0]
                s = struct.unpack('<b', remaining[offset+24:offset+25])[0]
                sign[i] = 1.0 if s > 0 else -1.0

    return {
        'n': n_actual,
        'box': float(box),
        'step': 0,
        'z': float(z),
        'pos': pos,
        'sign': sign
    }

def detect_and_load(path):
    """Auto-detect snapshot format and load"""
    with open(path, 'rb') as f:
        first_8 = f.read(8)

    # Check for JSNP magic
    if first_8[:4] == b'JSNP':
        return load_snapshot_jsnp(path)

    # Check first u32 as particle count
    n_v2 = struct.unpack('<I', first_8[:4])[0]
    # Check first u64 as particle count
    n_v1 = struct.unpack('<Q', first_8)[0]

    file_size = Path(path).stat().st_size

    # V2 format: 16 byte header + 16 bytes/particle
    expected_v2 = 16 + n_v2 * 16
    # V1 format: 8 byte header + 28 bytes/particle
    expected_v1 = 8 + n_v1 * 28

    if abs(expected_v2 - file_size) < 100:
        return load_snapshot_v2(path)
    elif abs(expected_v1 - file_size) < 100:
        return load_snapshot_v1(path)
    else:
        raise ValueError(f"Unknown snapshot format: {file_size} bytes")

def compute_pk(pos, box, G=128):
    """
    Compute power spectrum P(k) using FFT

    pos: (N,3) positions in Mpc
    box: box size in Mpc
    G: grid resolution
    """
    N_part = len(pos)

    # Grid positions (wrap to [0, box])
    pos_wrapped = pos % box

    # NGP assignment to grid
    ix = np.clip((pos_wrapped[:, 0] / box * G).astype(np.int32), 0, G-1)
    iy = np.clip((pos_wrapped[:, 1] / box * G).astype(np.int32), 0, G-1)
    iz = np.clip((pos_wrapped[:, 2] / box * G).astype(np.int32), 0, G-1)

    grid = np.zeros((G, G, G), dtype=np.float64)
    np.add.at(grid, (iz, iy, ix), 1.0)

    # Density contrast delta = (n - n_mean) / n_mean
    n_mean = N_part / G**3
    delta = (grid - n_mean) / (n_mean + 1e-30)

    # 3D FFT
    delta_k = np.fft.fftn(delta)
    Pk_3d = np.abs(delta_k)**2 * (box / G)**3 / box**3

    # k magnitudes
    kf = 2.0 * np.pi / box
    freq = np.fft.fftfreq(G, d=1.0/G).astype(np.int32)
    kx, ky, kz = np.meshgrid(freq * kf, freq * kf, freq * kf, indexing='ij')
    k_3d = np.sqrt(kx**2 + ky**2 + kz**2)

    # Radial binning
    k_max = np.pi * G / box
    n_bins = G // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    P_k = np.zeros(n_bins)
    N_modes = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (k_3d >= k_edges[i]) & (k_3d < k_edges[i+1])
        if np.any(mask):
            P_k[i] = np.mean(Pk_3d[mask])
            N_modes[i] = np.sum(mask)

    # Shot noise subtraction
    shot = box**3 / N_part
    P_k = np.maximum(P_k - shot, 0)

    return k_centers, P_k, N_modes

def lcdm_pk_approx(k, sigma8=0.8, n_s=0.965):
    """Approximate ΛCDM P(k) using Bardeen transfer function"""
    # Simplified T(k) approximation
    k_eq = 0.02  # h/Mpc
    q = k / k_eq

    # Bardeen (1986) transfer function approximation
    T_k = np.log(1 + 2.34*q) / (2.34*q) / (1 + 3.89*q + (16.2*q)**2 + (5.47*q)**3 + (6.71*q)**4)**0.25

    # Primordial spectrum
    P_prim = k**n_s

    # Normalize to sigma8
    P_k = P_prim * T_k**2

    # Rough normalization (not exact)
    P_k *= sigma8**2 * 1e4

    return P_k

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyse_pk_etape3a.py <snapshot.bin>")
        sys.exit(1)

    snap_path = sys.argv[1]
    print(f"Loading snapshot: {snap_path}")

    try:
        data = detect_and_load(snap_path)
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        sys.exit(1)

    print(f"  N particles: {data['n']:,}")
    print(f"  Box size: {data['box']:.1f} Mpc")
    print(f"  Step: {data['step']}")
    print(f"  Redshift: z = {data['z']:.3f}")

    # Compute P(k) for all, m+, m-
    pos = data['pos']
    sign = data['sign']
    box = data['box']

    print("\nComputing P(k)...")

    # All particles
    k, P_all, N = compute_pk(pos, box)

    # m+ only
    mask_plus = sign > 0
    k, P_plus, _ = compute_pk(pos[mask_plus], box)

    # m- only
    mask_minus = sign < 0
    k, P_minus, _ = compute_pk(pos[mask_minus], box)

    # ΛCDM reference
    P_lcdm = lcdm_pk_approx(k)

    # Compute sigma_8 approximation
    # sigma_8^2 = integral of P(k) * W(kR)^2 * k^2 / (2*pi^2) dk
    # Simplified: just use P(k) at k ~ 0.1-0.2 h/Mpc
    k_s8 = 0.1  # h/Mpc
    idx_s8 = np.argmin(np.abs(k - k_s8))
    if idx_s8 > 0 and P_all[idx_s8] > 0:
        sigma8_approx = np.sqrt(P_all[idx_s8] / P_lcdm[idx_s8]) * 0.8
    else:
        sigma8_approx = 0.0

    print(f"\n=== ÉTAPE 3a RÉSULTATS ===")
    print(f"σ₈ approximatif: {sigma8_approx:.3f}")
    print(f"Cible: σ₈ ∈ [0.65, 0.85]")

    # Check for BAO
    k_bao = 0.1  # BAO peak expected around k ~ 0.06-0.15 h/Mpc
    idx_bao = np.argmin(np.abs(k - k_bao))

    # Look for wiggles in P(k)/P_smooth
    if len(k) > 10:
        # Smooth P(k) with rolling average
        window = 3
        P_smooth = np.convolve(P_all, np.ones(window)/window, mode='same')
        ratio = P_all / (P_smooth + 1e-30)

        # Check for oscillations
        bao_amplitude = np.std(ratio[5:-5])
        print(f"BAO wiggle amplitude: {bao_amplitude:.3f}")
        print(f"BAO détection: {'OUI' if bao_amplitude > 0.05 else 'NON (amplitude < 0.05)'}")

    # Save plot
    out_dir = Path(snap_path).parent.parent / "pk_analysis"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P(k) plot
    ax = axes[0]
    ax.loglog(k, P_all, 'k-', label='Janus (total)', lw=2)
    ax.loglog(k, P_plus, 'b--', label='m+ only', alpha=0.7)
    ax.loglog(k, P_minus, 'r--', label='m- only', alpha=0.7)
    ax.loglog(k, P_lcdm, 'g:', label='ΛCDM (approx)', lw=2)
    ax.axvline(0.1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('k [Mpc⁻¹]')
    ax.set_ylabel('P(k) [Mpc³]')
    ax.set_title(f'Power Spectrum z={data["z"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(k[1], k[-1])

    # Ratio plot
    ax = axes[1]
    valid = (P_lcdm > 0) & (P_all > 0)
    if np.any(valid):
        ax.semilogx(k[valid], P_all[valid] / P_lcdm[valid], 'k-', lw=2, label='Janus/ΛCDM')
        ax.axhline(1.0, color='gray', ls='--')
        ax.axhspan(0.8, 1.2, color='green', alpha=0.1, label='±20%')
    ax.set_xlabel('k [Mpc⁻¹]')
    ax.set_ylabel('P_Janus / P_ΛCDM')
    ax.set_title('Ratio to ΛCDM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    fig.tight_layout()
    out_file = out_dir / f"pk_z{data['z']:.2f}.png"
    fig.savefig(out_file, dpi=150)
    print(f"\nPlot saved: {out_file}")

    # Summary CSV
    csv_file = out_dir / f"pk_z{data['z']:.2f}.csv"
    with open(csv_file, 'w') as f:
        f.write("k,P_all,P_plus,P_minus,P_lcdm,N_modes\n")
        for i in range(len(k)):
            f.write(f"{k[i]:.6f},{P_all[i]:.6e},{P_plus[i]:.6e},{P_minus[i]:.6e},{P_lcdm[i]:.6e},{N[i]}\n")
    print(f"Data saved: {csv_file}")

    print("\n=== Étape 3a: ANALYSE PRÉLIMINAIRE COMPLÈTE ===")

if __name__ == "__main__":
    main()
