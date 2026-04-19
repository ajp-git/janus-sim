#!/usr/bin/env python3
"""
JANUS SNAPSHOT ANALYZER DAEMON
Monitors snapshot directory and analyzes new v3 snapshots as they appear.
Writes comprehensive metrics to analysis.csv in real-time.

Usage:
    python3 scripts/snapshot_analyzer.py \
        --snap-dir output/janus_adaptive_10M/snapshots/ \
        --out-csv output/janus_adaptive_10M/analysis.csv \
        --n-halos 4 --grid-size 64
"""

import argparse
import glob
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOT V3 READER
# ═══════════════════════════════════════════════════════════════════════════

MAGIC_V3 = b'JANUSV3\n'
VERSION_V3 = 3

class SnapshotHeaderV3:
    """Header structure for v3 snapshots (408 bytes)"""
    SIZE = 408

    def __init__(self):
        self.magic = MAGIC_V3
        self.version = VERSION_V3
        self.header_size = 408
        self.n_total = 0
        self.a = 0.0
        self.t_gyr = 0.0
        self.l_box = 0.0
        self.h0 = 0.0
        self.mu = 0.0
        self.omega_b = 0.0
        self.m_part_plus_base = 0.0
        self.m_part_minus_base = 0.0
        self.eps_plus_base = 0.0
        self.eps_minus_base = 0.0
        self.n_split_max = 0
        self.seed_ic = 0
        self.z_init = 0.0
        self.n_stars = 0
        self.z_start_run = 0.0
        self.sfr = 0.0
        self.rho_max = 0.0
        self.run_label = ""

    @property
    def z(self):
        return 1.0 / self.a - 1.0 if self.a > 0 else 0.0


class ParticleV3:
    """Particle structure (36 bytes)

    pos: [f32; 3]       12 bytes
    vel: [f32; 3]       12 bytes
    mass: f32            4 bytes
    epsilon: f32         4 bytes
    sign: u8             1 byte
    split_level: u8      1 byte
    is_star: u8          1 byte
    flags: u8            1 byte
    Total: 36 bytes
    """
    SIZE = 36

    def __init__(self, data: bytes):
        # Unpack all fields at once
        vals = struct.unpack('<3f3fffBBBB', data[:36])
        self.pos = np.array(vals[0:3], dtype=np.float32)
        self.vel = np.array(vals[3:6], dtype=np.float32)
        self.mass = vals[6]
        self.epsilon = vals[7]
        self.sign = vals[8]      # 1 for +, 255 for -
        self.split_level = vals[9]
        self.is_star = vals[10]
        self.flags = vals[11]


def read_snapshot_v3(path: str) -> Tuple[SnapshotHeaderV3, List[ParticleV3]]:
    """Read a v3 snapshot file (408-byte header)"""
    with open(path, 'rb') as f:
        # Read header (408 bytes)
        header_data = f.read(408)

        header = SnapshotHeaderV3()

        # Parse header - magic is 8 bytes: "JANUSV3\n"
        magic = header_data[0:8]
        if magic != MAGIC_V3:
            raise ValueError(f"Invalid magic: {magic}")

        # Offset 8: version (u32)
        header.version = struct.unpack('<I', header_data[8:12])[0]
        # Offset 12: header_size (u32)
        header.header_size = struct.unpack('<I', header_data[12:16])[0]
        # Offset 16: n_total (u64)
        header.n_total = struct.unpack('<Q', header_data[16:24])[0]
        # Offset 24: a (f64)
        header.a = struct.unpack('<d', header_data[24:32])[0]
        # Offset 32: t_gyr (f64)
        header.t_gyr = struct.unpack('<d', header_data[32:40])[0]
        # Offset 40: l_box (f64)
        header.l_box = struct.unpack('<d', header_data[40:48])[0]
        # Offset 48: h0 (f64)
        header.h0 = struct.unpack('<d', header_data[48:56])[0]
        # Offset 56: mu (f64)
        header.mu = struct.unpack('<d', header_data[56:64])[0]
        # Offset 64: omega_b (f64)
        header.omega_b = struct.unpack('<d', header_data[64:72])[0]
        # Offset 72: m_part_plus_base (f64)
        header.m_part_plus_base = struct.unpack('<d', header_data[72:80])[0]
        # Offset 80: m_part_minus_base (f64)
        header.m_part_minus_base = struct.unpack('<d', header_data[80:88])[0]
        # Offset 88: eps_plus_base (f64)
        header.eps_plus_base = struct.unpack('<d', header_data[88:96])[0]
        # Offset 96: eps_minus_base (f64)
        header.eps_minus_base = struct.unpack('<d', header_data[96:104])[0]
        # Offset 104: n_split_max (u32)
        header.n_split_max = struct.unpack('<I', header_data[104:108])[0]
        # Offset 108: seed_ic (u32)
        header.seed_ic = struct.unpack('<I', header_data[108:112])[0]
        # Offset 112: z_init (f64)
        header.z_init = struct.unpack('<d', header_data[112:120])[0]
        # Offset 120: n_stars (u64)
        header.n_stars = struct.unpack('<Q', header_data[120:128])[0]
        # Offset 128: z_start_run (f64)
        header.z_start_run = struct.unpack('<d', header_data[128:136])[0]
        # Offset 136: sfr (f64)
        header.sfr = struct.unpack('<d', header_data[136:144])[0]
        # Offset 144: rho_max (f64)
        header.rho_max = struct.unpack('<d', header_data[144:152])[0]
        # Offset 152: run_label (256 bytes, null-terminated string)
        label_bytes = header_data[152:408]
        header.run_label = label_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')

        # FAST: Use numpy structured array to read all particles at once
        dt = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'), ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.fromfile(f, dtype=dt, count=header.n_total)

        return header, particles


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_sph_density_knn(positions: np.ndarray, masses: np.ndarray, k: int = 32) -> np.ndarray:
    """Compute SPH-like density using k-nearest neighbors"""
    n = len(positions)
    if n < k:
        return np.zeros(n)

    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=k+1)  # +1 because point is its own neighbor

    # r_k is the distance to the k-th neighbor (excluding self)
    r_k = distances[:, k]
    r_k = np.maximum(r_k, 1e-10)  # Avoid division by zero

    # Volume of sphere with radius r_k
    volume = (4.0 / 3.0) * np.pi * r_k**3

    # Density = k × m_mean / V
    m_mean = np.mean(masses)
    densities = k * m_mean / volume

    return densities


def find_top_halos(positions: np.ndarray, densities: np.ndarray,
                   l_box: float, n_halos: int = 4, min_sep: float = 10.0,
                   border: float = 20.0) -> List[Dict]:
    """Find top N density peaks with minimum separation

    Args:
        border: Exclude halos within this distance from box edges (Mpc)
    """
    halos = []
    mask = np.ones(len(positions), dtype=bool)
    half = l_box / 2.0

    # Pre-mask particles near edges
    edge_mask = (
        (np.abs(positions[:, 0]) < half - border) &
        (np.abs(positions[:, 1]) < half - border) &
        (np.abs(positions[:, 2]) < half - border)
    )

    for _ in range(n_halos * 3):  # Try more candidates to find n_halos valid ones
        if len(halos) >= n_halos:
            break
        if not np.any(mask):
            break

        # Find max density among unmasked
        masked_densities = np.where(mask, densities, -np.inf)
        idx_max = np.argmax(masked_densities)

        if densities[idx_max] <= 0:
            break

        pos_max = positions[idx_max]
        rho_max = densities[idx_max]

        # Count particles within r < 2 Mpc
        distances = np.linalg.norm(positions - pos_max, axis=1)

        # Mask out particles within min_sep for next iteration
        mask &= (distances >= min_sep)

        # Skip if too close to edge
        if not edge_mask[idx_max]:
            continue

        n_in_halo = np.sum(distances < 2.0)

        halos.append({
            'x': pos_max[0],
            'y': pos_max[1],
            'z': pos_max[2],
            'rho': rho_max,
            'n_particles': n_in_halo,
            'idx': idx_max
        })

    return halos


def compute_segregation(pos_plus: np.ndarray, pos_minus: np.ndarray,
                        l_box: float, grid_size: int = 64) -> Tuple[float, float]:
    """
    Compute segregation metrics:
    - corr_delta: Correlation between δ+ and δ- fields
    - r_k_mean: Mean cross-correlation coefficient
    """
    half = l_box / 2.0
    bins = np.linspace(-half, half, grid_size + 1)

    # 3D histograms
    hist_plus, _ = np.histogramdd(pos_plus, bins=[bins, bins, bins])
    hist_minus, _ = np.histogramdd(pos_minus, bins=[bins, bins, bins])

    # Overdensity fields
    mean_plus = np.mean(hist_plus)
    mean_minus = np.mean(hist_minus)

    if mean_plus > 0 and mean_minus > 0:
        delta_plus = (hist_plus - mean_plus) / mean_plus
        delta_minus = (hist_minus - mean_minus) / mean_minus

        # Correlation coefficient
        delta_plus_flat = delta_plus.flatten()
        delta_minus_flat = delta_minus.flatten()

        corr = np.corrcoef(delta_plus_flat, delta_minus_flat)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    # Cross-correlation in Fourier space
    try:
        fft_plus = np.fft.fftn(hist_plus)
        fft_minus = np.fft.fftn(hist_minus)

        P_plus = np.abs(fft_plus)**2
        P_minus = np.abs(fft_minus)**2
        P_cross = np.real(fft_plus * np.conj(fft_minus))

        # r(k) = P_cross / sqrt(P+ × P-)
        denom = np.sqrt(P_plus * P_minus)
        denom = np.maximum(denom, 1e-10)
        r_k = P_cross / denom

        # Mean r(k) excluding k=0
        r_k_flat = r_k.flatten()
        r_k_mean = np.mean(r_k_flat[1:])  # Exclude DC component
        if np.isnan(r_k_mean):
            r_k_mean = 0.0
    except:
        r_k_mean = 0.0

    return corr, r_k_mean


def analyze_snapshot(header: SnapshotHeaderV3, particles: np.ndarray,
                     n_halos: int = 4, grid_size: int = 64) -> Dict:
    """Analyze a snapshot and return all metrics

    particles is now a numpy structured array with fields:
    x, y, z, vx, vy, vz, mass, epsilon, sign, split_level, is_star, flags
    """

    n = len(particles)
    if n == 0:
        return {}

    # Extract arrays from structured array (FAST)
    pos = np.column_stack([particles['x'], particles['y'], particles['z']]).astype(np.float64)
    vel = np.column_stack([particles['vx'], particles['vy'], particles['vz']]).astype(np.float64)
    masses = particles['mass'].astype(np.float64)
    signs = np.where(particles['sign'] == 1, 1, -1).astype(np.int32)
    split_levels = particles['split_level'].astype(np.int32)
    is_star = particles['is_star'].astype(np.int32)

    # Masks
    mask_plus = signs > 0
    mask_minus = signs < 0
    mask_hr = split_levels > 0
    mask_star = is_star > 0

    pos_plus = pos[mask_plus]
    pos_minus = pos[mask_minus]
    vel_plus = vel[mask_plus]
    masses_plus = masses[mask_plus]

    # ═══════════════════════════════════════════════════════════════════════
    # BASIC COUNTS
    # ═══════════════════════════════════════════════════════════════════════
    n_plus = np.sum(mask_plus)
    n_minus = np.sum(mask_minus)
    n_stars = np.sum(mask_star)

    # Split level counts
    n_split_1 = np.sum(split_levels == 1)
    n_split_2 = np.sum(split_levels == 2)
    n_split_3plus = np.sum(split_levels >= 3)
    split_level_max = int(np.max(split_levels)) if n > 0 else 0

    # ═══════════════════════════════════════════════════════════════════════
    # DENSITY (m+ only)
    # ═══════════════════════════════════════════════════════════════════════
    if len(pos_plus) > 64:
        densities_plus = compute_sph_density_knn(pos_plus, masses_plus, k=32)
        rho_max_plus = float(np.max(densities_plus))
        rho_mean_plus = float(np.mean(densities_plus))
        rho_p50 = float(np.percentile(densities_plus, 50))
        rho_p90 = float(np.percentile(densities_plus, 90))
        rho_p99 = float(np.percentile(densities_plus, 99))
        rho_p999 = float(np.percentile(densities_plus, 99.9))
    else:
        densities_plus = np.zeros(len(pos_plus))
        rho_max_plus = rho_mean_plus = rho_p50 = rho_p90 = rho_p99 = rho_p999 = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # HALOS
    # ═══════════════════════════════════════════════════════════════════════
    halos = find_top_halos(pos_plus, densities_plus, l_box=header.l_box,
                           n_halos=n_halos, min_sep=10.0, border=20.0)

    # Compute additional halo metrics
    for i, halo in enumerate(halos):
        # Find split_max in this halo
        halo_pos = np.array([halo['x'], halo['y'], halo['z']])
        distances = np.linalg.norm(pos - halo_pos, axis=1)
        in_halo = distances < 2.0
        if np.any(in_halo):
            halo['split_max'] = int(np.max(split_levels[in_halo]))
        else:
            halo['split_max'] = 0

    # Pad halos list to n_halos
    while len(halos) < n_halos:
        halos.append({'x': 0, 'y': 0, 'z': 0, 'rho': 0, 'n_particles': 0, 'split_max': 0})

    # ═══════════════════════════════════════════════════════════════════════
    # KINEMATICS
    # ═══════════════════════════════════════════════════════════════════════
    v_mag = np.linalg.norm(vel, axis=1)
    v_rms_global = float(np.sqrt(np.mean(v_mag**2))) * 977.8  # Mpc/Gyr → km/s

    if n_plus > 0:
        v_mag_plus = np.linalg.norm(vel_plus, axis=1)
        v_rms_plus = float(np.sqrt(np.mean(v_mag_plus**2))) * 977.8
    else:
        v_rms_plus = 0.0

    if np.sum(mask_hr) > 0:
        v_mag_hr = np.linalg.norm(vel[mask_hr], axis=1)
        v_rms_hr = float(np.sqrt(np.mean(v_mag_hr**2))) * 977.8
    else:
        v_rms_hr = 0.0

    # Velocity dispersion in halo0
    if halos[0]['n_particles'] > 10:
        halo0_pos = np.array([halos[0]['x'], halos[0]['y'], halos[0]['z']])
        distances = np.linalg.norm(pos - halo0_pos, axis=1)
        in_halo0 = distances < 2.0
        if np.sum(in_halo0) > 1:
            vel_halo0 = vel[in_halo0]
            v_mean = np.mean(vel_halo0, axis=0)
            v_rel = vel_halo0 - v_mean
            sigma_v_halo0 = float(np.sqrt(np.mean(np.sum(v_rel**2, axis=1)))) * 977.8
        else:
            sigma_v_halo0 = 0.0
    else:
        sigma_v_halo0 = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # SEGREGATION
    # ═══════════════════════════════════════════════════════════════════════
    if len(pos_plus) > 100 and len(pos_minus) > 100:
        corr_delta, r_k_mean = compute_segregation(pos_plus, pos_minus, header.l_box, grid_size)
    else:
        corr_delta, r_k_mean = 0.0, 0.0

    # N_minus in halos (r < 5 Mpc from any halo center)
    n_minus_in_halos = 0
    for halo in halos[:n_halos]:
        if halo['rho'] > 0:
            halo_pos = np.array([halo['x'], halo['y'], halo['z']])
            distances = np.linalg.norm(pos_minus - halo_pos, axis=1)
            n_minus_in_halos += np.sum(distances < 5.0)

    # Segregation index in halos
    n_plus_in_halos = 0
    n_minus_in_halos_strict = 0
    for halo in halos[:n_halos]:
        if halo['rho'] > 0:
            halo_pos = np.array([halo['x'], halo['y'], halo['z']])
            d_plus = np.linalg.norm(pos_plus - halo_pos, axis=1)
            d_minus = np.linalg.norm(pos_minus - halo_pos, axis=1)
            n_plus_in_halos += np.sum(d_plus < 5.0)
            n_minus_in_halos_strict += np.sum(d_minus < 5.0)

    if n_plus_in_halos + n_minus_in_halos_strict > 0:
        segregation_index = (n_plus_in_halos - n_minus_in_halos_strict) / (n_plus_in_halos + n_minus_in_halos_strict)
    else:
        segregation_index = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # STAR FORMATION
    # ═══════════════════════════════════════════════════════════════════════
    m_stars_total = float(np.sum(masses[mask_star])) if n_stars > 0 else 0.0
    sfr = header.sfr

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD RESULT
    # ═══════════════════════════════════════════════════════════════════════
    result = {
        # Cosmology
        'step': 0,  # Will be extracted from filename
        'z': header.z,
        't_Gyr': header.t_gyr,
        'a': header.a,

        # Particles
        'N_total': n,
        'N_plus': n_plus,
        'N_minus': n_minus,
        'N_stars': n_stars,
        'split_level_max': split_level_max,
        'N_split_1': n_split_1,
        'N_split_2': n_split_2,
        'N_split_3plus': n_split_3plus,

        # Density m+
        'rho_max_plus': rho_max_plus,
        'rho_mean_plus': rho_mean_plus,
        'rho_p50': rho_p50,
        'rho_p90': rho_p90,
        'rho_p99': rho_p99,
        'rho_p999': rho_p999,

        # Halos
        'halo0_x': halos[0]['x'],
        'halo0_y': halos[0]['y'],
        'halo0_z': halos[0]['z'],
        'halo0_rho': halos[0]['rho'],
        'halo0_n_particles': halos[0]['n_particles'],
        'halo0_split_max': halos[0]['split_max'],

        'halo1_x': halos[1]['x'],
        'halo1_y': halos[1]['y'],
        'halo1_z': halos[1]['z'],
        'halo1_rho': halos[1]['rho'],
        'halo1_n_particles': halos[1]['n_particles'],
        'halo1_split_max': halos[1]['split_max'],

        'halo2_x': halos[2]['x'],
        'halo2_y': halos[2]['y'],
        'halo2_z': halos[2]['z'],
        'halo2_rho': halos[2]['rho'],
        'halo2_n_particles': halos[2]['n_particles'],
        'halo2_split_max': halos[2]['split_max'],

        'halo3_x': halos[3]['x'],
        'halo3_y': halos[3]['y'],
        'halo3_z': halos[3]['z'],
        'halo3_rho': halos[3]['rho'],
        'halo3_n_particles': halos[3]['n_particles'],
        'halo3_split_max': halos[3]['split_max'],

        # Kinematics
        'v_rms_global': v_rms_global,
        'v_rms_plus': v_rms_plus,
        'v_rms_hr': v_rms_hr,
        'sigma_v_halo0': sigma_v_halo0,

        # Segregation
        'corr_delta': corr_delta,
        'r_k_mean': r_k_mean,
        'n_minus_in_halos': n_minus_in_halos,
        'segregation_index': segregation_index,

        # Star formation
        'SFR': sfr,
        'M_stars_total': m_stars_total,
    }

    return result


def extract_step_from_path(path: str) -> int:
    """Extract step number from snapshot filename like snap_00100.bin"""
    basename = os.path.basename(path)
    # Expected format: snap_NNNNN.bin
    try:
        step_str = basename.replace('snap_', '').replace('.bin', '')
        return int(step_str)
    except:
        return 0


def append_to_csv(csv_path: str, row: Dict, write_header: bool = False):
    """Append a row to CSV file"""
    columns = [
        'step', 'z', 't_Gyr', 'a',
        'N_total', 'N_plus', 'N_minus', 'N_stars',
        'split_level_max', 'N_split_1', 'N_split_2', 'N_split_3plus',
        'rho_max_plus', 'rho_mean_plus', 'rho_p50', 'rho_p90', 'rho_p99', 'rho_p999',
        'halo0_x', 'halo0_y', 'halo0_z', 'halo0_rho', 'halo0_n_particles', 'halo0_split_max',
        'halo1_x', 'halo1_y', 'halo1_z', 'halo1_rho', 'halo1_n_particles', 'halo1_split_max',
        'halo2_x', 'halo2_y', 'halo2_z', 'halo2_rho', 'halo2_n_particles', 'halo2_split_max',
        'halo3_x', 'halo3_y', 'halo3_z', 'halo3_rho', 'halo3_n_particles', 'halo3_split_max',
        'v_rms_global', 'v_rms_plus', 'v_rms_hr', 'sigma_v_halo0',
        'corr_delta', 'r_k_mean', 'n_minus_in_halos', 'segregation_index',
        'SFR', 'M_stars_total'
    ]

    mode = 'w' if write_header else 'a'
    with open(csv_path, mode) as f:
        if write_header:
            f.write(','.join(columns) + '\n')

        values = []
        for col in columns:
            v = row.get(col, 0)
            if isinstance(v, float):
                values.append(f'{v:.6e}')
            else:
                values.append(str(v))
        f.write(','.join(values) + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DAEMON LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Janus Snapshot Analyzer Daemon')
    parser.add_argument('--snap-dir', type=str, required=True,
                        help='Directory containing snapshots')
    parser.add_argument('--out-csv', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--n-halos', type=int, default=4,
                        help='Number of halos to detect (default: 4)')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Grid size for segregation analysis (default: 64)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Check interval in seconds (default: 30)')
    parser.add_argument('--one-shot', action='store_true',
                        help='Analyze all existing snapshots and exit')

    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    out_csv = Path(args.out_csv)

    print(f"═══════════════════════════════════════════════════════════════════")
    print(f"  JANUS SNAPSHOT ANALYZER DAEMON")
    print(f"═══════════════════════════════════════════════════════════════════")
    print(f"  Snap dir : {snap_dir}")
    print(f"  Output   : {out_csv}")
    print(f"  N halos  : {args.n_halos}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Interval : {args.interval}s")
    print(f"═══════════════════════════════════════════════════════════════════")
    print()

    processed = set()
    first_star_z = None

    # Check if CSV exists and load already processed
    if out_csv.exists():
        with open(out_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) > 0:
                        step = int(parts[0])
                        processed.add(step)
                print(f"Loaded {len(processed)} already processed steps from CSV")

    iteration = 0
    while True:
        iteration += 1

        # Find all snapshots
        snaps = sorted(glob.glob(str(snap_dir / "snap_*.bin")))

        # Filter new snapshots
        new_snaps = []
        for snap_path in snaps:
            step = extract_step_from_path(snap_path)
            if step not in processed:
                new_snaps.append((step, snap_path))

        if new_snaps:
            print(f"[{time.strftime('%H:%M:%S')}] Found {len(new_snaps)} new snapshot(s)")

        for step, snap_path in sorted(new_snaps):
            try:
                # Wait a moment to ensure file is fully written
                time.sleep(1)

                t0 = time.time()
                header, particles = read_snapshot_v3(snap_path)
                t_read = time.time() - t0

                t0 = time.time()
                row = analyze_snapshot(header, particles,
                                       n_halos=args.n_halos,
                                       grid_size=args.grid_size)
                row['step'] = step
                t_analyze = time.time() - t0

                # Track first star
                if first_star_z is None and row['N_stars'] > 0:
                    first_star_z = row['z']
                    row['z_first_star'] = first_star_z
                elif first_star_z is not None:
                    row['z_first_star'] = first_star_z
                else:
                    row['z_first_star'] = 0

                # Write to CSV
                write_header = not out_csv.exists() or out_csv.stat().st_size == 0
                append_to_csv(str(out_csv), row, write_header=write_header)

                processed.add(step)

                print(f"  Step {step:5d} | z={row['z']:.3f} | N={row['N_total']:,} | "
                      f"ρ_max={row['rho_max_plus']:.2e} | v_rms={row['v_rms_global']:.1f} km/s | "
                      f"corr={row['corr_delta']:.3f} | "
                      f"read={t_read:.1f}s analyze={t_analyze:.1f}s")

            except Exception as e:
                print(f"  ERROR {snap_path}: {e}")
                import traceback
                traceback.print_exc()

        if args.one_shot:
            print(f"\nOne-shot mode: processed {len(processed)} snapshots. Exiting.")
            break

        # Wait before next check
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
