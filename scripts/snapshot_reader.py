#!/usr/bin/env python3
"""
snapshot_reader.py - Unified Janus snapshot reader

Supports both format versions:
- v1: 25 bytes/particle (x,y,z,sign) - sign is i8
- v2: 26 bytes/particle (x,y,z,sign,type) - type is u8

Particle types:
- 0: gas m+ (positive mass gas)
- 1: sink particle (star)
- 255: m- (negative mass)

Header format (32 bytes):
- 4 bytes: magic "JSNP"
- 4 bytes: version (u32)
- 8 bytes: n_particles (u64)
- 8 bytes: redshift z (f64)
- 8 bytes: box_size (f64)
"""

import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# Particle type constants
TYPE_GAS_PLUS = 0
TYPE_SINK_STAR = 1
TYPE_MASS_MINUS = 255


@dataclass
class Snapshot:
    """Container for snapshot data"""
    n: int
    z: float
    box_size: float
    version: int
    positions: np.ndarray  # (n, 3) float64
    signs: np.ndarray      # (n,) int8
    types: np.ndarray      # (n,) uint8 - only valid for v2


def read_snapshot(path: str) -> Snapshot:
    """
    Read a Janus binary snapshot file.

    Args:
        path: Path to snapshot file

    Returns:
        Snapshot dataclass with all particle data
    """
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic bytes: {magic}, expected b'JSNP'")

        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]

        # Read particle data
        positions = np.zeros((n, 3), dtype=np.float64)
        signs = np.zeros(n, dtype=np.int8)
        types = np.zeros(n, dtype=np.uint8)

        if version == 1:
            # v1: 25 bytes per particle (3×f64 + i8)
            for i in range(n):
                x, y, z_pos = struct.unpack('<ddd', f.read(24))
                s = struct.unpack('<b', f.read(1))[0]
                positions[i] = [x, y, z_pos]
                signs[i] = s
                # Infer type from sign for v1
                types[i] = TYPE_GAS_PLUS if s > 0 else TYPE_MASS_MINUS

        elif version == 2:
            # v2: 26 bytes per particle (3×f64 + i8 + u8)
            for i in range(n):
                x, y, z_pos = struct.unpack('<ddd', f.read(24))
                s = struct.unpack('<b', f.read(1))[0]
                t = struct.unpack('<B', f.read(1))[0]
                positions[i] = [x, y, z_pos]
                signs[i] = s
                types[i] = t
        else:
            raise ValueError(f"Unknown snapshot version: {version}")

    return Snapshot(
        n=n,
        z=z,
        box_size=box_size,
        version=version,
        positions=positions,
        signs=signs,
        types=types
    )


def read_snapshot_fast(path: str) -> Snapshot:
    """
    Fast vectorized snapshot reader using numpy fromfile.
    Much faster for large snapshots.
    """
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]

    # Define dtype based on version
    if version == 1:
        dt = np.dtype([
            ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
            ('sign', 'i1')
        ])
    elif version == 2:
        dt = np.dtype([
            ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
            ('sign', 'i1'), ('type', 'u1')
        ])
    else:
        raise ValueError(f"Unknown version: {version}")

    # Read all particles at once
    data = np.fromfile(path, dtype=dt, count=n, offset=32)

    positions = np.column_stack([data['x'], data['y'], data['z']])
    signs = data['sign']

    if version == 2:
        types = data['type']
    else:
        # Infer types from signs
        types = np.where(signs > 0, TYPE_GAS_PLUS, TYPE_MASS_MINUS).astype(np.uint8)

    return Snapshot(
        n=n,
        z=z,
        box_size=box_size,
        version=version,
        positions=positions,
        signs=signs,
        types=types
    )


def get_particle_masks(snap: Snapshot) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get boolean masks for different particle types.

    Returns:
        Tuple of (gas_plus_mask, stars_mask, mass_minus_mask)
    """
    gas_plus = snap.types == TYPE_GAS_PLUS
    stars = snap.types == TYPE_SINK_STAR
    mass_minus = snap.types == TYPE_MASS_MINUS
    return gas_plus, stars, mass_minus


def count_particles(snap: Snapshot) -> dict:
    """Count particles by type"""
    gas_plus, stars, mass_minus = get_particle_masks(snap)
    return {
        'n_gas_plus': np.sum(gas_plus),
        'n_stars': np.sum(stars),
        'n_mass_minus': np.sum(mass_minus),
        'n_total': snap.n
    }


# Test
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python snapshot_reader.py <snapshot.bin>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading {path}...")

    snap = read_snapshot_fast(path)
    counts = count_particles(snap)

    print(f"Version: {snap.version}")
    print(f"N particles: {snap.n:,}")
    print(f"Redshift z: {snap.z:.4f}")
    print(f"Box size: {snap.box_size} Mpc")
    print(f"Gas m+: {counts['n_gas_plus']:,}")
    print(f"Stars: {counts['n_stars']:,}")
    print(f"m-: {counts['n_mass_minus']:,}")
