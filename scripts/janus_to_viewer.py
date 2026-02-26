#!/usr/bin/env python3
"""
janus_to_viewer.py — Convertit les snapshots Janus en JSON pour le viewer 3D

Supporte 2 formats :
  1. Ancien (PM léger) : Header 32B + données interleaved (x,y,z,sign) × N
  2. Nouveau (85M)     : Header 128B texte + pos(N×3×f32) + vel(N×3×f32) + signs(N×i8)

Usage :
  python3 janus_to_viewer.py snapshot.bin -o snap.json
  python3 janus_to_viewer.py snapshot.bin -n 500000 -o snap.json  # subsample pour 85M
"""

import argparse, json, struct, sys, os, re
import numpy as np

def detect_format(path):
    """Détecte le format du fichier snapshot"""
    with open(path, 'rb') as f:
        header = f.read(128)

    # Nouveau format : commence par "step="
    if header[:5] == b'step=':
        return 'new'
    else:
        return 'old'


def read_snapshot_new(path, n_subsample=None, seed=42):
    """Lit le nouveau format 85M (header texte 128B + données contiguës)"""
    rng = np.random.default_rng(seed)

    with open(path, 'rb') as f:
        # Header 128 bytes texte
        header = f.read(128).decode('utf-8', errors='ignore').strip()

        # Parse header: "step=X time=X.XXX eta=X n=XXXXXXXX"
        parts = {}
        for part in header.split():
            if '=' in part:
                k, v = part.split('=', 1)
                parts[k] = v

        n = int(parts.get('n', 0))
        step = int(parts.get('step', 0))
        eta = float(parts.get('eta', 1.045))
        time_val = float(parts.get('time', 0))

        print(f"  [NEW FORMAT] step={step} | N={n:,} | eta={eta} | t={time_val:.3f}")

        # pos: N × 3 × f32
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3).copy()

        # vel: N × 3 × f32 (skip)
        f.seek(n * 3 * 4, 1)

        # signs: N × i8
        signs = np.frombuffer(f.read(n), dtype=np.int8).copy()

    # Auto-detect box size
    box_size = (pos.max() - pos.min()) * 1.05

    # Compute segregation
    pos_plus = pos[signs > 0]
    pos_minus = pos[signs <= 0]
    com_plus = pos_plus.mean(axis=0) if len(pos_plus) > 0 else np.zeros(3)
    com_minus = pos_minus.mean(axis=0) if len(pos_minus) > 0 else np.zeros(3)
    segregation = np.linalg.norm(com_plus - com_minus) / box_size

    print(f"  box={box_size:.2f} | S={segregation:.6f}")

    # Shuffle + subsample
    shuffle_idx = rng.permutation(n)
    pos = pos[shuffle_idx]
    signs = signs[shuffle_idx]

    if n_subsample and n_subsample < n:
        pos = pos[:n_subsample]
        signs = signs[:n_subsample]
        print(f"  Sous-échantillonné → {n_subsample:,}")

    n_out = len(signs)
    n_pos = int((signs > 0).sum())
    print(f"  → {n_out:,} particules ({n_pos:,} m+, {n_out-n_pos:,} m−)")

    return {
        'x': [round(float(v), 2) for v in pos[:, 0]],
        'y': [round(float(v), 2) for v in pos[:, 1]],
        'z': [round(float(v), 2) for v in pos[:, 2]],
        's': [int(v) for v in signs],
        'meta': {
            'step': step, 'time': time_val, 'eta': eta,
            'segregation': segregation, 'seg': segregation,
            'n_total': n, 'n_subsample': n_out,
            'box_size': box_size,
        }
    }


def read_snapshot_old(path, n_subsample=None, seed=42, box_size=500.0):
    """Lit l'ancien format PM (header 32B + données interleaved)"""
    rng = np.random.default_rng(seed)
    HEADER_SIZE = 32

    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)
        n, step, scale_factor, segregation = struct.unpack_from('<QQdd', header)
        z = 1.0 / scale_factor - 1.0 if scale_factor > 0 else 0.0
        print(f"  [OLD FORMAT] step={step} | N={n:,} | a={scale_factor:.4f} | z={z:.2f} | S={segregation:.6f}")

        # Format INTERLEAVED : x,y,z,sign par particule (13 bytes)
        raw = np.frombuffer(f.read(n * 13), dtype=np.uint8).reshape(n, 13)

    # Extraire x, y, z (f32) et sign (i8)
    raw_x = np.frombuffer(raw[:, 0:4].tobytes(), dtype='<f4').copy()
    raw_y = np.frombuffer(raw[:, 4:8].tobytes(), dtype='<f4').copy()
    raw_z = np.frombuffer(raw[:, 8:12].tobytes(), dtype='<f4').copy()
    signs = raw[:, 12].view(np.int8).copy()

    # Filtrer NaN et Inf
    valid = np.isfinite(raw_x) & np.isfinite(raw_y) & np.isfinite(raw_z)
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"  Filtrées: {n_invalid:,} particules NaN/Inf")
        raw_x = raw_x[valid]; raw_y = raw_y[valid]
        raw_z = raw_z[valid]; signs = signs[valid]

    # Centrer sur 0
    raw_x = raw_x - box_size / 2.0
    raw_y = raw_y - box_size / 2.0
    raw_z = raw_z - box_size / 2.0

    # Shuffle + subsample
    shuffle_idx = rng.permutation(len(raw_x))
    raw_x = raw_x[shuffle_idx]; raw_y = raw_y[shuffle_idx]
    raw_z = raw_z[shuffle_idx]; signs = signs[shuffle_idx]

    if n_subsample and n_subsample < len(raw_x):
        raw_x = raw_x[:n_subsample]; raw_y = raw_y[:n_subsample]
        raw_z = raw_z[:n_subsample]; signs = signs[:n_subsample]
        print(f"  Sous-échantillonné → {n_subsample:,}")

    n_out = len(raw_x)
    n_pos = int((signs > 0).sum())
    print(f"  → {n_out:,} particules ({n_pos:,} m+, {n_out-n_pos:,} m−)")

    return {
        'x': [round(float(v), 2) for v in raw_x],
        'y': [round(float(v), 2) for v in raw_y],
        'z': [round(float(v), 2) for v in raw_z],
        's': [int(v) for v in signs],
        'meta': {
            'step': int(step), 'scale_factor': float(scale_factor),
            'z': round(z, 4), 'segregation': float(segregation), 'seg': float(segregation),
            'n_total': int(n), 'n_subsample': n_out,
            'box_size': box_size,
        }
    }


def read_snapshot(path, n_subsample=None, seed=42, box_size=500.0):
    """Lit un snapshot en détectant automatiquement le format"""
    fmt = detect_format(path)
    if fmt == 'new':
        return read_snapshot_new(path, n_subsample, seed)
    else:
        return read_snapshot_old(path, n_subsample, seed, box_size)


def main():
    parser = argparse.ArgumentParser(description='Snapshots Janus → JSON viewer 3D')
    parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--output', default='viewer_data.json')
    parser.add_argument('-n', '--subsample', type=int, default=None,
                        help='Max particles (important pour 85M!)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--box-size', type=float, default=500.0,
                        help='Box size pour ancien format (ignoré pour nouveau)')
    args = parser.parse_args()

    files = sorted([f for f in args.files if os.path.exists(f)])
    if not files:
        print("Aucun fichier trouvé."); sys.exit(1)

    # Warning pour 85M
    total_size = sum(os.path.getsize(f) for f in files)
    if total_size > 1e9 and args.subsample is None:
        print(f"⚠ Fichiers volumineux ({total_size/1e9:.1f} GB)")
        print(f"  Utilisez -n 500000 pour sous-échantillonner")
        print()

    print(f"Traitement de {len(files)} fichier(s)...\n")
    snapshots = []
    for path in files:
        print(f"→ {os.path.basename(path)}")
        try:
            data = read_snapshot(path, n_subsample=args.subsample,
                                 seed=args.seed, box_size=args.box_size)
            if data:
                snapshots.append(data)
        except Exception as e:
            print(f"  ERREUR: {e}")
        print()

    if not snapshots:
        print("Aucun snapshot valide."); sys.exit(1)

    output = snapshots[0] if len(snapshots) == 1 else snapshots
    json_str = json.dumps(output, separators=(',', ':'))
    json_str = re.sub(r'\bNaN\b', '0.0', json_str)
    json_str = re.sub(r'\bInfinity\b', '0.0', json_str)
    json_str = re.sub(r'\b-Infinity\b', '0.0', json_str)

    with open(args.output, 'w') as f:
        f.write(json_str)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"✓ {len(snapshots)} snapshot(s) → {args.output} ({size_mb:.1f} MB)")
    if len(snapshots) > 1:
        segs = [s['meta']['segregation'] for s in snapshots]
        idx_max = segs.index(max(segs))
        print(f"  S_max = {max(segs):.6f} au snapshot {idx_max+1} (step {snapshots[idx_max]['meta']['step']})")
    print(f"\n  → Ouvrir janus_viewer.html et charger {args.output}")

if __name__ == '__main__':
    main()
