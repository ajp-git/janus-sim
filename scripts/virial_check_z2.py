#!/usr/bin/env python3
"""
Virial check at z=2 (Task 3).

Reads the binary snapshot saved by test_eds_growing_mode (when SNAP_SAVE_Z=2).
Performs FoF clustering with linking length b = 0.2 × mean inter-particle separation.
For top-10 most massive halos with N_part >= 100, computes:
  T = Σ ½·m·v_pec²
  |U| = Σ G·m·m / r_proper  (sum over intra-halo pairs, r_proper = a·r_co)
  η_virial = 2T / |U|

Expected: η_virial ∈ [0.7, 1.3] for ≥7/10 halos → ✅ couplage cosmo OK en NL.
"""
import struct
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# Code-unit constants
G_CODE = 4.499e-15  # Mpc³/(M_sun·Gyr²)
RHO_CRIT = 2.775e11  # M_sun/Mpc³

def read_snapshot(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        a = struct.unpack("<d", f.read(8))[0]
        t_gyr = struct.unpack("<d", f.read(8))[0]
        l_box = struct.unpack("<d", f.read(8))[0]
        # Per-particle: pos(3 f64) + vel(3 f64) + sign(i32) = 6*8 + 4 = 52 bytes
        dt = np.dtype([
            ('pos', '<f8', 3), ('vel', '<f8', 3), ('sign', '<i4'),
        ])
        particles = np.frombuffer(f.read(n * 52), dtype=dt)
    return n, a, t_gyr, l_box, particles

def fof_clusters(positions, b_link, box_size):
    """Friends-of-Friends clustering with periodic boundaries.
    Returns array of cluster ids per particle."""
    print(f"[FoF] Building KDTree (N={len(positions)})...")
    tree = cKDTree(positions, boxsize=box_size)
    print(f"[FoF] Querying pairs within b={b_link:.4f} Mpc...")
    # Ball query: for each particle, find neighbors within b
    pairs = tree.query_pairs(r=b_link, output_type='ndarray')
    print(f"[FoF] Found {len(pairs)} pairs")

    n = len(positions)
    if len(pairs) == 0:
        return np.arange(n)

    # Build sparse adjacency, find connected components
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    print(f"[FoF] Computing connected components...")
    n_components, labels = connected_components(graph, directed=False)
    print(f"[FoF] Found {n_components} clusters")
    return labels

def main():
    if len(sys.argv) < 2:
        path = "/app/output/eds_snapshot_save.bin"
        if not __import__("os").path.exists(path):
            path = "/mnt/T2/janus-sim/output/eds_snapshot_save.bin"
    else:
        path = sys.argv[1]
    print(f"=== Virial check on {path} ===")

    n, a, t_gyr, l_box, particles = read_snapshot(path)
    z = 1.0/a - 1.0
    print(f"[SNAP] N={n}  a={a:.4f}  z={z:.3f}  t={t_gyr:.3f} Gyr  L={l_box} Mpc")

    pos = particles['pos']      # (N, 3) comoving Mpc
    vel = particles['vel']      # (N, 3) peculiar Mpc/Gyr
    signs = particles['sign']

    # Mass per particle (in code units G·m, Mpc³/Gyr²): same as new_with_state
    omega_m_eff = 1.0  # after set_mass_factor(1/0.3)
    g_m_total = G_CODE * RHO_CRIT * omega_m_eff * l_box**3
    g_m_per_part = g_m_total / n
    # Total mass in M_sun (without G):
    m_per_part = RHO_CRIT * omega_m_eff * l_box**3 / n
    print(f"[MASS] G·m_per_part = {g_m_per_part:.4e}  m_per_part = {m_per_part:.4e} M_sun")

    # FoF: linking length b = 0.2 × mean separation
    n_per_l3 = n / l_box**3
    mean_sep = (1.0 / n_per_l3)**(1/3)
    b_link = 0.2 * mean_sep
    print(f"[FoF] mean sep = {mean_sep:.4f} Mpc  → b = {b_link:.4f} Mpc")

    # Shift positions to [0, L_box] for KDTree
    pos_shifted = pos + l_box / 2.0
    pos_shifted = np.mod(pos_shifted, l_box)

    labels = fof_clusters(pos_shifted, b_link, l_box)

    # Find sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Sort by size descending
    order = np.argsort(-counts)
    print()
    print("[HALOS] Top 20 by N_part:")
    for k in range(min(20, len(order))):
        lbl = unique_labels[order[k]]
        cnt = counts[order[k]]
        print(f"  rank {k+1}: id={lbl}  N_part={cnt}")
    print()

    # Top-10 with N_part >= 100
    selected = [(unique_labels[order[k]], counts[order[k]]) for k in range(len(order))
                if counts[order[k]] >= 100][:10]
    if len(selected) < 3:
        print(f"❌ Only {len(selected)} halos with N>=100 — increase b or use lower z")
        return 1

    print(f"[VIRIAL] Computing T and |U| for top {len(selected)} halos...")
    log_path = "/app/output/virial_z2.log"
    if not __import__("os").path.exists("/app/output"):
        log_path = "/mnt/T2/janus-sim/output/virial_z2.log"
    log = open(log_path, "w")
    log.write(f"# Virial check, snapshot a={a:.4f} z={z:.3f}\n")
    log.write(f"# halo_id  N_part  M_halo[M_sun]  T[code]  |U|[code]  eta=2T/|U|\n")

    eta_list = []
    for halo_idx, (lbl, n_h) in enumerate(selected):
        mask = labels == lbl
        idx = np.where(mask)[0]
        pos_h = pos[idx]   # comoving
        vel_h = vel[idx]   # peculiar
        m_halo = n_h * m_per_part

        # Subtract halo COM motion (only relevant peculiar motion within halo)
        v_com = vel_h.mean(axis=0)
        v_rel = vel_h - v_com

        # T = Σ ½·m·v_rel²  (in code units, m is g·m so T is in (Mpc/Gyr)²·Mpc³/Gyr²·...)
        # Use G·m units consistent with U_code = Σ G·m_i·m_j / r_proper
        # We'll compute both in code units (G·m, Mpc, Gyr).
        v2 = (v_rel ** 2).sum(axis=1)   # (Mpc/Gyr)²
        # Standard: T = ½ Σ m·v². Use m_per_part in M_sun, v² in (Mpc/Gyr)².
        # Units: M_sun·Mpc²/Gyr² (same as |U| below).
        T_code = 0.5 * m_per_part * v2.sum()

        # |U| = Σ_{i<j} G·m_i·m_j / r_proper_ij
        # r_proper = a · r_co (comoving distance × scale factor)
        # We compute pairs within the halo (need the proper periodic distances).
        # For halos that span periodic box edges, need to unwrap.
        # Use minimum-image convention since halo is small.
        from scipy.spatial.distance import pdist
        # Unwrap halo by selecting one particle as reference and applying minimum image
        ref = pos_h[0]
        deltas = pos_h - ref
        # min image
        deltas[deltas >  l_box/2] -= l_box
        deltas[deltas < -l_box/2] += l_box
        pos_h_unwrap = ref + deltas
        # Compute pairwise distances (comoving)
        d_co = pdist(pos_h_unwrap)
        d_co = np.where(d_co < 1e-3, 1e-3, d_co)  # avoid div by zero
        d_proper = a * d_co
        # |U| = Σ G·m_i·m_j / r_proper. Units: Mpc³/(M_sun·Gyr²)·M_sun²/Mpc = M_sun·Mpc²/Gyr²
        u_pairs = G_CODE * m_per_part * m_per_part / d_proper
        U_code = u_pairs.sum()

        eta = 2.0 * T_code / U_code
        eta_list.append(eta)
        line = f"{lbl:>6}  {n_h:>6}  {m_halo:.4e}  {T_code:.4e}  {U_code:.4e}  {eta:.4f}"
        print(f"  rank {halo_idx+1}: {line}")
        log.write(line + "\n")
    log.close()

    print()
    print("=== VERDICT ===")
    eta_arr = np.array(eta_list)
    in_band = (eta_arr >= 0.7) & (eta_arr <= 1.3)
    n_pass = in_band.sum()
    n_total = len(eta_arr)
    print(f"η_virial = 2T/|U|: median={np.median(eta_arr):.3f}  mean={eta_arr.mean():.3f}")
    print(f"In [0.7, 1.3]: {n_pass}/{n_total}")
    if n_pass >= max(7, int(0.7*n_total)):
        print("✅ Viriel respecté sur ≥70% des halos majeurs")
        return 0
    elif eta_arr.mean() > 2.0:
        print("❌ η systématiquement > 2 — sur-amplification persiste en NL")
        return 1
    elif eta_arr.mean() < 0.5:
        print("❌ η systématiquement < 0.5 — possible bug drag")
        return 1
    else:
        print("⚠ MARGINAL — investigate")
        return 2

if __name__ == "__main__":
    sys.exit(main())
