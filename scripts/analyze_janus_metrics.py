#!/usr/bin/env python3
"""
Analyse des métriques Janus séparées m+ / m- sur un snapshot.

Calcule :
  - ρ_max_plus et ρ_max_minus (densités max séparées par signe)
  - ρ_mean_plus et ρ_mean_minus
  - δ+_max, δ-_max (overdensités)
  - Corrélation(δ+, δ-) sur grille
  - r(k) à 3 bandes
  - var(δ+), var(δ-), ratio
  - Fraction de vide en m+
  - Conglomérats m- : compter les cellules avec δ_minus > 5
"""
import argparse
import struct
import numpy as np
import sys
import json
from pathlib import Path


def load_snapshot_v3(path):
    """Load v3 snapshot format (408 byte header)."""
    with open(path, 'rb') as f:
        # Header layout from snapshot_v3.rs:
        # magic: u64, version: u32, header_size: u32,
        # n_total: u64, a: f64, t_gyr: f64, l_box: f64,
        # h0: f64, mu: f64, omega_b: f64,
        # m_part_plus_base: f64, m_part_minus_base: f64,
        # eps_plus_base: f64, eps_minus_base: f64,
        # n_split_max: u32, seed_ic: u32,
        # z_init: f64, n_stars: u64, z_start_run: f64,
        # sfr: f64, rho_max: f64, run_label: [u8; 256]

        magic = struct.unpack('<Q', f.read(8))[0]
        version = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<I', f.read(4))[0]
        n_total = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t_gyr = struct.unpack('<d', f.read(8))[0]
        l_box = struct.unpack('<d', f.read(8))[0]
        h0 = struct.unpack('<d', f.read(8))[0]
        mu = struct.unpack('<d', f.read(8))[0]
        omega_b = struct.unpack('<d', f.read(8))[0]
        m_plus = struct.unpack('<d', f.read(8))[0]
        m_minus = struct.unpack('<d', f.read(8))[0]
        eps_plus = struct.unpack('<d', f.read(8))[0]
        eps_minus = struct.unpack('<d', f.read(8))[0]
        n_split_max = struct.unpack('<I', f.read(4))[0]
        seed_ic = struct.unpack('<I', f.read(4))[0]
        z_init = struct.unpack('<d', f.read(8))[0]
        n_stars = struct.unpack('<Q', f.read(8))[0]
        z_start_run = struct.unpack('<d', f.read(8))[0]
        sfr = struct.unpack('<d', f.read(8))[0]
        rho_max = struct.unpack('<d', f.read(8))[0]
        run_label = f.read(256).rstrip(b'\x00').decode('utf-8', errors='ignore')

        z = 1.0 / a - 1.0  # Calculate redshift from scale factor

        # Read particles (36 bytes each)
        particle_dtype = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('eps', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(), dtype=particle_dtype)

    return {
        'pos': particles['pos'],
        'vel': particles['vel'],
        'mass': particles['mass'],
        'sign': particles['sign'],
        'n_total': len(particles),
        'z': z,
        'a': a,
        'l_box': l_box,
        'n_stars': n_stars,
        'sfr': sfr,
        'm_plus': m_plus,
        'm_minus': m_minus,
        't_gyr': t_gyr,
        'rho_max': rho_max,
        'n_split_max': n_split_max,
        'run_label': run_label,
        'mu': mu,
        'h0': h0,
    }


def main(snap_path, grid_size=64):
    snap = load_snapshot_v3(snap_path)
    L = snap['l_box']
    pos = snap['pos']
    sign = snap['sign']
    mass = snap['mass']
    vel = snap['vel']

    # Sign convention: 1 = positive, 255 = negative (u8)
    mask_p = sign == 1
    mask_m = sign == 255

    pos_p = pos[mask_p]
    mass_p = mass[mask_p]
    vel_p = vel[mask_p]

    pos_m = pos[mask_m]
    mass_m = mass[mask_m]
    vel_m = vel[mask_m]

    # v_rms séparés
    v_rms_p = np.sqrt((vel_p**2).sum(axis=1).mean()) * 977.8  # km/s
    v_rms_m = np.sqrt((vel_m**2).sum(axis=1).mean()) * 977.8

    # Histogrammes 3D pondérés par masses
    # Positions are in [0, L] (box coordinates)
    bins = np.linspace(0, L, grid_size + 1)
    rho_p, _ = np.histogramdd(pos_p, bins=[bins]*3, weights=mass_p)
    rho_m, _ = np.histogramdd(pos_m, bins=[bins]*3, weights=mass_m)

    # Densité moyenne (par cellule, en masse totale)
    mean_p = rho_p.mean()
    mean_m = rho_m.mean()

    # Overdensités
    delta_p = (rho_p - mean_p) / mean_p if mean_p > 0 else np.zeros_like(rho_p)
    delta_m = (rho_m - mean_m) / mean_m if mean_m > 0 else np.zeros_like(rho_m)

    # Corrélation
    dp_flat = delta_p.flatten()
    dm_flat = delta_m.flatten()
    corr = float(np.corrcoef(dp_flat, dm_flat)[0, 1])

    # Variances
    var_p = float(delta_p.var())
    var_m = float(delta_m.var())

    # ρ_max / ρ_mean séparés
    cell_volume = (L / grid_size) ** 3  # Mpc³
    rho_max_p_density = float(rho_p.max() / cell_volume)  # M☉/Mpc³
    rho_max_m_density = float(rho_m.max() / cell_volume)
    rho_mean_p_density = float(mean_p / cell_volume)
    rho_mean_m_density = float(mean_m / cell_volume)

    # Overdensité max
    delta_max_p = float(delta_p.max())
    delta_max_m = float(delta_m.max())

    # Void fraction : cellules avec δ+ < -0.5
    void_frac_plus = float((delta_p < -0.5).mean())
    void_frac_minus = float((delta_m < -0.5).mean())

    # Conglomérats : fraction cellules avec δ > 5
    conglomerates_plus = float((delta_p > 5).mean())
    conglomerates_minus = float((delta_m > 5).mean())

    # r(k) à 3 bandes
    from numpy.fft import fftn, fftfreq
    fft_p = fftn(delta_p)
    fft_m = fftn(delta_m)
    power_p = np.abs(fft_p)**2
    power_m = np.abs(fft_m)**2
    power_pm = np.real(fft_p * np.conj(fft_m))

    n = grid_size
    kx = fftfreq(n, d=L/n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bandes en Mpc⁻¹
    bands = [
        ('large', 0.01, 0.05),   # ~125-630 Mpc
        ('mid', 0.05, 0.20),     # ~30-125 Mpc
        ('small', 0.20, 1.00),   # ~6-30 Mpc
    ]
    rk = {}
    for name, k_lo, k_hi in bands:
        mask = (K >= k_lo) & (K < k_hi)
        if mask.sum() > 0:
            p_p = power_p[mask].mean()
            p_m = power_m[mask].mean()
            p_pm = power_pm[mask].mean()
            denom = np.sqrt(p_p * p_m)
            rk[name] = float(p_pm / denom) if denom > 0 else float('nan')
        else:
            rk[name] = float('nan')

    # COM séparés
    com_p = pos_p.mean(axis=0)
    com_m = pos_m.mean(axis=0)
    com_sep = np.sqrt(((com_p - com_m)**2).sum())
    segregation = com_sep / L

    # Rapport
    print(f"\n{'='*70}")
    print(f"ANALYSE JANUS MÉTRIQUES SÉPARÉES")
    print(f"{'='*70}")
    print(f"Snapshot : {snap_path}")
    print(f"z = {snap['z']:.4f}, a = {snap['a']:.4f}, t = {snap['t_gyr']:.3f} Gyr")
    print(f"L = {L:.0f} Mpc, Grid = {grid_size}³")
    print(f"N_total = {snap['n_total']:,}")
    print(f"N+ = {mask_p.sum():,}, N- = {mask_m.sum():,}")
    print(f"m_plus = {snap['m_plus']:.3e} M☉, m_minus = {snap['m_minus']:.3e} M☉")
    print(f"N_stars = {snap['n_stars']}, SFR = {snap['sfr']:.3e} M☉/Gyr")
    print(f"split_max = {snap['n_split_max']}, μ = {snap['mu']}")
    print(f"{'='*70}")

    print(f"\n=== VITESSES ===")
    print(f"v_rms_plus      = {v_rms_p:.1f} km/s")
    print(f"v_rms_minus     = {v_rms_m:.1f} km/s")
    print(f"v_rms_global    = {np.sqrt((vel**2).sum(axis=1).mean()) * 977.8:.1f} km/s")

    print(f"\n=== DENSITÉS (grille {grid_size}³, cell = {L/grid_size:.1f} Mpc) ===")
    print(f"ρ_mean_plus     = {rho_mean_p_density:.3e} M☉/Mpc³")
    print(f"ρ_mean_minus    = {rho_mean_m_density:.3e} M☉/Mpc³")
    print(f"ρ_max_plus      = {rho_max_p_density:.3e} M☉/Mpc³")
    print(f"ρ_max_minus     = {rho_max_m_density:.3e} M☉/Mpc³")
    print(f"δ_max_plus      = {delta_max_p:.2f}")
    print(f"δ_max_minus     = {delta_max_m:.2f}")

    print(f"\n=== SIGNATURES JANUS ===")
    print(f"Corr(δ+, δ-)    = {corr:+.4f}  (Janus prédit < 0, ΛCDM = +1)")
    print(f"r(k) large      = {rk['large']:+.4f}  (k=0.01-0.05)")
    print(f"r(k) mid        = {rk['mid']:+.4f}  (k=0.05-0.20)")
    print(f"r(k) small      = {rk['small']:+.4f}  (k=0.20-1.00)")
    print(f"var(δ+)         = {var_p:.4f}")
    print(f"var(δ-)         = {var_m:.4f}")
    print(f"var(δ-)/var(δ+) = {var_m/var_p if var_p > 0 else float('nan'):.3f}")

    print(f"\n=== MORPHOLOGIE ===")
    print(f"Void frac m+ (δ<-0.5)     = {void_frac_plus:.3f}")
    print(f"Void frac m- (δ<-0.5)     = {void_frac_minus:.3f}")
    print(f"Conglomérats m+ (δ>5)     = {conglomerates_plus:.4f}")
    print(f"Conglomérats m- (δ>5)     = {conglomerates_minus:.4f}")
    print(f"|COM+ - COM-|             = {com_sep:.2f} Mpc")
    print(f"Segregation S             = {segregation:.4f}")

    # Diagnostic
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC")
    print(f"{'='*70}")

    issues = []
    janus_signatures = []

    if corr < -0.1:
        janus_signatures.append(f"✓ Anti-corrélation δ+/δ- ({corr:+.3f})")
    elif corr > 0.5:
        issues.append(f"✗ Co-localisation m+/m- ({corr:+.3f}) — signature ΛCDM, pas Janus")
    else:
        janus_signatures.append(f"~ Corrélation faible ({corr:+.3f}) — transition?")

    if delta_max_m > delta_max_p * 1.5:
        janus_signatures.append(f"✓ m- plus denses (δ_max- = {delta_max_m:.1f} vs δ_max+ = {delta_max_p:.1f})")
    elif max(delta_max_m, delta_max_p) > 0 and abs(delta_max_m - delta_max_p) / max(delta_max_m, delta_max_p) < 0.2:
        issues.append(f"? δ_max similaires ({delta_max_p:.1f} vs {delta_max_m:.1f})")

    var_ratio = var_m / var_p if var_p > 0 else 1
    if var_ratio > 2:
        janus_signatures.append(f"✓ m- plus contrastés (var ratio = {var_ratio:.2f})")
    elif var_ratio < 0.5:
        issues.append(f"✗ m+ plus contrastés (var ratio = {var_ratio:.2f}) — inverse attendu")

    if conglomerates_minus > conglomerates_plus * 2:
        janus_signatures.append(f"✓ Plus de conglomérats m- ({conglomerates_minus:.4f} vs {conglomerates_plus:.4f})")

    if segregation > 0.01:
        janus_signatures.append(f"✓ Ségrégation spatiale (S = {segregation:.4f})")

    if janus_signatures:
        print("Signatures Janus détectées:")
        for s in janus_signatures:
            print(f"  {s}")

    if issues:
        print("\nPoints d'attention:")
        for i in issues:
            print(f"  {i}")

    if not issues and len(janus_signatures) >= 3:
        print("\n→ VERDICT: Physique Janus fonctionne correctement")
    elif len(issues) > len(janus_signatures):
        print("\n→ VERDICT: Comportement suspect — vérifier les forces")
    else:
        print("\n→ VERDICT: Résultats mixtes — régime de transition?")

    # Sauvegarde JSON
    results = {
        'snap_path': str(snap_path),
        't_gyr': float(snap['t_gyr']),
        'z': float(snap['z']),
        'a': float(snap['a']),
        'l_box': float(L),
        'grid_size': grid_size,
        'n_plus': int(mask_p.sum()),
        'n_minus': int(mask_m.sum()),
        'v_rms_plus': float(v_rms_p),
        'v_rms_minus': float(v_rms_m),
        'rho_max_plus': rho_max_p_density,
        'rho_max_minus': rho_max_m_density,
        'rho_mean_plus': rho_mean_p_density,
        'rho_mean_minus': rho_mean_m_density,
        'delta_max_plus': delta_max_p,
        'delta_max_minus': delta_max_m,
        'corr_delta': corr,
        'rk': rk,
        'var_plus': var_p,
        'var_minus': var_m,
        'var_ratio': var_ratio,
        'void_frac_plus': void_frac_plus,
        'void_frac_minus': void_frac_minus,
        'conglomerates_plus': conglomerates_plus,
        'conglomerates_minus': conglomerates_minus,
        'com_separation': float(com_sep),
        'segregation': float(segregation),
        'n_stars': int(snap['n_stars']),
        'sfr': float(snap['sfr']),
        'split_max': int(snap['n_split_max']),
        'mu': float(snap['mu']),
    }

    out_json = str(snap_path).replace('.bin', '_janus_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON sauvegardé : {out_json}")

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Analyse métriques Janus séparées m+/m-')
    ap.add_argument('--snap', required=True, help='Chemin du snapshot')
    ap.add_argument('--grid', type=int, default=64, help='Taille grille (défaut: 64)')
    args = ap.parse_args()
    main(args.snap, args.grid)
