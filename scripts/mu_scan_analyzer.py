#!/usr/bin/env python3
"""
Analyse d'un run de scan μ pour le modèle Janus de Petit.

Lit un snapshot v3 (typiquement à z≈2 pour scan rapide) et calcule
les observables Janus-spécifiques :

  - corr(δ+, δ-)         : décorrélation spatiale m+/m-
  - Segregation index S  : variance ExcessVar de m-
  - r(k) à 3 échelles    : cross-correlation P(k)
  - L_J⁻/L_J⁺            : rapport longueurs de Jeans (theorique, non mesurable directement)
  - Void fraction        : fraction de volume avec δ+ < -0.5
  - Stabilité            : max(v_rms), max(ρ_max) pendant le run

Usage :
  python3 mu_scan_analyzer.py --run-dir /app/output/janus_mu_X \
                              --mu 19.0 --out results_mu_19.json

Produit un JSON + une figure PNG récapitulative.
"""
import argparse
import json
import struct
import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_snapshot_v3(path):
    """Parse snapshot_v3.bin format."""
    with open(path, 'rb') as f:
        _magic = f.read(8)
        _version = struct.unpack('<I', f.read(4))[0]
        n_total = struct.unpack('<Q', f.read(8))[0]
        n_hr = struct.unpack('<Q', f.read(8))[0]
        n_stars = struct.unpack('<Q', f.read(8))[0]
        split_max = struct.unpack('<I', f.read(4))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t_gyr = struct.unpack('<d', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        m_plus = struct.unpack('<d', f.read(8))[0]
        m_minus = struct.unpack('<d', f.read(8))[0]
        l_box = struct.unpack('<d', f.read(8))[0]
        sfr = struct.unpack('<d', f.read(8))[0]
        rho_max = struct.unpack('<d', f.read(8))[0]
        _reserved = f.read(16)

        pos = np.zeros((n_total, 3), dtype=np.float32)
        vel = np.zeros((n_total, 3), dtype=np.float32)
        signs = np.zeros(n_total, dtype=np.int32)
        mass = np.zeros(n_total, dtype=np.float32)
        for i in range(n_total):
            pos[i] = struct.unpack('<fff', f.read(12))
            vel[i] = struct.unpack('<fff', f.read(12))
            signs[i] = struct.unpack('<i', f.read(4))[0]
            mass[i] = struct.unpack('<f', f.read(4))[0]
            f.read(4 + 4 + 4 + 4)  # eps, split_level, is_star, flags

    return {
        'pos': pos, 'vel': vel, 'signs': signs, 'mass': mass,
        'n_total': n_total, 'n_hr': n_hr, 'z': z, 'l_box': l_box,
        'a': a, 't_gyr': t_gyr, 'rho_max': rho_max, 'split_max': split_max
    }


def compute_density_fields(snap, grid_size=64):
    """Grille 3D de δ+ et δ- avec pondération par masses (patch #1)."""
    L = snap['l_box']
    half = L / 2
    pos = snap['pos']
    signs = snap['signs']
    mass = snap['mass']

    bins = np.linspace(-half, half, grid_size + 1)

    mask_plus = signs > 0
    mask_minus = signs < 0

    pos_p = pos[mask_plus]
    pos_m = pos[mask_minus]
    mass_p = mass[mask_plus]
    mass_m = mass[mask_minus]

    # Histogrammes 3D pondérés par masses
    rho_plus, _ = np.histogramdd(pos_p, bins=[bins, bins, bins], weights=mass_p)
    rho_minus, _ = np.histogramdd(pos_m, bins=[bins, bins, bins], weights=mass_m)

    mean_p = rho_plus.mean()
    mean_m = rho_minus.mean()

    delta_plus = (rho_plus - mean_p) / mean_p if mean_p > 0 else np.zeros_like(rho_plus)
    delta_minus = (rho_minus - mean_m) / mean_m if mean_m > 0 else np.zeros_like(rho_minus)

    return delta_plus, delta_minus, rho_plus, rho_minus


def compute_corr(delta_plus, delta_minus):
    """Corr(δ+, δ-) global."""
    dp = delta_plus.flatten()
    dm = delta_minus.flatten()
    if dp.std() > 0 and dm.std() > 0:
        return float(np.corrcoef(dp, dm)[0, 1])
    return float('nan')


def compute_rk(delta_plus, delta_minus, l_box):
    """r(k) = P(δ+δ-) / sqrt(P(δ+) P(δ-)) sur 3 bandes k."""
    from numpy.fft import fftn, fftfreq

    n = delta_plus.shape[0]
    fft_p = fftn(delta_plus)
    fft_m = fftn(delta_minus)

    power_p = np.abs(fft_p)**2
    power_m = np.abs(fft_m)**2
    power_pm = np.real(fft_p * np.conj(fft_m))

    kx = fftfreq(n, d=l_box/n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # 3 bandes k en h/Mpc (approx)
    bands = [
        ('large',  0.01, 0.05),
        ('mid',    0.05, 0.20),
        ('small',  0.20, 1.00),
    ]
    rk = {}
    for name, k_lo, k_hi in bands:
        mask = (K >= k_lo) & (K < k_hi)
        if mask.sum() == 0:
            rk[name] = float('nan')
            continue
        p_p = power_p[mask].mean()
        p_m = power_m[mask].mean()
        p_pm = power_pm[mask].mean()
        denom = np.sqrt(p_p * p_m)
        if denom > 0:
            rk[name] = float(p_pm / denom)
        else:
            rk[name] = float('nan')
    return rk


def compute_void_fraction(delta_plus, threshold=-0.5):
    """Fraction de cellules avec δ+ < threshold (zones vides en m+)."""
    return float((delta_plus < threshold).mean())


def compute_segregation_index(delta_plus, delta_minus):
    """Variance excédentaire négative : plus |δ-| est grand,
    plus m- est concentré en halos denses (ségrégation)."""
    var_minus = float(delta_minus.var())
    var_plus = float(delta_plus.var())
    return {'var_plus': var_plus, 'var_minus': var_minus,
            'ratio': var_minus / var_plus if var_plus > 0 else float('nan')}


def compute_jeans_ratio(rho_plus, rho_minus, c_plus=1.0, c_minus=10.0,
                        G_plus=1.0, G_minus=1.0):
    """L_J ∝ c / sqrt(G ρ). Rapport L_J-/L_J+ = (c-/c+) × sqrt(ρ+ G+ / ρ- G-).

    Avec relations de jauge Petit : c- = 10 c+, G- ≈ G+ (première approximation),
    le rapport mesuré en simulation dépend uniquement des densités moyennes.
    """
    mean_p = float(rho_plus.mean())
    mean_m = float(rho_minus.mean())
    if mean_p <= 0 or mean_m <= 0:
        return float('nan')
    # L_J⁻/L_J⁺ = (c⁻/c⁺) × √(ρ⁺ G⁺ / ρ⁻ G⁻)
    # En supposant G⁻ ≈ G⁺ (cf paper 2014/2018) :
    ratio = (c_minus/c_plus) * np.sqrt(mean_p / mean_m)
    return float(ratio)


def compute_rms_vel(snap):
    """v_rms = sqrt(mean |v|^2)."""
    vel = snap['vel']
    v2 = (vel**2).sum(axis=1)
    return float(np.sqrt(v2.mean()))


def compute_petit_score(results):
    """Score composite : plus bas = meilleur accord avec la physique Janus.

    Cible Petit (paper 2018 eq.10) : a⁻/a⁺ ≈ 1/100
      → si on prend L_J ≈ a, alors L_J⁻/L_J⁺ ≈ 1/100
      → le rapport doit être proche de 0.01 (ou 100 selon sens pris)
    """
    score = 0.0
    components = {}

    # 1. Corrélation δ+, δ- → le plus négatif ou proche de 0 possible
    corr = results['corr_delta']
    components['corr_penalty'] = abs(corr + 0.07)  # cible ~ -0.07 comme run μ=19 antérieur
    score += components['corr_penalty']

    # 2. Segregation (variance m- élevée relative à m+)
    ratio_var = results['segregation']['ratio']
    if not np.isnan(ratio_var):
        components['seg_bonus'] = -np.log10(max(ratio_var, 0.1))  # favor high ratio
        score += components['seg_bonus']

    # 3. r(k) doit être négatif
    rk_mid = results['rk']['mid']
    if not np.isnan(rk_mid):
        components['rk_penalty'] = max(0, rk_mid)  # pénalise si positif
        score += components['rk_penalty']

    # 4. Stabilité : v/c < 0.1
    v_rms = results['v_rms_km_s']
    c_kms = 299792
    v_ratio = v_rms / c_kms
    components['stability_penalty'] = max(0, (v_ratio - 0.05) * 10)  # pénalise > 0.05c
    score += components['stability_penalty']

    return float(score), components


def analyze(run_dir, mu, out_json=None, out_png=None):
    run_dir = Path(run_dir)
    snap_dir = run_dir / 'snapshots'
    ts_path = run_dir / 'time_series.csv'

    # Dernier snapshot
    snaps = sorted(snap_dir.glob('snap_*.bin'))
    if not snaps:
        print(f"ERROR: pas de snapshots dans {snap_dir}", file=sys.stderr)
        return None
    snap = load_snapshot_v3(snaps[-1])

    # Lecture stabilité depuis time_series
    stability = {}
    if ts_path.exists():
        import csv
        with open(ts_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            v_rms_max = max(float(r['v_rms']) for r in rows)
            rho_max_run = max(float(r['rho_max']) for r in rows)
            stability = {
                'v_rms_max': v_rms_max,
                'rho_max_max': rho_max_run,
                'n_steps': len(rows),
                'z_final': float(rows[-1]['z']),
            }

    # Computations
    print(f"Analyzing run μ={mu}, z={snap['z']:.3f}, N={snap['n_total']:,}")

    delta_p, delta_m, rho_p, rho_m = compute_density_fields(snap, grid_size=64)
    corr = compute_corr(delta_p, delta_m)
    rk = compute_rk(delta_p, delta_m, snap['l_box'])
    void_frac = compute_void_fraction(delta_p, threshold=-0.5)
    seg = compute_segregation_index(delta_p, delta_m)
    jeans = compute_jeans_ratio(rho_p, rho_m)
    v_rms = compute_rms_vel(snap)

    n_plus = int((snap['signs'] > 0).sum())
    n_minus = int((snap['signs'] < 0).sum())

    results = {
        'mu': float(mu),
        'z_snap': float(snap['z']),
        'n_total': int(snap['n_total']),
        'n_plus': n_plus,
        'n_minus': n_minus,
        'frac_plus': n_plus / snap['n_total'],
        'l_box': float(snap['l_box']),
        'corr_delta': corr,
        'rk': rk,
        'void_fraction': void_frac,
        'segregation': seg,
        'jeans_ratio_minus_plus': jeans,
        'v_rms_km_s': v_rms,
        'stability': stability,
    }

    score, components = compute_petit_score(results)
    results['score'] = score
    results['score_components'] = components

    print(f"\n  === Résultats μ={mu} ===")
    print(f"  corr(δ+, δ-)      = {corr:+.4f}")
    print(f"  r(k) large        = {rk['large']:+.4f}")
    print(f"  r(k) mid          = {rk['mid']:+.4f}")
    print(f"  r(k) small        = {rk['small']:+.4f}")
    print(f"  void fraction     = {void_frac:.3f}")
    print(f"  var(δ+)           = {seg['var_plus']:.3e}")
    print(f"  var(δ-)           = {seg['var_minus']:.3e}")
    print(f"  ratio var         = {seg['ratio']:.3f}")
    print(f"  L_J⁻/L_J⁺ (c⁻/c⁺=10)= {jeans:.3e}")
    print(f"  v_rms             = {v_rms:.1f} km/s (v/c = {v_rms/299792:.3f})")
    print(f"  SCORE             = {score:.4f} (plus bas = mieux)")

    # Figure
    if out_png:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Slice δ+
        ax = axes[0, 0]
        mid = delta_p.shape[0] // 2
        im = ax.imshow(delta_p[:, :, mid], cmap='RdBu_r', vmin=-2, vmax=2, origin='lower')
        ax.set_title(f'δ+ slice (z={snap["z"]:.2f})')
        plt.colorbar(im, ax=ax)

        # 2. Slice δ-
        ax = axes[0, 1]
        im = ax.imshow(delta_m[:, :, mid], cmap='RdBu_r', vmin=-2, vmax=2, origin='lower')
        ax.set_title('δ- slice')
        plt.colorbar(im, ax=ax)

        # 3. Corrélation scatter
        ax = axes[0, 2]
        sub = np.random.choice(delta_p.size, 2000, replace=False)
        ax.scatter(delta_p.flatten()[sub], delta_m.flatten()[sub], s=2, alpha=0.3)
        ax.set_xlabel('δ+'); ax.set_ylabel('δ-')
        ax.set_title(f'corr = {corr:+.4f}')
        ax.axhline(0, color='gray', alpha=0.3); ax.axvline(0, color='gray', alpha=0.3)

        # 4. Scatter particules m+ XY
        ax = axes[1, 0]
        half = snap['l_box']/2
        mp = snap['pos'][snap['signs']>0]
        ax.scatter(mp[:, 0], mp[:, 1], s=0.3, c='blue', alpha=0.3, rasterized=True)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_aspect('equal'); ax.set_title(f'm+ (N={n_plus:,})')

        # 5. Scatter m-
        ax = axes[1, 1]
        mm = snap['pos'][snap['signs']<0]
        ax.scatter(mm[:, 0], mm[:, 1], s=0.3, c='red', alpha=0.3, rasterized=True)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_aspect('equal'); ax.set_title(f'm- (N={n_minus:,})')

        # 6. Barres métriques
        ax = axes[1, 2]
        labels = ['corr', 'rk_large', 'rk_mid', 'rk_small', 'void_f', 'log(var_ratio)', 'score']
        values = [corr, rk['large'], rk['mid'], rk['small'], void_frac,
                  np.log10(seg['ratio']) if seg['ratio']>0 else 0, score]
        ax.barh(labels, values)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(f'Métriques μ={mu}')

        plt.suptitle(f'μ-scan analysis — μ={mu}, z={snap["z"]:.2f}, N={snap["n_total"]:,}')
        plt.tight_layout()
        plt.savefig(out_png, dpi=90, bbox_inches='tight')
        plt.close()

    if out_json:
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2, default=float)

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--mu', required=True, type=float)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-png', default=None)
    args = ap.parse_args()

    results = analyze(args.run_dir, args.mu, args.out_json, args.out_png)
    sys.exit(0 if results else 1)
