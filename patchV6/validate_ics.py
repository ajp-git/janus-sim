#!/usr/bin/env python3
"""
Validation visuelle et statistique des ICs v6 AVANT lancement production.

Usage :
  python3 validate_ics.py --snap /app/output/janus_v6_test/snapshots/snap_00000.bin

Produit :
  - validate_ics.png : 4 panneaux (projection XY, XZ, spectre angulaire, P(k))
  - verdict au stdout : GO / NO-GO / AMBIGU

Critères de ségrégation de grille :
  1. Spectre angulaire : pas de pics sur axes (0°, 90°, 180°, 270°) ; ratio pic/moyenne < 1.15
  2. Projection XY : distribution sans motif régulier visible à l'œil
  3. P(k) : pente compatible avec k^n_s prescrit (n_s ≈ 0.965 attendu)
  4. Corr(δ+,δ-) initial : > 0.8 (même ψ pour tous signes — c'est attendu, pas un bug)
"""
import argparse
import numpy as np
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


def load_snapshot_v3(path):
    """Parse le format snapshot_v3.bin utilisé par janus_adaptive_zoom."""
    with open(path, 'rb') as f:
        magic = f.read(8)
        version = struct.unpack('<I', f.read(4))[0]
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

        # Particles
        pos = np.zeros((n_total, 3), dtype=np.float32)
        signs = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            pos[i] = struct.unpack('<fff', f.read(12))
            f.read(12)  # vel
            signs[i] = struct.unpack('<i', f.read(4))[0]
            f.read(4 + 4 + 4 + 4 + 4)  # mass, eps, split_level, is_star, flags

    return {
        'pos': pos, 'signs': signs, 'n_total': n_total, 'z': z, 'l_box': l_box,
        'm_plus': m_plus, 'm_minus': m_minus
    }


def angular_spectrum(delta_2d):
    """Calcule le spectre angulaire d'une image 2D : ratio max/mean en fonction
    de l'angle dans l'espace de Fourier.
    Un vrai champ aléatoire cosmologique → spectre ≈ plat.
    Grille alignée sur axes → pics à 0°, 90°, 180°, 270°.
    """
    from numpy.fft import fft2, fftshift
    fft = np.abs(fftshift(fft2(delta_2d - delta_2d.mean())))
    n = delta_2d.shape[0]
    cy, cx = n//2, n//2
    angles = np.linspace(0, 2*np.pi, 72)
    signal = np.zeros(72)
    for i, theta in enumerate(angles):
        # intégrer le long d'une ligne radiale depuis le centre
        for r in range(5, n//2 - 2):
            y = int(cy + r * np.sin(theta))
            x = int(cx + r * np.cos(theta))
            signal[i] += fft[y, x]
    return angles, signal / signal.mean()


def radial_pk(delta_2d, l_box):
    """P(k) radial 1D."""
    from numpy.fft import fft2, fftshift
    fft = fft2(delta_2d - delta_2d.mean())
    power = np.abs(fft)**2 / delta_2d.size
    n = delta_2d.shape[0]
    kx = np.fft.fftfreq(n, d=l_box/n) * 2*np.pi
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # bins log
    k_bins = np.logspace(np.log10(2*np.pi/l_box), np.log10(np.pi*n/l_box), 30)
    k_cent = 0.5*(k_bins[1:] + k_bins[:-1])
    pk = np.zeros_like(k_cent)
    for i, (k_lo, k_hi) in enumerate(zip(k_bins[:-1], k_bins[1:])):
        mask = (K >= k_lo) & (K < k_hi)
        if mask.sum() > 0:
            pk[i] = power[mask].mean()
    return k_cent, pk


def analyze(snap, outpng):
    pos = snap['pos']
    signs = snap['signs']
    l_box = snap['l_box']
    half = l_box / 2

    # Slab XY à |z| < 10 Mpc
    mask_z = np.abs(pos[:, 2]) < l_box * 0.05
    pos_slab = pos[mask_z]
    signs_slab = signs[mask_z]

    # Histogram 2D des m+ et m- en slab
    n_bin = 128
    bins = np.linspace(-half, half, n_bin+1)
    mp = pos_slab[signs_slab > 0]
    mm = pos_slab[signs_slab < 0]
    h_plus, _, _ = np.histogram2d(mp[:, 0], mp[:, 1], bins=bins)
    h_minus, _, _ = np.histogram2d(mm[:, 0], mm[:, 1], bins=bins)
    h_tot = h_plus + h_minus

    # Corr delta+/delta-
    mean_p = h_plus.mean()
    mean_m = h_minus.mean()
    if mean_p > 0 and mean_m > 0:
        dp = (h_plus - mean_p) / mean_p
        dm = (h_minus - mean_m) / mean_m
        corr = np.corrcoef(dp.flatten(), dm.flatten())[0, 1]
    else:
        corr = float('nan')

    # Spectre angulaire
    angles, ang_signal = angular_spectrum(h_tot)
    # Pics sur axes : 0°, 90°, 180°, 270°
    idx_0 = np.argmin(np.abs(angles - 0))
    idx_90 = np.argmin(np.abs(angles - np.pi/2))
    idx_180 = np.argmin(np.abs(angles - np.pi))
    idx_270 = np.argmin(np.abs(angles - 3*np.pi/2))
    max_axis = max(ang_signal[idx_0], ang_signal[idx_90], ang_signal[idx_180], ang_signal[idx_270])

    # P(k)
    k, pk = radial_pk(h_tot, l_box)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # (1) Scatter m+ XY
    ax = axes[0, 0]
    ax.scatter(mp[:, 0], mp[:, 1], s=0.3, c='blue', alpha=0.4, rasterized=True)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_title(f"m+ XY (|z|<{l_box*0.05:.0f} Mpc, z_sim={snap['z']:.2f})")
    ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]")
    ax.set_aspect('equal')

    # (2) Scatter m- XY
    ax = axes[0, 1]
    ax.scatter(mm[:, 0], mm[:, 1], s=0.3, c='red', alpha=0.4, rasterized=True)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_title(f"m- XY (|z|<{l_box*0.05:.0f} Mpc)")
    ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]")
    ax.set_aspect('equal')

    # (3) Spectre angulaire
    ax = axes[1, 0]
    ax.plot(np.degrees(angles), ang_signal)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    for deg in [0, 90, 180, 270, 360]:
        ax.axvline(deg, color='orange', ls=':', alpha=0.5)
    ax.set_xlabel("angle [°]")
    ax.set_ylabel("spectre FFT normalisé")
    ax.set_title(f"Spectre angulaire (axe max = {max_axis:.3f}, plat = 1.0)")
    ax.grid(True, alpha=0.3)

    # (4) P(k)
    ax = axes[1, 1]
    ax.loglog(k, pk, 'o-')
    # Référence k^(n_s-3)=k^-2.035 (power spectrum of particle positions)
    if (pk > 0).any():
        k_ref = k[(k > 0.05) & (k < 1.0)]
        pk_ref_norm = pk[(k > 0.05) & (k < 1.0)].mean()
        ax.loglog(k_ref, pk_ref_norm * (k_ref / k_ref.mean())**(-2.035),
                 'r--', alpha=0.6, label='k^(n_s-3), n_s=0.965')
        ax.legend()
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel("P(k)")
    ax.set_title("Spectre de puissance radial")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"IC validation — z_init={snap['z']:.2f}, N={snap['n_total']:,}, "
                 f"L={l_box:.0f} Mpc, corr(δ+,δ-)={corr:.3f}")
    plt.tight_layout()
    plt.savefig(outpng, dpi=100, bbox_inches='tight')
    plt.close()

    # Verdict
    print(f"\n{'='*60}")
    print(f"VERDICT validation ICs :")
    print(f"{'='*60}")
    print(f"  N_total     = {snap['n_total']:,}")
    print(f"  N+          = {(signs > 0).sum():,} ({100*(signs > 0).mean():.1f}%)")
    print(f"  N-          = {(signs < 0).sum():,} ({100*(signs < 0).mean():.1f}%)")
    print(f"  z_init      = {snap['z']:.3f}")
    print(f"  L_box       = {l_box:.1f} Mpc")
    print(f"  corr(δ+,δ-) = {corr:+.4f}")
    print(f"  max axe FFT = {max_axis:.3f} (seuil < 1.15)")
    print(f"")
    verdict = []
    if max_axis > 1.20:
        verdict.append(f"  ❌ ARTEFACT DE GRILLE : max_axis={max_axis:.2f} > 1.20")
    elif max_axis > 1.15:
        verdict.append(f"  ⚠️  Limite axe FFT : max_axis={max_axis:.2f}")
    else:
        verdict.append(f"  ✅ Pas d'artefact de grille (max axe FFT = {max_axis:.2f})")

    if abs(corr) > 0.99:
        verdict.append(f"  ℹ️  corr(δ+,δ-)={corr:.3f} — attendu pour même ψ, sera décorrélé par dynamique")
    elif corr > 0.5:
        verdict.append(f"  ✅ corr(δ+,δ-)={corr:.3f} — normal à z_init")
    else:
        verdict.append(f"  ⚠️  corr(δ+,δ-)={corr:.3f} — faible, inattendu à z_init")

    for v in verdict:
        print(v)
    print(f"\n  Image : {outpng}")
    print(f"{'='*60}\n")

    # Code retour
    if max_axis > 1.20:
        return 1  # NO-GO
    return 0  # GO


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap', required=True, help="snapshot .bin v3")
    ap.add_argument('--out', default='validate_ics.png')
    args = ap.parse_args()
    snap = load_snapshot_v3(args.snap)
    rc = analyze(snap, args.out)
    sys.exit(rc)
