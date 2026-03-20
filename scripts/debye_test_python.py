#!/usr/bin/env python3
"""
debye_test_python.py
====================
Test rapide du screening Debye en Python pur — pas de GPU nécessaire.

Force répulsion m+/m− modifiée :
    F_Debye = G × |m+| × |m−| / r² × exp(-r/λ_Debye)  (répulsion)
    F_std   = -G × m × m / r²                           (attraction)

Question : avec λ_Debye ≈ 20-50 Mpc, est-ce que les filaments
persistent jusqu'à z=0 ? Réponse en 30 min sans GPU.

Usage :
    python debye_test_python.py

Durée : ~20-30 min (N=50K particules, 2000 steps)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

OUT_DIR = '/mnt/T2/janus-sim/output/debye_test'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paramètres simulation ─────────────────────────────────────────────
N        = 2_000        # particules totales
BOX      = 200.0         # Mpc (boîte plus petite → filaments = fraction plus grande)
ETA      = 1.045         # N−/N+
STEPS    = 500
DT       = 0.005         # pas de temps adimensionnel
EPS      = 1.0           # softening Mpc
Z_INIT   = 5.0
Z_FINAL  = 0.0

# Valeurs de λ_Debye à tester
LAMBDA_DEBYE_LIST = [0.0, 5.0, 20.0, 50.0, 100.0]   # 0 = Janus pur sans screening

# ── Cosmologie Janus ─────────────────────────────────────────────────

def H_janus(z, eta=ETA):
    Om = 1/(1+eta); E = (1-eta)/(1+eta)
    a  = 1/(1+z)
    ad2 = Om + 3*E*(1/a - 1)
    return np.sqrt(max(ad2, 0)/a**2) * np.sqrt(Om)**(-1)  # normalisé H(0)=1

def z_to_a(z): return 1.0 / (1.0 + z)

# Grille z → H(z) pour l'interpolation
Z_GRID = np.linspace(0, Z_INIT, 500)
H_GRID = np.array([H_janus(z) for z in Z_GRID])


def H_interp(z):
    return np.interp(z, Z_GRID, H_GRID)


# ── Conditions initiales Zel'dovich simplifiées ───────────────────────

def make_ics(N, box, eta, rng, k_min=10):
    """
    ICs simplifiées : grille + perturbations aléatoires + mass_sign.
    Pas de vraie FFT Zel'dovich — suffisant pour test qualitatif.
    """
    N_m = int(N * eta / (1 + eta))
    N_p = N - N_m

    # Positions aléatoires uniformes
    pos = rng.uniform(0, box, (N, 3))

    # Perturbations cohérentes (modes k_min = 10 comme V14)
    # On génère des fluctuations gaussiennes filtrées
    sigma_disp = box / (k_min * 10)   # dispersion ~ longueur d'onde / 10
    disp = rng.normal(0, sigma_disp, (N, 3))
    pos  = (pos + disp) % box

    # Vitesses initiales petites (régime linéaire)
    vel = rng.normal(0, 0.1, (N, 3))

    # Mass sign : alternance
    mass = np.ones(N)
    mass[:N_m] = -1.0
    rng.shuffle(mass)

    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32)


# ── Force Barnes-Hut simplifiée (O(N²) pour N=50K — ~30s/step) ───────
# Pour N=50K on utilise un calcul direct avec vectorisation numpy
# C'est O(N²) mais numpy vectorisé → acceptable pour test

def compute_forces(pos, mass, eps, box, lambda_debye=0.0):
    """
    Force gravitationnelle avec périodicité et Debye optionnel.
    Vectorisé numpy — O(N²) mais rapide pour N=50K avec chunking.
    """
    N    = len(pos)
    F    = np.zeros((N, 3), np.float64)
    eps2 = eps**2

    chunk = 500   # traiter 500 particules à la fois

    for i0 in range(0, N, chunk):
        i1  = min(i0 + chunk, N)
        pi  = pos[i0:i1][:, np.newaxis, :]   # (chunk, 1, 3)
        pj  = pos[np.newaxis, :, :]           # (1, N, 3)

        # Distance avec conditions périodiques minimales
        dr  = pj - pi                          # (chunk, N, 3)
        dr  = dr - box * np.round(dr / box)    # périodicité
        r2  = (dr**2).sum(axis=-1) + eps2      # (chunk, N)
        r   = np.sqrt(r2)

        # Force gravitationnelle standard : F ∝ 1/r²
        inv_r3 = 1.0 / (r2 * r)

        # Interaction selon les signes
        mi = mass[i0:i1][:, np.newaxis]   # (chunk, 1)
        mj = mass[np.newaxis, :]           # (1, N)

        same_sign = (mi * mj > 0)    # attraction
        diff_sign = (mi * mj < 0)    # répulsion (Janus)

        # Amplitude de la force
        force_amp = np.zeros_like(r2)
        force_amp[same_sign] = -inv_r3[same_sign]   # attraction → vers j

        if lambda_debye > 0:
            # Force répulsive avec screening Debye
            screening = np.exp(-r / lambda_debye)
            force_amp[diff_sign] = +inv_r3[diff_sign] * screening[diff_sign]
        else:
            # Janus pur : répulsion 1/r²
            force_amp[diff_sign] = +inv_r3[diff_sign]

        # Éliminer l'auto-interaction
        idx = np.arange(i0, i1)
        for k, i in enumerate(idx):
            force_amp[k, i] = 0.0

        # Accélération : F = force_amp × dr
        F[i0:i1] = (force_amp[:, :, np.newaxis] * dr).sum(axis=1)

    return F.astype(np.float32)


def segregation(pos, mass, box):
    """σ_P = écart-type du champ de polarisation sur grille 32³."""
    G = 32
    ix = np.clip((pos[:,0]/box*G).astype(int), 0, G-1)
    iy = np.clip((pos[:,1]/box*G).astype(int), 0, G-1)
    iz = np.clip((pos[:,2]/box*G).astype(int), 0, G-1)
    rho_p = np.zeros((G,G,G)); rho_m = np.zeros((G,G,G))
    np.add.at(rho_p, (iz,iy,ix), (mass > 0).astype(float))
    np.add.at(rho_m, (iz,iy,ix), (mass < 0).astype(float))
    rho_t = rho_p + rho_m + 1e-10
    P     = (rho_p - rho_m) / rho_t
    return np.std(P)


def filament_connectivity(pos, mass, box, sign=-1, G=32, threshold_pct=80):
    """
    Mesure la connectivité filamentaire : fraction de matière en dehors
    des halos compacts (cellules non-peak).

    Méthode :
      - Projeter sur grille G²
      - Identifier les cellules 'halo' (densité > seuil percentile)
      - f_filament = fraction de particules dans cellules non-halo

    Si f_filament élevée → matière diffuse entre halos → filaments
    Si f_filament faible → tout collapsed en halos → Janus pur
    """
    mask = mass * sign > 0
    p    = pos[mask]
    if len(p) < 100: return 0.0, 0.0

    # Projection 2D (XY)
    ix = np.clip((p[:,0]/box*G).astype(int), 0, G-1)
    iy = np.clip((p[:,1]/box*G).astype(int), 0, G-1)
    grid = np.zeros((G,G), np.float32)
    np.add.at(grid, (iy,ix), 1.0)

    # Seuil halo : densité > percentile 80
    threshold = np.percentile(grid[grid > 0], threshold_pct)
    halo_mask = grid > threshold

    # Fraction de particules dans cellules non-halo (matière diffuse)
    in_halo = halo_mask[iy, ix]
    f_filament = (~in_halo).mean()   # fraction hors halos = matière diffuse

    # Variance de densité (mesure clustering)
    nm    = len(p) / G**2
    delta = (grid - nm) / (nm + 1e-10)
    sigma = np.std(delta)

    return f_filament, sigma


def run_simulation(lambda_debye, seed=42, verbose=True):
    """Lance un run complet avec λ_Debye donné."""
    rng = np.random.default_rng(seed)
    pos, vel, mass = make_ics(N, BOX, ETA, rng, k_min=10)

    # Conversion z → steps
    z_vals = np.linspace(Z_INIT, Z_FINAL, STEPS + 1)
    a_vals = 1.0 / (1.0 + z_vals)

    S_history = []
    snap_steps = [0, 200, 500, 1000, 1500, 2000]

    if verbose:
        print(f"\n  λ_Debye = {lambda_debye} Mpc")
        print(f"  {'Step':>6} {'z':>6} {'S':>8} {'t (s)':>8}")
        print("  " + "─"*35)

    t_start = time.time()
    snaps   = {}

    for step in range(STEPS + 1):
        z = z_vals[step]
        a = a_vals[step]
        H = H_interp(z)

        # Enregistrer
        if step % 100 == 0 or step in snap_steps:
            S = segregation(pos, mass, BOX)
            f_fil, sig = filament_connectivity(pos, mass, BOX, sign=-1)
            S_history.append((step, z, S, f_fil))
            if verbose and step % 200 == 0:
                dt = time.time() - t_start
                print(f"  {step:>6} {z:>6.3f} {S:>8.4f} "
                      f"f_fil={f_fil:.3f}  {dt:>8.1f}s")
            if step in snap_steps:
                snaps[step] = (pos.copy(), mass.copy(), z)

        if step == STEPS: break

        # Intégration leapfrog
        F = compute_forces(pos, mass, EPS, BOX, lambda_debye)

        # Demi-step vitesse
        vel += 0.5 * DT * F

        # Friction de Hubble
        vel *= (1.0 - H * DT * 0.5)

        # Mise à jour position
        pos = (pos + DT * vel) % BOX

        # Demi-step vitesse (fin)
        F   = compute_forces(pos, mass, EPS, BOX, lambda_debye)
        vel += 0.5 * DT * F
        vel *= (1.0 - H * DT * 0.5)

    return S_history, snaps


# ── Lancer les tests ──────────────────────────────────────────────────

print("="*60)
print("TEST DEBYE — Janus N-corps Python pur")
print(f"N={N:,}  BOX={BOX} Mpc  STEPS={STEPS}  EPS={EPS} Mpc")
print("="*60)

all_results = {}

for lam in LAMBDA_DEBYE_LIST:
    t0 = time.time()
    S_hist, snaps = run_simulation(lam)
    dt = time.time() - t0
    all_results[lam] = {'S_hist': S_hist, 'snaps': snaps, 'time': dt}
    S_final = S_hist[-1][2]
    print(f"  λ={lam:5.0f} Mpc  S_final={S_final:.4f}  ({dt/60:.1f} min)")


# ── Figure ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, len(LAMBDA_DEBYE_LIST), figsize=(6*len(LAMBDA_DEBYE_LIST), 10))
fig.suptitle(f'Test Debye Janus — N={N:,} BOX={BOX} Mpc\n'
             f'Effet du screening λ_Debye sur la ségrégation et les filaments',
             fontsize=11)

colors_lam = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAMBDA_DEBYE_LIST)))

# Ligne 1 : S(z) pour chaque λ
ax_s = axes[0, 0]
for lam, c in zip(LAMBDA_DEBYE_LIST, colors_lam):
    S_hist  = all_results[lam]['S_hist']
    z_arr   = [h[1] for h in S_hist]
    S_arr   = [h[2] for h in S_hist]
    lbl     = f'λ={lam:.0f} Mpc' if lam > 0 else 'Janus pur'
    ax_s.semilogy(z_arr[::-1], S_arr, color=c, lw=2, label=lbl)

ax_s.set_xlabel('z'); ax_s.set_ylabel('S = σ_P')
ax_s.set_title('Ségrégation S(z)\ntoutes valeurs de λ', fontsize=9)
ax_s.invert_xaxis(); ax_s.legend(fontsize=8); ax_s.grid(alpha=0.3)

# Ligne 1 panels 1+ : structure finale (projection 2D)
G_proj = 64
for col, (lam, c) in enumerate(zip(LAMBDA_DEBYE_LIST, colors_lam)):
    ax = axes[0, col]
    snaps = all_results[lam]['snaps']
    snap_final = snaps.get(2000) or snaps.get(max(snaps.keys()))
    if snap_final:
        pos_f, mass_f, z_f = snap_final
        # Projection XY
        mask_m = mass_f < 0; mask_p = mass_f > 0
        ax.scatter(pos_f[mask_p,0], pos_f[mask_p,1],
                   s=0.3, c='red', alpha=0.3, rasterized=True)
        ax.scatter(pos_f[mask_m,0], pos_f[mask_m,1],
                   s=0.3, c='blue', alpha=0.3, rasterized=True)
        lbl = f'λ={lam:.0f} Mpc' if lam > 0 else 'Janus pur'
        ax.set_title(f'{lbl}\nz={z_f:.2f}  S={all_results[lam]["S_hist"][-1][2]:.3f}',
                     fontsize=9)
        ax.set_xlabel('X (Mpc)'); ax.set_ylabel('Y (Mpc)')
        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX)

# Ligne 2 : S_final vs λ + comparaison
ax2 = axes[1, 0]
lam_arr = np.array(LAMBDA_DEBYE_LIST)
S_final_arr = np.array([all_results[lam]['S_hist'][-1][2] for lam in LAMBDA_DEBYE_LIST])
ax2.plot(lam_arr, S_final_arr, 'ko-', ms=8, lw=2)
ax2.set_xlabel('λ_Debye (Mpc)'); ax2.set_ylabel('S à z=0')
ax2.set_title('Ségrégation finale vs λ_Debye\n(bas = filaments mélangés)', fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xticks(LAMBDA_DEBYE_LIST)

# Ligne 2 panels suivants : densité 2D log
for col, lam in enumerate(LAMBDA_DEBYE_LIST[1:], 1):
    ax = axes[1, col]
    snaps = all_results[lam]['snaps']
    snap_mid = snaps.get(500) or snaps.get(min(snaps.keys()))
    if snap_mid:
        pos_m2, mass_m2, z_m2 = snap_mid
        mask_m2 = mass_m2 < 0
        H, xe, ye = np.histogram2d(pos_m2[mask_m2,0], pos_m2[mask_m2,1],
                                    bins=64, range=[[0,BOX],[0,BOX]])
        ax.imshow(np.log1p(H.T), origin='lower',
                  extent=[0,BOX,0,BOX], cmap='Blues')
        lbl = f'λ={lam:.0f} Mpc'
        ax.set_title(f'm− densité log\n{lbl}  z={z_m2:.2f}', fontsize=9)
        ax.set_xlabel('X (Mpc)')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'debye_test_result.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nFigure → {out}")

# ── Rapport ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RAPPORT DEBYE TEST")
print(f"{'='*60}")
print(f"{'λ_Debye':>10} {'S_final':>10} {'f_diffuse':>12} {'Verdict':>15}")
print("─"*52)
S_janus_pur = all_results[0.0]['S_hist'][-1][2]
f_janus_pur = all_results[0.0]['S_hist'][-1][3] if len(all_results[0.0]['S_hist'][-1]) > 3 else 0
for lam in LAMBDA_DEBYE_LIST:
    row = all_results[lam]['S_hist'][-1]
    S   = row[2]
    f   = row[3] if len(row) > 3 else 0
    if lam == 0:
        verdict = "Référence"
    elif f > f_janus_pur * 1.3 and S < S_janus_pur * 0.85:
        verdict = "✅ Filaments!"
    elif f > f_janus_pur * 1.1:
        verdict = "⚠️  Partiels"
    else:
        verdict = "❌ Halos seuls"
    lbl = f'{lam:.0f} Mpc' if lam > 0 else 'Janus pur'
    print(f"  {lbl:>10} {S:>10.4f} {f:>12.4f} {verdict:>15}")

print()
print("Interprétation :")
print("  S faible + filaments visibles → screening Debye efficace")
print("  S élevé + halos compacts      → screening insuffisant")
