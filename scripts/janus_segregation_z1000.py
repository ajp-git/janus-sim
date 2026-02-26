"""
janus_segregation_z1000.py
Simulation minimale Janus — mesure S(z) depuis z=1000

Question unique : à quel z S(z) dépasse 4.3% ?
N=10000 particules, PM brut (pas de BH), CPU seul.
"""

import numpy as np
from scipy.integrate import quad
import time
import sys

# ── Paramètres physiques ────────────────────────────────────
H0    = 76.0       # km/s/Mpc
eta   = 1.045
Op    = 1/(1+eta)  # 0.4890
Om    = eta/(1+eta)# 0.5110
G     = 4.302e-3   # pc·M_sun^{-1}·(km/s)²  → on travaille en unités Janus

N     = 10000      # total particules
N_p   = N // 2    # secteur +
N_m   = N // 2    # secteur -
L     = 100.0      # Mpc — taille de boîte comobile
z_init = 1000.0
z_end  = 0.0
S_CRIT = 0.043     # seuil critique

# ── Intégrateur cosmologique ────────────────────────────────
def H_janus(z):
    """H(z) Janus complet : Milne pour z>1062, Janus sinon"""
    if z > 1062:
        return H0 * (1+z)
    return H0 * np.sqrt(max(Op*(1+z)**3 + (1-Op), 1e-10))

def a_of_z(z): return 1.0/(1+z)
def z_of_a(a): return 1.0/a - 1

# ── Conditions initiales ────────────────────────────────────
def init_particles(seed=42):
    rng = np.random.default_rng(seed)
    
    # Positions uniformes dans la boîte
    pos_p = rng.uniform(0, L, (N_p, 3))
    pos_m = rng.uniform(0, L, (N_m, 3))
    
    # Vitesses thermiques faibles (z=1000 : univers froid)
    v_thermal = 1e-4 * H0 * L  # ~0.76 km/s
    vel_p = rng.normal(0, v_thermal, (N_p, 3))
    vel_m = rng.normal(0, v_thermal, (N_m, 3))
    
    return pos_p, pos_m, vel_p, vel_m

# ── Calcul des forces (PM simplifié) ────────────────────────
def compute_forces_direct(pos_p, pos_m, a, eps=1.0):
    """
    Forces directes N² pour N=10000 : trop lent.
    On utilise un calcul de champ moyen (centre de masse).
    
    Force sur chaque particule + :
      F = G * m * Σ_j sign(j) * (r_j - r_i) / |r_j - r_i|³
    
    Pour la ségrégation GLOBALE, on peut simplifier :
    on calcule juste le dipôle du champ.
    """
    # Centre de masse de chaque secteur
    cm_p = np.mean(pos_p, axis=0)
    cm_m = np.mean(pos_m, axis=0)
    
    # Déplacement relatif
    delta_cm = cm_p - cm_m
    r = np.linalg.norm(delta_cm)
    
    if r < eps:
        return np.zeros_like(pos_p), np.zeros_like(pos_m)
    
    # Force dipôlaire effective
    # F_+ ∝ -(cm_p - cm_m) / |cm_p - cm_m|³  (répulsion)
    # F_- ∝ +(cm_p - cm_m) / |cm_p - cm_m|³  (répulsion)
    G_eff = 1.5 * H0**2 * Op / (8 * np.pi)  # en unités Janus
    
    F_unit = delta_cm / r**3 * G_eff * (N//2)
    
    # Les particules ressentent une force vers l'opposé de leur secteur
    acc_p = -F_unit * np.ones((N_p, 1)) / a**2
    acc_m = +F_unit * np.ones((N_m, 1)) / a**2
    
    return acc_p, acc_m

# ── Mesure de ségrégation ───────────────────────────────────
def measure_S(pos_p, pos_m, L):
    """
    S = |COM+ - COM-| / (L/2)

    Centre-of-mass based segregation measure:
    - S = 0 for perfectly mixed (COMs coincide)
    - S → 1 for complete segregation (COMs at L/2 apart)

    Uses periodic boundary minimum image convention.
    """
    # Centre of mass of each sector
    cm_p = np.mean(pos_p, axis=0)
    cm_m = np.mean(pos_m, axis=0)

    # Periodic distance (minimum image convention)
    d = cm_p - cm_m
    d = d - L * np.round(d / L)
    dist = np.sqrt(np.sum(d**2))

    # Normalize by L/2 (max meaningful separation)
    S = dist / (L / 2)
    return min(S, 1.0)  # Cap at 1

# ── Intégrateur leapfrog cosmologique ──────────────────────
def run_simulation():
    print("═"*60)
    print("JANUS SEGREGATION — z=1000 → z=0")
    print(f"N={N}, L={L} Mpc, η={eta}")
    print(f"Seuil critique S_crit = {S_CRIT*100:.1f}%")
    print("═"*60)
    
    # Init
    pos_p, pos_m, vel_p, vel_m = init_particles()
    a = a_of_z(z_init)
    z = z_init
    
    # Pas de temps : dt = 0.001 en unités H0^{-1}
    dt = 0.001
    
    # Données de sortie
    results = []
    z_threshold = None
    
    step = 0
    t_start = time.time()
    
    print(f"\n  {'step':>6}  {'z':>8}  {'S':>8}  {'H²>0?':>7}")
    print("  " + "-"*35)
    
    while a < 1.0:
        z = z_of_a(a)
        Hz = H_janus(z)
        
        # Mesure S tous les 50 steps
        if step % 50 == 0:
            S = measure_S(pos_p, pos_m, L)
            H2_ok = "✓" if (Op - (1-S)*(1-Op)) > 0 else "✗"
            results.append((z, S, a))
            
            elapsed = time.time() - t_start
            print(f"  {step:>6}  {z:>8.2f}  {S:>8.4f}  {H2_ok:>7}  [{elapsed:.1f}s]")
            sys.stdout.flush()
            
            if z_threshold is None and S > S_CRIT:
                z_threshold = z
                print(f"\n  *** SEUIL S={S_CRIT*100:.1f}% ATTEINT À z = {z:.2f} ***\n")
        
        # Forces
        acc_p, acc_m = compute_forces_direct(pos_p, pos_m, a)
        
        # Leapfrog
        vel_p += acc_p * dt - 2 * (Hz/H0) * vel_p * dt  # amortissement Hubble
        vel_m += acc_m * dt - 2 * (Hz/H0) * vel_m * dt
        
        # Mise à jour positions (coordonnées comobiles)
        pos_p = (pos_p + vel_p * dt / a) % L
        pos_m = (pos_m + vel_m * dt / a) % L
        
        # Avancer a
        a += a * (Hz/H0) * dt
        step += 1
        
        if step > 50000:
            print("  MAX STEPS atteint")
            break
    
    return results, z_threshold

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    results, z_thresh = run_simulation()
    
    print("\n═"*60)
    print("RÉSULTAT")
    print("═"*60)
    
    if z_thresh is not None:
        print(f"  S > {S_CRIT*100:.1f}% atteint à z = {z_thresh:.2f}")
        if z_thresh > 2.59:
            print(f"  → H²_eff > 0 pour tout z < {z_thresh:.2f}  ✓")
            print(f"  → Zone orpheline résolue ✓")
        else:
            print(f"  → Seuil atteint trop tard (z < 2.59)")
            print(f"  → Zone orpheline NON résolue ✗")
    else:
        print(f"  Seuil S={S_CRIT*100:.1f}% jamais atteint")
        print(f"  → Ségrégation trop lente dans ce modèle")
    
    # Sauvegarder les résultats
    import json
    with open("segregation_results.json", "w") as f:
        json.dump({
            "z_threshold": z_thresh,
            "results": [(float(z), float(S), float(a)) for z,S,a in results]
        }, f, indent=2)
    print("\n  Résultats sauvegardés dans segregation_results.json")
