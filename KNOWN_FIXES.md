# KNOWN_FIXES.md — Corrections critiques janus-sim
# À lire AVANT toute modification de nbody_gpu.rs ou pm_kernels.cu
# Dernière mise à jour : 28 février 2026

---

## RÈGLE ABSOLUE

Avant de modifier un fichier source, vérifier que ces corrections
sont présentes. Ne JAMAIS les réintroduire sous leur forme erronée.

---

## nbody_gpu.rs

### [FIX-001] inv_rp3 — calcul de force gravitationnelle
```cuda
// ✅ CORRECT
double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));

// ❌ INCORRECT — instable numériquement (rsqrt approximatif)
double inv_rp3 = rsqrt(rp2) / rp2;
```

### [FIX-002] Format snapshot — interleaved, pas planaire
```
// ✅ Format réel des snapshots légers :
// Header 32 bytes : n(u64), step(u64), scale_factor(f64), segregation(f64)
// Par particule 13 bytes : x(f32), y(f32), z(f32), sign(i8) — INTERLEAVED

// ❌ Format planaire INCORRECT (ne pas réintroduire) :
// pos_x[n], pos_y[n], pos_z[n], signs[n]  ← FAUX
```

### [FIX-003] Header snapshot — 32 bytes, pas 64
```rust
// ✅ Header réel : 4 champs × 8 bytes = 32 bytes
// n(u64), step(u64), scale_factor(f64), segregation(f64)

// ❌ Header snapshot.rs (version théorique, pas utilisée en prod) : 64 bytes
// Ne pas utiliser snapshot.rs comme référence de format
```

### [FIX-004] Ordre des champs header
```rust
// ✅ CORRECT : n en premier, step en second
let (n, step, scale_factor, segregation) = struct.unpack('<QQdd', header)

// ❌ INCORRECT : step en premier (induit step=1_000_000, N=1_200)
let (step, n, ...) = ...
```

### [FIX-005] DKD intégrateur — kernel leapfrog_kick_drift
```rust
// ✅ DKD correct : Drift(dt/2) → Force → Kick(dt) → Drift(dt/2)
// Validé 2.0× speedup vs KDK, S(t) et KE/KE₀ conservés ±1%

// ❌ Ne pas revenir au KDK (2 builds d'arbre par step)
// ❌ Ne pas passer kick_dt=0.0 pour "drift pur" sans vérifier les termes cosmo
```

### [FIX-008] Positions centrées [-box/2, box/2] — OOM LinearOctree
```rust
// ✅ CORRECT : positions centrées (comme dans test_anisotropic.rs)
let half_box = box_size / 2.0;
let x0 = (ix as f64 + 0.5) * spacing - half_box;
let y0 = (iy as f64 + 0.5) * spacing - half_box;
let z0 = (iz as f64 + 0.5) * spacing - half_box;

// ❌ INCORRECT : positions dans [0, box] → OOM lors du build LinearOctree
// L'arbre octree s'attend à des positions centrées autour de l'origine.
// Des positions [0, box] causent une allocation mémoire infinie (26+ GB)
let x0 = (ix as f64 + 0.5) * spacing;  // FAUX!
```
**Symptôme :** Exit code 137 (OOM killer), même avec quelques milliers de particules.

### [FIX-008b] Conditions périodiques après déplacement Zel'dovich
```rust
// ✅ CORRECT : wrap APRÈS ajout du déplacement (amplitude_test.rs)
let mut px = x0 + displacement_x;
while px > half_box { px -= box_size; }
while px < -half_box { px += box_size; }

// ❌ INCORRECT : pas de wrap → positions hors [-L/2, L/2] si A grand
// Avec A=10% (40 Mpc) dans boîte 400 Mpc, positions jusqu'à ±240 Mpc
// → LinearOctree alloue arbre géant → OOM
positions.push(x0 + displacement_x);  // FAUX si |displacement| > box/2 - |x0|
```
**Symptôme :** OOM uniquement avec grandes amplitudes (A≥10%), pas avec A≤5%.

---

## pm_kernels.cu / gpu_simulation.rs

## pm_kernels.cu / gpu_simulation.rs

### [FIX-009] TreePM — architecture dual-grid OBLIGATOIRE
```
⚠️ CRITIQUE : La physique Janus interdit une grille PM unique.

// ❌ INCORRECT — une seule grille (PM standard)
rho[cell] += sign_i * mass;  // signe annule les contributions
phi = FFT(rho) → force uniforme  // INVALIDE : masse négative = trou, pas répulsion

// ✅ CORRECT — deux grilles séparées
rho_plus[cell]  += mass  (uniquement si sign_i > 0)
rho_minus[cell] += mass  (valeur absolue, uniquement si sign_i < 0)

FFT(rho_plus)  → phi_plus
FFT(rho_minus) → phi_minus

Force sur particule + :  F = -∇φ_plus  + ∇φ_minus  (attirée par +, repoussée par -)
Force sur particule - :  F = -∇φ_minus + ∇φ_plus   (attirée par -, repoussée par +)
```
**Symptôme si incorrect :** ségrégation nulle ou négative, comportement identique
à une simulation newtonienne pure. Voir Run PM-5 (KNOWN_FIXES résultats référence).

### [FIX-010] TreePM — artefact grille Barnes-Hut avec forces répulsives
```
Contexte : avec θ=0.7 (valeur par défaut), les forces répulsives Janus +/-
amplifient les erreurs anisotropes de l'approximation multipôle de l'arbre,
créant des filaments alignés sur les axes cartésiens (artefact grille).

Tests effectués :
  θ=0.7  → grille marquée (éliminatoire)
  θ=0.3  → grille réduite mais toujours visible
  Newton seul θ=0.7 → pas de grille (confirme que c'est l'interaction répulsive qui amplifie)

Solution : TreePM avec FFT longue portée (isotrope par construction).
Ne PAS tenter de descendre θ en dessous de 0.3 — coût 25x pour résultat insuffisant.
```
```
// ⚠️ snapshot.rs dans le repo décrit un format théorique (64 bytes header)
// Le code réel écrit un format différent (32 bytes header, interleaved)
// Toujours se fier aux mesures hexdump plutôt qu'au code snapshot.rs
```

### [FIX-007] GRID_SIZE par défaut dans pm5_resume.rs
```rust
// ✅ CORRECT pour Run B
const GRID_SIZE: usize = 512;  // 512³

// ❌ Valeur ancienne à ne pas réintroduire
const GRID_SIZE: usize = 256;  // 256³ — Run A uniquement
```

---

### [FIX-012] TreePM production-ready — configuration validée
```
Date validation : 2026-02-28
Run référence : TreePM_validation_100K (S_max=0.659 @ z=1.88)
Optimisation finale : Morton + warp-coherent (tag optim-warpcoherent-v1.0)

✅ PARAMÈTRES VALIDÉS :
  θ = 0.7              (obligatoire pour physique correcte)
  r_cut = box/16       (split BH/PM)
  PM grid = 128³       (cuFFT GPU)
  erfc splitting: BH short-range + PM k-space damping exp(-k²r_s²)
  r_s = r_cut/3        (Gaussian splitting scale)

  virial_factor:
    - 0.3 : INVALIDE (collapse prématuré, KE/KE₀ = 850)
    - 0.5 : Borderline (KE/KE₀ = 103 @ 500K, step 2000)
    - 0.8 : RECOMMANDÉ pour N > 100K (KE/KE₀ < 10 validé @ 100K)
  virial_velocity = sqrt(N/box) × virial_factor

✅ OPTIMISATIONS ACTIVÉES :
  Morton ordering      → 7.4x speedup (cache-friendly tree traversal)
  Warp-coherent kernel → 3x additional (32 threads traverse together)
  Fonction: step_treepm_gpu_morton() dans nbody_gpu_twopass.rs

✅ PERFORMANCE MESURÉE (RTX 3060, tag optim-warpcoherent-v1.0) :
  500K : 212 ms/step
  2M   : 759 ms/step
  4M   : 1626 ms/step
  85M  : ~35s/step (extrapolation O(N log N) depuis 4M)

  12000 steps @ 85M ≈ 5 jours sur RTX 3060

❌ PARAMÈTRES INVALIDÉS :
  θ = 0.5              → ségrégation trop faible (~0.40 vs 0.65)
  TreePM sans erfc     → double-comptage forces, non publiable
  PM seul (sans BH)    → ségrégation nulle (FIX-009)
  r_cut = ∞            → revient à BH pur (valide mais pas TreePM)
  stack[16]            → 6% plus lent que stack[32] (testé 2026-02-28)
  shmem top 1024 nodes → 2x plus lent (occupancy loss 44KB)

Convention coordonnées : [-L/2, +L/2] (centré)
  - ICs : (rng - 0.5) * box_size
  - drift : wrap sur [-box_half, +box_half]
  - CIC : (px + box_half) * inv_cell_size
  - Visualisation : pos += box/2 pour afficher en [0, L]
```

### [FIX-013] Limite VRAM RTX 3060 12GB — N_max = 63M particules
```
Date validation : 2026-02-28
Mesure empirique avec test_vram_limit.rs (GPU 100% libre)

⚠️ IMPORTANT: Tuer tous les processus GPU parasites avant mesure!
   nvidia-smi → sudo kill -9 <pid> pour chaque processus

MESURES RÉELLES (TreePM step complet, GPU clean) :
  N=30M →  5.77 GB (188 bytes/particle) ✅
  N=40M →  7.58 GB (190 bytes/particle) ✅
  N=50M →  9.39 GB (189 bytes/particle) ✅
  N=60M → 11.17 GB (188 bytes/particle) ✅
  N=62M → 11.55 GB (188 bytes/particle) ✅
  N=63M → 11.74 GB (188 bytes/particle) ✅ MAX
  N=64M → OOM at buffer allocation ❌
  N=65M → OOM at buffer allocation ❌

Formule empirique : VRAM(N) ≈ 0.1 GB + N × 188 bytes

Composants mesurés (N=60M) :
  Particles (pos+vel+sign)    : 2.94 GB
  Extract buffers             : 0.98 GB
  Morton sort buffers         : 0.86 GB
  BVH single-sign (61M nodes) : 3.92 GB
  cuFFT overhead              : ~0.06 GB (négligeable)
  Total step peak             : 11.17 GB

✅ N_max RTX 3060 12GB = 63M particules
✅ Recommandé production  = 60M (marge 0.8 GB)

Performance @ 60M: ~30s/step → 100h (4.2 jours) pour 12000 steps

❌ 64M+ : CUDA_ERROR_OUT_OF_MEMORY au GPU buffer allocation
❌ 85M  : impossible sur RTX 3060 (nécessite ~16 GB)

Pour 85M : nécessite GPU 16GB+ (RTX 4080, A4000)
Pour 100M+ : nécessite GPU 24GB+ (RTX 4090, A5000, A100)

Note IC génération :
  FFT Zel'dovich utilise CPU RAM (rustfft), pas VRAM
  60M : grille 391³ ≈ 0.5 GB × 3 champs = 1.5 GB CPU RAM → OK
```

---

## Résultats de référence (ne pas invalider)

### Run BH 2M (janus-sim, KDK original)
```
S_max   = 0.6940 au step 2192 (z≈1.8)
KE/KE₀  = 1.924 au pic
Runtime = 16.3h
θ       = 0.7
softening = 0.1
```

### Run PM-5 Phase 1 (janus-pm, 512³, 45M)
```
S_max   = 0.000140 au step 119 (bruit IC — pas de ségrégation PM)
KE/KE₀  = 0.349 final
Conclusion : ségrégation Janus sub-Mpc, PM insuffisant
```

### Optimisations BH validées
```
KDK baseline        : 7810 ms/step (2M)
+ DKD               : 3868 ms/step (2.0×) ✅
+ Morton CPU        : 2662 ms/step (2.9×) ✅
+ Async θ           : 3228 ms/step — REJETÉ (overhead > gain en CI mixtes)

TreePM (2026-02-28) :
  baseline          : 16989 ms/step (2M)
  + Morton GPU      : 2296 ms/step (7.4×) ✅
  + warp-coherent   : 759 ms/step (22.4×) ✅ ← PRODUCTION
```

---

## Tests de non-régression

Après toute modification de nbody_gpu.rs, valider :
```bash
# Test 500K, 50 steps
cargo run --release --features cuda --bin test_dkd -- \
  --n-particles 500000 --steps 50

# Critères PASS :
#   S(50)   dans [0.0001, 0.002]  (pas de divergence)
#   KE/KE₀  dans [0.85, 1.15]    (pas d'explosion)
#   Temps/step < 300ms pour 500K
```

---

## Physique Janus — rappels

```
Masses de même signe  → attraction  (Newton)
Masses de signes opposés → répulsion (anti-Newton)

η = 1.045 (validé Pantheon+, χ²/dof = 0.914)
Ω₊ ≈ 0.31, Ω₋ ≈ 0.69
H₀ = 70 km/s/Mpc

Ségrégation sub-Mpc confirmée (Run PM 45M, 512³, z=5→0.3)
→ méthode BH obligatoire pour capturer la ségrégation
```
