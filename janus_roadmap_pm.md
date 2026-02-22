# Feuille de route — Projet Janus PM-FFT
# Projet Particle-Mesh f32 — Objectif 20M particules
# Projet parallèle à janus_roadmap.md (Barnes-Hut f64)
# Date : 22 février 2026

---

## CONTEXTE

Le projet Barnes-Hut f64 (`janus_roadmap.md`) montre que la ségrégation
augmente avec N : S_max(500K)=0.513, S_max(2M)=0.694 (+35%).
La longueur caractéristique des structures dépasse la boîte de 500K.

**Objectif PM :** atteindre 20M particules en sacrifiant la résolution
aux petites échelles. Gain estimé : **65×** (f32 sur RTX 3060).

---

## ARCHITECTURE CARGO WORKSPACE

Le projet PM est un **membre du workspace** `janus-sim/` :
```
janus-sim/                    ← racine du workspace (git existant)
├── Cargo.toml                ← ajout [workspace] + membres
├── src/                      ← code BH existant INCHANGÉ
└── janus-pm/                 ← membre workspace (ce projet)
    ├── Cargo.toml
    └── src/
```

**janus-pm réutilise directement janus-sim :**
```rust
use janus_sim::friedmann::{JanusParams, CosmoInterpolator};
use janus_sim::MassSign;
use janus_sim::constants;
```
Zéro copie de code. CosmoInterpolator, JanusParams, MassSign importés en live.

---

## MIGRATION CARGO WORKSPACE (ÉTAPE 0)

### Cargo.toml racine — ajouter AU DÉBUT, ne rien supprimer
```toml
[workspace]
members = [".", "janus-pm"]
resolver = "2"

```

### janus-pm/Cargo.toml
```toml
[package]
name = "janus-pm"
version = "0.1.0"
edition = "2021"

[features]
cuda = ["dep:cudarc"]

[dependencies]
janus-sim = { path = "..", features = ["cuda"] }
cudarc = { version = "0.9", features = ["cuda-12030", "cufft"], optional = true }
rand = { version = "0.8" }

[[bin]]
name = "pm_solver"
path = "src/main.rs"
required-features = ["cuda"]
```

### Build commands
```bash
# Depuis janus-sim/ (racine workspace) :
docker compose run --rm dev cargo check -p janus-pm
docker compose run --rm dev cargo run --release -p janus-pm --features cuda --bin pm_solver

# Run BH prod inchangé :
docker compose run --rm dev cargo run --release --features cuda --bin nbody_overnight -- [args]
```

---

## ARCHITECTURE PM-FFT

### Pipeline
```
Deux grilles f32 sur GPU :

  ρ+(x) → CuFFT → ρ̂+(k) → × G(k) → CuFFT⁻¹ → φ+(x) → -∇φ+ = g+(x)
  ρ-(x) → CuFFT → ρ̂-(k) → × G(k) → CuFFT⁻¹ → φ-(x) → -∇φ- = g-(x)

Force sur masse+ : F = g+ - g-   (attraction propre, répulsion croisée)
Force sur masse- : F = g- - g+
```

### Green's function discrète (PBC)
```rust
fn green(ix: i32, iy: i32, iz: i32, nx: i32, dx: f32) -> f32 {
    let kx = if ix < nx/2 { ix } else { ix-nx } as f32 * 2.0*PI/(nx as f32 * dx);
    let ky = /* idem iy */;
    let kz = /* idem iz */;
    let k2 = kx*kx + ky*ky + kz*kz;
    if k2 < 1e-10 { 0.0 } else { -1.0 / k2 }  // DC = 0 OBLIGATOIRE
}
```

### CosmoInterpolator — import direct
```rust
use janus_sim::friedmann::{JanusParams, CosmoInterpolator};

let params = JanusParams::from_eta(1.045);
let cosmo = CosmoInterpolator::new(&params, 5.0);  // z_init=5.0 (prod)
let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / total_steps as f64;

// Dans la boucle :
let (a, h) = cosmo.get_params_at_tau(cosmo.tau_start + step as f64 * dtau_cosmo);
// Hubble friction : vel += -h * vel * 0.013205 * dt
// 0.013205 = dtau_per_dt validé en production
```

### Virialization — même méthode que prod
```rust
// Référence : GpuNBodySimulation::virialize() — nbody_gpu.rs ligne 822
// PE_binding = potentiel paires MÊME signe uniquement (toujours < 0)
// alpha = sqrt(|PE_binding| / (2*KE)) = 4.57 pour η=1.045
// Seg₀ validé = 0.0024, seed = 42
// η=1.045 → n_positive = N * 0.4890, n_negative = N * 0.5110
```

### Précision numérique (schéma Gadget-4 / RAMSES)
```
Champ           Type    Raison
─────────────────────────────────────────────────────────────
positions       f64     Évite téléportation sur grandes boîtes
                        (erreur accumulation > box_size avec f32)
vitesses        f32     Précision suffisante, gain mémoire
grilles ρ, φ    f32     Là où vit le gain 65× (CuFFT f32)
forces          f32     Interpolées depuis grille f32
mass_sign       i8      +1 ou -1
─────────────────────────────────────────────────────────────
```
Justification : à L_box = 500 Mpc, f32 donne ~10 m de précision absolue.
Sur 10⁴ steps avec v ~ 1000 km/s, l'erreur cumulée peut atteindre ~1 kpc.
f64 positions élimine ce problème (précision ~10⁻¹⁰ Mpc).

### Budget mémoire (512³ + 20M particules)
```
Grilles ρ+, ρ-, φ+, φ- + buffers FFT :  ~2.0 GB
Particules 20M :
  - positions f64 (3 × 8B)              ~480 MB
  - vitesses f32  (3 × 4B)              ~240 MB
  - forces f32    (3 × 4B)              ~240 MB
  - mass_sign i8                        ~ 20 MB
─────────────────────────────────────────────────
Total janus-pm :  ~3.0 GB
Run 2M en cours : ~1.4 GB
Disponible :      ~7.6 GB ✅
```

### Moteur FFT — CuFFT FFI direct (full Rust)

**Décision PM-1b** : FFI local vers libcufft.so pour C2C 3D.

Justification :
- `cudarc 0.12` n'a pas de feature `cufft`
- `cufft_rust 0.6` ne supporte que R2C (pas C2C 3D nécessaire)
- FFI direct : `cufftPlan3d`, `cufftExecC2C`, `cufftDestroy`
- Full Rust, zéro dépendance Python, RAII pour cleanup GPU

```
Architecture full Rust :
┌─────────────────────────────────────────────────────────────────┐
│ janus-pm/src/cufft_ffi.rs                                       │
│   - FFI bindings: CufftHandle, CufftComplex, CufftResult        │
│   - extern "C" { cufftPlan3d, cufftExecC2C, cufftDestroy }      │
│   - Link: libcufft.so + libcudart.so                            │
├─────────────────────────────────────────────────────────────────┤
│ janus-pm/src/cufft.rs                                           │
│   - Safe wrapper: Cufft3dC2C avec RAII                          │
│   - Méthodes: forward(), inverse(), roundtrip()                 │
│   - Gestion mémoire GPU: cudaMalloc/cudaFree/cudaMemcpy         │
└─────────────────────────────────────────────────────────────────┘
```

Performance validée (PM-1b) :
- 256³ : 104 ms (vs 1424 ms CPU rustfft) → **14× speedup**
- 512³ : 843 ms (inclut copies host↔device)
- Erreur reconstruction : 7.4e-7 < 1e-4 ✓

---

## TÂCHES — S'ARRÊTER APRÈS CHAQUE, RAPPORT, ATTENDRE INSTRUCTION

### ÉTAPE 0 — Migration Workspace ✅
```
1. cp Cargo.toml Cargo.toml.bak
2. Ajouter [workspace] au début du Cargo.toml existant
3. cargo check --features cuda   ← prod doit compiler encore
4. mkdir -p janus-pm/{src,validation,output}
5. Créer janus-pm/Cargo.toml
6. cargo check -p janus-pm
VALIDATION : les deux crates compilent ✓
```

### ÉTAPE PM-1 — FFT Round-trip CPU ✅
```
Grille 64³ puis 256³, gaussienne → FFT → IFFT (rustfft CPU)
RÉSULTATS :
  ✓ Erreur reconstruction 3.6e-7 < 1e-4
  ✗ Temps FFT 256³ = 1424 ms > 500 ms (CPU attendu → GPU requis)
```

### ÉTAPE PM-1b — FFT Round-trip GPU ✅
```
CuFFT FFI direct (full Rust) — janus-pm/src/{cufft_ffi.rs, cufft.rs}
RÉSULTATS :
  ✓ Erreur reconstruction 7.4e-7 < 1e-4
  ✓ Temps FFT 256³ = 104 ms < 500 ms
  ✓ Speedup vs CPU : 14×
  ℹ 512³ (prod) = 843 ms (inclut copies mémoire)
```

### ÉTAPE PM-2 — Cloud-In-Cell (CIC) ✅
```
100K→500K particules, grilles 64³ et 256³
Deux grilles séparées : ρ+, ρ-
RÉSULTATS :
  ✓ Conservation masse : 1.31e-10 < 1e-6
  ✓ CIC smoothing : 29.6% du bruit Poisson (< 50%)
  ✓ Variance grille : 0.155 < Poisson (0.524)
  ✓ Temps CIC 500K→64³ : 29 ms, 100K→256³ : 48 ms
```

### ÉTAPE PM-3 — Solver Poisson + leapfrog (gravité standard) ✅
```
Pipeline : ρ → FFT → ×G(k) → IFFT → φ → gradient → g
100K particules, 64³ grille, dt=0.01, 50 steps
RÉSULTATS :
  ✓ Conservation énergie : 0.67% < 2%
  ✓ KE/KE₀ max : 84.13 < 100
  ✓ Temps/step : 66 ms

FICHIERS :
  janus-pm/src/poisson.rs    — Green G(k)=-4πG/k², gradient spectral
  janus-pm/src/integrator.rs — Leapfrog KDK, CIC interp
```

### ÉTAPE PM-4 — Physique Janus ✅ (partiel)
```
Deux grilles ρ+/ρ-, forces croisées, CosmoInterpolator, IC virialisées (α=4.57)
Softening spectral : k_s = π/(8*dx) — empêche collapse sans tuer grandes échelles

RÉSULTATS (100K particules, 256³, dt=0.005, 1000 steps) :
  ✓ KE/KE₀ max = 1.78 < 20 — énergie stable
  ✗ S(1000) = 0.0019 < 0.01 — ségrégation non visible

ANALYSE :
  PM lisse les forces courte portée par construction (Δx ≈ 0.4).
  La ségrégation Janus émerge des interactions à petite échelle
  que seul BH capture. À 100K particules, densité trop faible.

DÉCISION : PM-4 validé pour la physique (forces Janus + cosmologie).
  Ségrégation émergera à 150M particules (densité 1500× plus haute).
  Passer à PM-5.

FICHIERS :
  janus-pm/src/janus_pm.rs    — JanusPMSimulation, dual-grid solver
  janus-pm/src/snapshot.rs    — Binary snapshots + time series CSV
  janus-pm/src/poisson.rs     — Green's function avec softening k_s
```

### ÉTAPE PM-4.5 — Optimisation GPU ✅
```
Éliminer allocations répétées, pré-allouer work buffers.

RÉSULTATS :
  ✓ Temps/step : 3000 ms → 1600 ms (-47%)
  ✗ Cible 200 ms non atteinte (nécessite kernels CUDA pour Green+gradient)

OPTIMISATIONS APPLIQUÉES :
  - Pre-allocated work buffers (work_rho_k, work_gx_k, work_gy_k, work_gz_k)
  - Combined Green's function + gradient computation in single pass
  - GPU-resident FFT methods (copy_to_gpu, forward_gpu, copy_from_gpu)

BOTTLENECK RESTANT :
  - 8 copies CPU↔GPU par solve_poisson (256³ × 8B × 8 = 1GB)
  - Solution : kernels CUDA pour appliquer Green + gradient sur GPU
  - Non implémenté — acceptable pour PM-5 prototype
```

### ÉTAPE PM-5 — 150M particules ⬜ EN PRÉPARATION
```
Simulation production : 150M particules, grille 512³, ~1000 steps
Objectif : valider ségrégation Janus à grande échelle

PARAMÈTRES :
  - N = 150M particules (vs 100K en PM-4)
  - Grid = 512³
  - dt = 0.005, η = 1.045, z_init = 5.0
  - Softening : k_s = π/(8*dx)
  - Alpha = 4.57 (hardcoded from BH reference)

SNAPSHOTS :
  - Step 0 : snapshot_0.bin (full)
  - Peak S_max : snapshot_peak.bin (détection temps réel)
  - Step final : snapshot_final.bin (full)
  - Tous les 50 steps : positions f32 + subsample 1M particules

VALIDATION :
  ✓ KE/KE₀ < 20
  ✓ S(final) > 0.01 (ségrégation visible)
  ✓ Mémoire GPU < 10 GB

ESTIMATION TEMPS :
  - 1600 ms/step × 1000 steps = ~27 min
  - Avec 150M : scaling O(N) pour CIC/interpolation → ~30-40 min total
```

---

## FORMAT RAPPORT APRÈS CHAQUE ÉTAPE

```
ÉTAPE PM-X — [SUCCÈS / ÉCHEC]
Validation : [PASS/FAIL] — [valeur] vs [seuil]
Temps : [durée]
Mémoire GPU : [MiB] (run 2M BH + janus-pm)
Prêt PM-X+1 : [OUI/NON]
[Description 3-5 lignes]
[Si ÉCHEC : diagnostic et correction appliquée]
```

---

## RÈGLES ABSOLUES

```
1. Vérifier docker ps avant toute action globale.

2. Ne JAMAIS toucher janus-sim/src/ ni les bins existants.

3. Ne pas arrêter le run 2M en cours.

4. Commandes INTERDITES :
   docker stop $(docker ps -q)
   docker rm -f $(docker ps -aq)
   docker system prune

5. S'arrêter après chaque étape PM-X, rapporter, attendre instruction.

6. En cas d'erreur workspace → restaurer avec :
   cp Cargo.toml.bak Cargo.toml
```

---

## RÉFÉRENCES

- Petit, Margnat & Zejli (2024), EPJC 84:1226
- Peebles (1980) — Hubble friction eq. 5.111
- Hockney & Eastwood (1981) — Particle-Mesh methods (référence PM classique)
