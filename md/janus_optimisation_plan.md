# JANUS — Plan d'optimisation par trichotomie itérative
## Document d'exécution pour Claude CLI
**Date :** 2026-03-20  
**Projet :** `/mnt/T2/janus-sim/`  
**Objectif :** Trouver (η*, λ_base*, R_smooth*) qui reproduit la structure cosmique observée  
**Modèle :** Jean-Pierre Petit — Janus bimétrique, masses négatives, répulsion croisée  

---

## ⚠️ Instructions générales Claude CLI

1. **Commence TOUJOURS par la Phase 0** (reconnaissance) — ne suppose rien sur la structure du code
2. **Chaque modification Rust doit compiler** avant de passer à la suivante : `cargo build --release`
3. **Sauvegarde l'état** avant chaque modification : `git add -A && git commit -m "avant: <description>"`
4. **Si un test échoue**, diagnostique avant de corriger — ne relance pas aveuglément
5. **Les métriques sont figées** à partir de Phase 2 — ne les modifie plus une fois validées
6. **Toute décision de trichotomie** passe par `trichotomy.py` — pas à la main

---

## Phase 0 — Reconnaissance du code (obligatoire)

### 0.1 Cartographie

```bash
# Structure générale
find /mnt/T2/janus-sim/src -name "*.rs" | head -40
find /mnt/T2/janus-sim/src -name "*.cu" -o -name "*.cuh" | head -20
cat /mnt/T2/janus-sim/Cargo.toml

# Config actuelle
find /mnt/T2/janus-sim -name "*.yaml" -o -name "*.toml" | grep -v target | head -10
cat /mnt/T2/janus-sim/config.yaml 2>/dev/null || echo "PAS DE CONFIG YAML"

# Point d'entrée
cat /mnt/T2/janus-sim/src/main.rs
```

### 0.2 Questions à répondre avant de coder

Après avoir lu le code, réponds à ces questions dans un bloc commentaire au début de chaque session :

```
- Format config actuel : YAML / TOML / args CLI / hardcodé ?
- Struct config principale : nom ? fichier ?
- Kernel CUDA force croisée : nom ? fichier ? signature ?
- Comment η (ou mass_ratio) est-il géré aujourd'hui ?
- La densité PM est-elle déjà calculée sur grille ? résolution ?
- Timestepper : leapfrog ? RK4 ? adaptatif ?
- Comment les snapshots sont-ils écrits (format, fréquence) ?
```

### 0.3 Baseline à mesurer AVANT toute modification

```bash
cd /mnt/T2/janus-sim
# Run minimal de référence (50 steps, 50k particles) pour mesurer timing
# Adapter les paramètres selon ce qui existe dans le code actuel
time cargo run --release -- --quick-test 2>&1 | tail -20
```

Objectif : connaître le temps/step sur le RTX 3060 avant toute modification.

---

## Phase 1 — Modifications du simulateur Rust/CUDA

### 1.1 Nouveaux paramètres de configuration

**Ajouter à la struct de configuration principale** (adapter le nom réel trouvé en Phase 0) :

```rust
// Dans le fichier config.rs (ou équivalent)
// NOUVEAUX CHAMPS — ajouter aux champs existants

/// Ratio masse_négative / masse_positive
/// Plage d'exploration : 0.5 – 1.5
/// Tour 1 : {0.5, 1.0, 1.5}
pub eta: f32,

/// Longueur de screening de base (Mpc)
/// λ_eff(x) = lambda_base / sqrt(rho_local(x) / rho_mean)
/// Tour 1 : fixé à 30.0
pub lambda_base_mpc: f32,

/// Rayon de lissage pour rho_local (Mpc comobiles, constant)
/// Valeur nominale : 5.0
/// Tour 3 exploration : {3.0, 5.0, 8.0}
pub r_smooth_mpc: f32,

/// Plancher pour rho_local/rho_mean (évite divergence dans les voids)
/// λ_eff est plafonné à lambda_base / sqrt(lambda_floor)
/// Valeur : 0.01 (ne pas changer sauf raison physique)
pub lambda_floor: f32,
```

**Valeurs par défaut à configurer dans le YAML** :

```yaml
# config_tour1_run_A.yaml
simulation:
  box_size_mpc: 150.0
  n_particles: 200000
  n_steps: 500
  z_start: 5.0
  z_end: 1.5
  seed: 42                    # FIGER pour tous les runs (même ICs)

physics:
  eta: 0.5                    # VARIE selon le run
  lambda_base_mpc: 30.0       # Fixe pour Tour 1
  r_smooth_mpc: 5.0           # Fixe pour Tours 1-2
  lambda_floor: 0.01          # Toujours fixe

  # VSL — Variable Speed of Light (Petit, 1988)
  # Activer dès le Tour 1 pour être fidèle au modèle complet
  # c(a) ∝ 1/√a  →  c_boost = c₀ / √a  pour z > z_vsl_cutoff
  vsl_enabled: true
  vsl_alpha: 1.7              # exposant c ∝ a^(-α/2), calibré sur ICs Janus
  vsl_z_cutoff: 10.0          # VSL actif seulement pour z > z_cutoff (ICs)

pm_grid:
  n_cells: 128                # 128³ optimal pour 200k particules

output:
  dir: "output/tour1_runA"
  snapshot_redshifts: [5.0, 3.0, 2.0, 1.5]
  metrics_every_steps: 25     # Nécessaire pour early stopping
```

Créer aussi `config_tour1_run_B.yaml` (eta: 1.0) et `config_tour1_run_C.yaml` (eta: 1.5).  
**Le seed DOIT être identique** dans les trois fichiers.

---

### 1.2 Module CUDA — Screening variable

**Créer `/mnt/T2/janus-sim/src/screening.cu`** :

```cuda
// screening.cu — Calcul du λ_effectif par particule depuis la grille PM
// Appelé APRÈS le calcul de densité PM, AVANT le kernel force croisée

#include <math.h>

/// Lisse la grille de densité PM avec un noyau gaussien 3D
/// σ_cells = r_smooth_mpc / (box_size_mpc / n_cells)
/// Utilise la convolution en espace de Fourier (appel cuFFT externe)
/// → NE PAS ré-implémenter ici si cuFFT est déjà utilisé pour le PM
///    Ajouter uniquement le passage de lissage sur la grille existante.
///
/// IMPORTANT : Cette fonction opère sur la grille de densité EXISTANTE.
/// Elle ne doit PAS modifier la grille utilisée pour les forces PM normales.
/// Créer une copie : density_smoothed[N³] séparée.

__global__ void compute_lambda_eff_grid(
    const float* __restrict__ density_smoothed,  // grille PM lissée (N³)
    float* __restrict__ lambda_eff_grid,          // sortie : λ_eff par cellule (N³)
    const float rho_mean,                         // densité moyenne globale (calculée avant)
    const float lambda_base,                      // λ_base en Mpc (depuis config)
    const float lambda_floor,                     // plancher ρ_local/ρ_mean (0.01)
    const int N                                   // nb cellules par dimension
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N * N) return;

    // Densité locale normalisée, bornée par le plancher
    const float rho_ratio = fmaxf(
        density_smoothed[idx] / fmaxf(rho_mean, 1e-30f),
        lambda_floor
    );

    // λ_eff = λ_base / sqrt(ρ_local / ρ_mean)
    // Dans les voids (ρ_ratio → λ_floor=0.01) : λ_eff = λ_base / 0.1 = 10×λ_base → longue portée
    // Dans les halos (ρ_ratio → 100) : λ_eff = λ_base / 10 = 3 Mpc → courte portée
    lambda_eff_grid[idx] = lambda_base / sqrtf(rho_ratio);
}

/// Interpolation trilinéaire : récupère λ_eff pour une particule à position (x,y,z)
/// x, y, z normalisés dans [0, 1] (coordonnées comobiles / box_size)
__device__ float get_lambda_eff(
    const float* __restrict__ lambda_eff_grid,
    const float x, const float y, const float z,
    const int N
) {
    // Coordonnées continues dans la grille
    const float gx = x * N - 0.5f;
    const float gy = y * N - 0.5f;
    const float gz = z * N - 0.5f;

    // Indices entiers (avec périodicité)
    const int ix = ((int)floorf(gx) % N + N) % N;
    const int iy = ((int)floorf(gy) % N + N) % N;
    const int iz = ((int)floorf(gz) % N + N) % N;
    const int ix1 = (ix + 1) % N;
    const int iy1 = (iy + 1) % N;
    const int iz1 = (iz + 1) % N;

    // Poids d'interpolation
    const float dx = gx - floorf(gx);
    const float dy = gy - floorf(gy);
    const float dz = gz - floorf(gz);

    // Interpolation trilinéaire
    #define IDX(a,b,c) ((a)*N*N + (b)*N + (c))
    return
        lambda_eff_grid[IDX(ix,  iy,  iz )] * (1-dx)*(1-dy)*(1-dz) +
        lambda_eff_grid[IDX(ix1, iy,  iz )] * dx*(1-dy)*(1-dz) +
        lambda_eff_grid[IDX(ix,  iy1, iz )] * (1-dx)*dy*(1-dz) +
        lambda_eff_grid[IDX(ix,  iy,  iz1)] * (1-dx)*(1-dy)*dz +
        lambda_eff_grid[IDX(ix1, iy1, iz )] * dx*dy*(1-dz) +
        lambda_eff_grid[IDX(ix1, iy,  iz1)] * dx*(1-dy)*dz +
        lambda_eff_grid[IDX(ix,  iy1, iz1)] * (1-dx)*dy*dz +
        lambda_eff_grid[IDX(ix1, iy1, iz1)] * dx*dy*dz;
    #undef IDX
}
```

---

### 1.3 Modification du kernel force croisée

**Localiser le kernel qui calcule la force entre particules de signes opposés.**  
Modifier pour ajouter le screening Yukawa avec λ_eff variable :

```cuda
// DANS le kernel force croisée existant (adapter le nom réel)
// Chercher : "cross", "repuls", "neg", "sign" dans les .cu

// AVANT modification (approximatif — adapter au code réel) :
//   float force_mag = G * m_i * m_j / (r2 + softening2);

// APRÈS modification :
// 1. Récupérer λ_eff pour les deux particules
float lambda_i = get_lambda_eff(lambda_eff_grid, x_i/box, y_i/box, z_i/box, N_grid);
float lambda_j = get_lambda_eff(lambda_eff_grid, x_j/box, y_j/box, z_j/box, N_grid);

// 2. Moyenne géométrique (symétrique, motivée physiquement)
float lambda_pair = sqrtf(lambda_i * lambda_j);

// 3. Distance
float r = sqrtf(r2 + softening2);
float inv_r = 1.0f / r;

// 4. Force Yukawa écrantée (gradient du potentiel -exp(-r/λ)/r)
//    F = G*|m+|*|m-| * exp(-r/λ) * (1/r² + 1/(r*λ)) * r_hat
//    Le signe + indique répulsion (m+ fuit m-, m- fuit m+)
float screening = expf(-r / lambda_pair);
float force_mag = G_code * fabsf(m_i) * fabsf(m_j) 
                * screening 
                * (inv_r * inv_r + inv_r / lambda_pair);

// 5. Direction : RÉPULSION (signe opposé à la gravité normale)
//    Appliquer +force_mag * r_hat sur m+, -force_mag * r_hat sur m-
```

**Note importante :** Si le code utilise déjà un paramètre `lambda_screening` fixe,
**conserver ce chemin de code** comme fallback et ajouter une branche conditionnelle :
```rust
if config.lambda_base_mpc > 0.0 {
    // Screening variable (nouveau)
} else {
    // Pas de screening (comportement original)
}
```

---

### 1.4 Métriques in-simulation

**Créer `/mnt/T2/janus-sim/src/metrics.rs`** — appelé tous les `metrics_every_steps` :

```rust
// metrics.rs — FIGÉ après validation. Ne pas modifier entre les tours.

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StepMetrics {
    pub step: u32,
    pub redshift: f32,
    pub scale_factor: f32,

    // Ségrégation
    pub s_segregation: f32,       // pureté moyenne des halos m+
    pub n_halos_plus: u32,         // nb halos m+ détectés (FOF, b=0.2)
    pub n_halos_minus: u32,        // nb halos m-

    // Structure filamentaire
    pub filament_mean_mpc: f32,   // longueur moyenne des filaments (Mpc)
    pub filament_max_mpc: f32,    // longueur du plus grand filament
    pub filament_count: u32,       // nb filaments détectés

    // Voids
    pub void_fraction: f32,       // fraction cellules PM avec ρ < 0.1 ρ_mean
    pub void_mode_mpc: f32,       // taille typique des voids (Mpc)

    // Spectre de puissance (simplifié — pente sur plage k)
    pub pk_slope: f32,            // pente log-log entre k=0.05 et k=0.5 Mpc⁻¹
    pub pk_excess_lcdm: f32,      // ratio P_janus/P_lcdm à k=0.1 Mpc⁻¹

    // Matière dans filaments (critère Euclid)
    pub fil_matter_fraction: f32, // fraction de masse dans filaments (ρ > 2ρ̄, aspect > 3:1)
                                  // objectif : > 0.15 (DESI/Euclid ~ 0.18-0.25)

    // Stabilité numérique
    pub v_rms: f32,               // vitesse rms (km/s)
    pub v_max: f32,               // vitesse maximale (km/s)
    pub e_kinetic: f32,           // énergie cinétique totale (normalisée)
    pub ke_ratio: f32,            // e_kinetic / e_kinetic_initial — diverge si runaway

impl StepMetrics {
    /// Calcule S_segregation par FOF simplifié
    /// b = 0.2 × espacement moyen inter-particules
    /// Retourne la pureté moyenne : fraction de m+ dans les halos m+
    pub fn compute_segregation(
        pos_plus: &[[f32; 3]],
        pos_minus: &[[f32; 3]],
        mean_separation_mpc: f32,
    ) -> (f32, u32, u32) {
        let b = 0.2 * mean_separation_mpc;
        // Implémentation FOF GPU-side ou CPU selon N
        // Pour 200k particules : CPU acceptable (< 1s)
        // TODO : implémenter FOF
        todo!("FOF segregation")
    }

    /// Détecte les filaments depuis la grille de densité PM
    /// Critère : cellules avec rho > 2*rho_mean, aspect ratio > 3:1
    pub fn compute_filaments(density_grid: &[f32], n_cells: usize, box_mpc: f32) -> (f32, f32, u32) {
        // TODO : labeling de composantes connexes + filtrage forme
        todo!("filament detection")
    }

    /// Score composite ∈ [0, 1]
    /// FORMULE EXACTE (figée — ne pas modifier entre les tours) :
    ///
    ///   score = 0.35 × min(S(z=0) / 0.5, 1)               ← ségrégation
    ///         + 0.30 × min(filament_mean_mpc / 10, 1)       ← filaments longs
    ///         + 0.20 × (fil_matter_fraction > 0.15 ? 1 : fil_matter_fraction/0.15)
    ///         + 0.15 × (void_fraction < 0.70 ? 1 : max(0, 1-(void_fraction-0.70)/0.25))
    ///
    /// Seuil convergence : score > 0.80 sur run 500k → lancer validation 1M
    /// ATTENTION : les poids sont calibrés après Tour 1 — ne pas changer entre tours
    pub fn composite_score(&self) -> f32 {
        // S1 : ségrégation — objectif S > 0.5, normalisé
        let s1 = (self.s_segregation / 0.5_f32).min(1.0);

        // S2 : filaments — objectif longueur moyenne > 10 Mpc
        let s2 = (self.filament_mean_mpc / 10.0_f32).min(1.0);

        // S3 : matière dans filaments — objectif > 15% (DESI~18-25%)
        //      Croissant jusqu'au seuil, pas de bonus au-delà (évite suroptimisation)
        let s3 = (self.fil_matter_fraction / 0.15_f32).min(1.0);

        // S4 : voids — objectif void_fraction < 0.70 (pas un univers fantôme)
        //      Pénalité progressive entre 0.70 et 0.95
        let s4 = if self.void_fraction < 0.70 {
            1.0
        } else {
            (1.0 - (self.void_fraction - 0.70) / 0.25_f32).max(0.0)
        };

        0.35 * s1 + 0.30 * s2 + 0.20 * s3 + 0.15 * s4
    }
}
```

**Écriture des métriques** : à chaque appel, sérialiser en JSON et appender dans :
```
output/<run_name>/metrics.jsonl
```
(une ligne JSON par step mesuré — format JSONL pour parsing incrémental)

---

### 1.5 Conditions d'arrêt anticipé

**Créer `/mnt/T2/janus-sim/src/early_stop.rs`** :

```rust
// early_stop.rs — Conditions d'arrêt. Vérifiées tous les metrics_every_steps.

pub enum StopDecision {
    Continue,
    Abort { reason: String },
    FlagWinner { score: f32 },  // continue mais marque ce run comme candidat
}

pub fn check_early_stop(
    metrics: &StepMetrics,
    history: &[StepMetrics],  // historique complet
    n_steps_total: u32,
) -> StopDecision {

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 1 — Divergence numérique (HARD STOP, priorité absolue)
    // Vérifiée à CHAQUE step (pas seulement metrics_every_steps)
    // ══════════════════════════════════════════════════════════════════

    // 1a. Runaway Janus : v_max absolu — 5000 km/s = limite physique
    //     (vitesse d'échappement cosmologique ~ 1500-2000 km/s à z=2)
    if metrics.v_max > 5_000.0 {
        return StopDecision::Abort {
            reason: format!(
                "RUNAWAY JANUS : v_max={:.0} km/s > 5000 km/s — kill immédiat (step {})",
                metrics.v_max, metrics.step
            ),
        };
    }

    // 1b. NaN ou Inf — corrompt le snapshot GPU
    if metrics.v_max.is_nan() || metrics.v_max.is_infinite()
        || metrics.e_kinetic.is_nan()
    {
        return StopDecision::Abort {
            reason: format!(
                "NaN/Inf détecté : v_max={}, KE={} (step {})",
                metrics.v_max, metrics.e_kinetic, metrics.step
            ),
        };
    }

    // 1c. Explosion d'énergie cinétique — KE_ratio = KE_now / KE_initial
    //     > 1e8 = instabilité numérique (timestep trop grand ou force divergente)
    if metrics.ke_ratio > 1.0e8 {
        return StopDecision::Abort {
            reason: format!(
                "Explosion KE : ke_ratio={:.2e} > 1e8 (step {}) — réduire dt ou softening",
                metrics.ke_ratio, metrics.step
            ),
        };
    }

    // 1d. Ratio v_max/v_rms — runaway d'une particule unique
    //     (ratio > 50 = une particule s'échappe, pas une divergence globale)
    if metrics.v_rms > 1.0 && metrics.v_max > 50.0 * metrics.v_rms {
        return StopDecision::Abort {
            reason: format!(
                "Particule runaway : v_max/v_rms={:.0} > 50 (step {})",
                metrics.v_max / metrics.v_rms, metrics.step
            ),
        };
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 2 — Effondrement structurel (step ≥ 200, z ≈ 3)
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= 200 {
        if metrics.n_halos_plus < 5 {
            return StopDecision::Abort {
                reason: format!(
                    "Effondrement : seulement {} halos m+ à z={:.1}",
                    metrics.n_halos_plus, metrics.redshift
                ),
            };
        }
        if metrics.s_segregation < 0.05 && metrics.filament_mean_mpc < 2.0 {
            return StopDecision::Abort {
                reason: format!(
                    "Pas de structure émergente à z={:.1} : S={:.3}, filaments={:.1} Mpc",
                    metrics.redshift, metrics.s_segregation, metrics.filament_mean_mpc
                ),
            };
        }
        if metrics.void_fraction > 0.95 {
            return StopDecision::Abort {
                reason: format!(
                    "Univers trop vide à z={:.1} : void_fraction={:.2}",
                    metrics.redshift, metrics.void_fraction
                ),
            };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 3 — Point de décision mi-parcours (step ≈ 250, z ≈ 2.2)
    // ══════════════════════════════════════════════════════════════════
    let midpoint = n_steps_total / 2;
    if metrics.step >= midpoint && metrics.step < midpoint + 25 {
        let score = metrics.composite_score();
        if score < 0.08 {
            return StopDecision::Abort {
                reason: format!(
                    "Score trop faible à mi-parcours : {:.3} < 0.08 (z={:.1})",
                    score, metrics.redshift
                ),
            };
        }
        if score > 0.65 {
            return StopDecision::FlagWinner { score };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 4 — Convergence du score (après step 300)
    // Si le score ne bouge plus → continuer ne sert à rien
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= 300 && history.len() >= 4 {
        let recent: Vec<f32> = history.iter().rev().take(4)
            .map(|m| m.composite_score())
            .collect();
        let score_now = recent[0];
        let all_stable = recent.windows(2).all(|w| {
            score_now > 1e-6 && (w[0] - w[1]).abs() / score_now < 0.005
        });
        if all_stable {
            return StopDecision::Abort {
                reason: format!(
                    "Convergence score sur 4 checks consécutifs : {:.4} (step {})",
                    score_now, metrics.step
                ),
            };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 5 — Score excellent en fin de run
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= (n_steps_total as f32 * 0.85) as u32 {
        let score = metrics.composite_score();
        if score > 0.80 {
            return StopDecision::FlagWinner { score };
        }
    }

    StopDecision::Continue
}
```

---

## Phase 2 — Infrastructure Python d'orchestration

**Répertoire :** `/mnt/T2/janus-sim/optim/`

### 2.1 `metrics.py` — Analyse post-run (FIGÉ)

```python
# optim/metrics.py
# FIGÉ — ne pas modifier entre les tours de trichotomie
# Lit le fichier metrics.jsonl produit par la simulation

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class RunMetrics:
    run_id: str
    eta: float
    lambda_base: float
    r_smooth: float

    # Métriques finales (dernier snapshot z=1.5)
    s_segregation: float
    filament_mean_mpc: float
    filament_max_mpc: float
    void_fraction: float
    void_mode_mpc: float
    pk_slope: float
    pk_excess_lcdm: float

    # Métriques de trajectoire
    s_at_z3: Optional[float]        # ségrégation à z=3 (step ≈ 200)
    s_at_z2: Optional[float]        # ségrégation à z=2 (step ≈ 310)
    steps_completed: int             # nb steps effectivement réalisés
    abort_reason: Optional[str]      # None si run complet

    @classmethod
    def from_jsonl(cls, run_dir: Path, config: dict) -> "RunMetrics":
        metrics_file = run_dir / "metrics.jsonl"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Pas de métriques dans {run_dir}")

        lines = [json.loads(l) for l in metrics_file.read_text().splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"Fichier métriques vide : {metrics_file}")

        # Dernier step disponible
        last = lines[-1]

        # Chercher métriques intermédiaires
        def find_at_z(target_z: float, tol: float = 0.3) -> Optional[dict]:
            return next(
                (m for m in lines if abs(m.get("redshift", 99) - target_z) < tol),
                None
            )

        at_z3 = find_at_z(3.0)
        at_z2 = find_at_z(2.0)

        # Lire abort_reason depuis log si présent
        log_file = run_dir / "run.log"
        abort_reason = None
        if log_file.exists():
            log = log_file.read_text()
            for line in log.splitlines():
                if "ABORT" in line or "Abort" in line:
                    abort_reason = line.split(":", 1)[-1].strip()
                    break

        return cls(
            run_id=run_dir.name,
            eta=config["physics"]["eta"],
            lambda_base=config["physics"]["lambda_base_mpc"],
            r_smooth=config["physics"]["r_smooth_mpc"],
            s_segregation=last.get("s_segregation", 0.0),
            filament_mean_mpc=last.get("filament_mean_mpc", 0.0),
            filament_max_mpc=last.get("filament_max_mpc", 0.0),
            void_fraction=last.get("void_fraction", 1.0),
            void_mode_mpc=last.get("void_mode_mpc", 0.0),
            pk_slope=last.get("pk_slope", 0.0),
            pk_excess_lcdm=last.get("pk_excess_lcdm", 0.0),
            s_at_z3=at_z3["s_segregation"] if at_z3 else None,
            s_at_z2=at_z2["s_segregation"] if at_z2 else None,
            steps_completed=last.get("step", 0),
            abort_reason=abort_reason,
        )
```

### 2.2 `score.py` — Score composite

```python
# optim/score.py
# ATTENTION : ne changer les poids QU'APRÈS avoir vu les résultats Tour 1

from metrics import RunMetrics
from dataclasses import dataclass

@dataclass
class ScoreBreakdown:
    s1_segregation: float   # ∈ [0,1]
    s2_filaments: float      # ∈ [0,1]
    s3_voids: float          # ∈ [0,1]
    s4_power: float          # ∈ [0,1]
    composite: float         # weighted sum

WEIGHTS = {
    "segregation": 0.35,
    "filaments":   0.30,
    "voids":       0.20,
    "power":       0.15,
}

def score(m: RunMetrics) -> ScoreBreakdown:
    """
    Score composite ∈ [0, 1]. 1 = structure cosmique parfaite.

    Formule (identique à metrics.rs — NE PAS désynchroniser) :
      score = 0.35 × min(S(z=0) / 0.5, 1)
            + 0.30 × min(filament_mean_mpc / 10, 1)
            + 0.20 × min(fil_matter_fraction / 0.15, 1)
            + 0.15 × (1 si void_fraction < 0.70, décroit sinon)
    """

    # S1 : ségrégation — objectif S > 0.5
    s1 = min(m.s_segregation / 0.5, 1.0)

    # S2 : filaments — objectif longueur moyenne > 10 Mpc
    s2 = min(m.filament_mean_mpc / 10.0, 1.0)

    # S3 : matière dans filaments — objectif > 15% (DESI/Euclid ~ 18-25%)
    s3 = min(m.fil_matter_fraction / 0.15, 1.0)

    # S4 : voids — objectif void_fraction < 0.70
    #      Pénalité progressive entre 0.70 et 0.95 (univers fantôme)
    if m.void_fraction < 0.70:
        s4 = 1.0
    else:
        s4 = max(0.0, 1.0 - (m.void_fraction - 0.70) / 0.25)

    composite = (
        WEIGHTS["segregation"] * s1
        + WEIGHTS["filaments"] * s2
        + WEIGHTS["voids"] * s3
        + WEIGHTS["power"] * s4
    )

    return ScoreBreakdown(s1, s2, s3, s4, composite)


def print_scoreboard(results: list[tuple[RunMetrics, ScoreBreakdown]]) -> None:
    """Affiche le tableau de scores pour un tour."""
    print(f"\n{'Run':<20} {'η':>6} {'λ_base':>8} {'S_seg':>6} {'Fil':>6} {'Void':>6} {'P(k)':>6} {'SCORE':>7}")
    print("-" * 75)
    for m, s in sorted(results, key=lambda x: x[1].composite, reverse=True):
        status = "✅ WINNER" if s.composite > 0.5 else ("⚠️ " if s.composite > 0.2 else "❌")
        abort = f" [{m.abort_reason[:30]}]" if m.abort_reason else ""
        print(f"{m.run_id:<20} {m.eta:>6.2f} {m.lambda_base:>8.1f} "
              f"{s.s1_segregation:>6.2f} {s.s2_filaments:>6.2f} "
              f"{s.s3_voids:>6.2f} {s.s4_power:>6.2f} "
              f"{s.composite:>7.3f} {status}{abort}")
```

### 2.3 `trichotomy.py` — Logique de zoom

```python
# optim/trichotomy.py
# Génère les paramètres du Tour N+1 depuis les résultats du Tour N

import yaml
from pathlib import Path
from metrics import RunMetrics
from score import score, ScoreBreakdown
from typing import NamedTuple

class TourParams(NamedTuple):
    eta: float
    lambda_base: float
    r_smooth: float = 5.0
    n_particles: int = 200000  # augmente selon le tour
    n_steps: int = 500


def next_tour(
    results: list[RunMetrics],
    tour_number: int,
    output_dir: Path,
) -> list[TourParams]:
    """
    Génère 3 nouveaux ensembles de paramètres par trichotomie.

    Tour 1 → Tour 2 : zoom sur η, exploration λ_base {20, 30, 40}
    Tour 2 → Tour 3 : zoom sur (η, λ_base), N_particles = 300k
    Tour 3 → Tour 4 : zoom fin + exploration R_smooth {3, 5, 8}
    """
    scored = [(m, score(m)) for m in results if m.abort_reason is None]
    if not scored:
        # Tous les runs ont avorté → agrandir l'espace
        print("⚠️ Tous les runs ont avorté ! Relance avec η plus faibles.")
        return [
            TourParams(eta=0.3, lambda_base=30.0),
            TourParams(eta=0.5, lambda_base=30.0),
            TourParams(eta=0.7, lambda_base=30.0),
        ]

    scored_sorted = sorted(scored, key=lambda x: x[1].composite, reverse=True)
    best_m, best_s = scored_sorted[0]

    print(f"\n🏆 Meilleur run : {best_m.run_id} (score={best_s.composite:.3f})")
    print(f"   η={best_m.eta}, λ_base={best_m.lambda_base}")

    if tour_number == 1:
        # ── Tour 1 → Tour 2 ───────────────────────────────────────────
        # On connaît le meilleur η parmi {0.5, 1.0, 1.5}
        # Zoom η autour du meilleur, exploration λ_base
        etas_sorted = [m.eta for m, _ in scored_sorted[:2]]
        eta_center = best_m.eta
        eta_delta = 0.25  # demi-intervalle du meilleur tiers

        new_params = [
            TourParams(
                eta=eta_center - eta_delta,
                lambda_base=20.0,
                n_particles=300_000,
                n_steps=700,
            ),
            TourParams(
                eta=eta_center,
                lambda_base=30.0,
                n_particles=300_000,
                n_steps=700,
            ),
            TourParams(
                eta=eta_center + eta_delta,
                lambda_base=40.0,
                n_particles=300_000,
                n_steps=700,
            ),
        ]

    elif tour_number == 2:
        # ── Tour 2 → Tour 3 ───────────────────────────────────────────
        # Zoom fin sur (η, λ_base)
        second_m = scored_sorted[1][0] if len(scored_sorted) > 1 else best_m
        eta_delta = abs(best_m.eta - second_m.eta) * 0.4
        lam_delta = abs(best_m.lambda_base - second_m.lambda_base) * 0.4

        eta_delta = max(eta_delta, 0.05)   # minimum 0.05
        lam_delta = max(lam_delta, 2.0)    # minimum 2 Mpc

        new_params = [
            TourParams(
                eta=best_m.eta - eta_delta,
                lambda_base=best_m.lambda_base - lam_delta,
                n_particles=500_000,
                n_steps=1000,
            ),
            TourParams(
                eta=best_m.eta,
                lambda_base=best_m.lambda_base,
                n_particles=500_000,
                n_steps=1000,
            ),
            TourParams(
                eta=best_m.eta + eta_delta,
                lambda_base=best_m.lambda_base + lam_delta,
                n_particles=500_000,
                n_steps=1000,
            ),
        ]

    else:
        # ── Tour ≥ 3 : zoom générique + exploration R_smooth ──────────
        eta_delta = 0.02
        lam_delta = 1.0
        r_values = [3.0, 5.0, 8.0] if tour_number == 3 else [best_m.r_smooth] * 3

        new_params = [
            TourParams(
                eta=best_m.eta - eta_delta,
                lambda_base=best_m.lambda_base,
                r_smooth=r_values[0],
                n_particles=500_000,
                n_steps=1200,
            ),
            TourParams(
                eta=best_m.eta,
                lambda_base=best_m.lambda_base + lam_delta,
                r_smooth=r_values[1],
                n_particles=500_000,
                n_steps=1200,
            ),
            TourParams(
                eta=best_m.eta + eta_delta,
                lambda_base=best_m.lambda_base - lam_delta,
                r_smooth=r_values[2],
                n_particles=500_000,
                n_steps=1200,
            ),
        ]

    # Générer les fichiers YAML pour le tour suivant
    next_tour_dir = output_dir / f"tour{tour_number + 1}"
    next_tour_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(new_params):
        config = _make_config(p, run_label=f"tour{tour_number+1}_run{'ABC'[i]}")
        yaml_path = next_tour_dir / f"config_run{'ABC'[i]}.yaml"
        yaml_path.write_text(yaml.dump(config, default_flow_style=False))
        print(f"   📄 Généré : {yaml_path}")

    return new_params


def _make_config(p: TourParams, run_label: str) -> dict:
    return {
        "simulation": {
            "box_size_mpc": 150.0,
            "n_particles": p.n_particles,
            "n_steps": p.n_steps,
            "z_start": 5.0,
            "z_end": 1.5,
            "seed": 42,  # TOUJOURS 42 — ICs identiques
        },
        "physics": {
            "eta": round(p.eta, 4),
            "lambda_base_mpc": round(p.lambda_base, 2),
            "r_smooth_mpc": round(p.r_smooth, 1),
            "lambda_floor": 0.01,
        },
        "pm_grid": {
            "n_cells": 128 if p.n_particles <= 300_000 else 256,
        },
        "output": {
            "dir": f"output/{run_label}",
            "snapshot_redshifts": [5.0, 3.0, 2.0, 1.5],
            "metrics_every_steps": 25,
        },
    }
```

### 2.4 `run_tour.py` — Orchestration

```python
# optim/run_tour.py
# Lance séquentiellement les 3 runs d'un tour, analyse, décide du suivant

import subprocess
import sys
import yaml
import time
from pathlib import Path
from metrics import RunMetrics
from score import score, print_scoreboard
from trichotomy import next_tour

BINARY = Path("/mnt/T2/janus-sim/target/release/janus-sim")

def run_simulation(config_path: Path) -> bool:
    """Lance un run. Retourne True si succès."""
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path("/mnt/T2/janus-sim") / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    print(f"\n🚀 Lancement : {config_path.name}")
    print(f"   η={config['physics']['eta']}, λ_base={config['physics']['lambda_base_mpc']}")
    print(f"   N={config['simulation']['n_particles']:,}, steps={config['simulation']['n_steps']}")
    print(f"   → {output_dir}")

    t0 = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run(
            [str(BINARY), "--config", str(config_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd="/mnt/T2/janus-sim",
        )
    elapsed = time.time() - t0

    status = "✅" if result.returncode == 0 else "❌"
    print(f"   {status} Terminé en {elapsed/60:.1f} min (code={result.returncode})")
    return result.returncode == 0


def analyze_tour(tour_dir: Path, tour_number: int) -> list[RunMetrics]:
    """Charge et analyse tous les runs d'un tour."""
    results = []
    sim_root = Path("/mnt/T2/janus-sim")

    for config_path in sorted(tour_dir.glob("config_run*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        run_output = sim_root / config["output"]["dir"]
        try:
            m = RunMetrics.from_jsonl(run_output, config)
            results.append(m)
        except Exception as e:
            print(f"⚠️ Erreur lecture {run_output}: {e}")

    return results


def main(tour_number: int, config_dir: Path):
    optim_dir = Path("/mnt/T2/janus-sim/optim")

    # 1. Lancer les 3 runs
    configs = sorted(config_dir.glob("config_run*.yaml"))
    if not configs:
        print(f"❌ Aucun config trouvé dans {config_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  TOUR {tour_number} — {len(configs)} runs")
    print(f"{'='*60}")

    for cfg in configs:
        run_simulation(cfg)

    # 2. Analyser
    print(f"\n📊 Analyse Tour {tour_number}...")
    results = analyze_tour(config_dir, tour_number)
    if not results:
        print("❌ Aucun résultat exploitable")
        sys.exit(1)

    scored = [(m, score(m)) for m in results]
    print_scoreboard(scored)

    # 3. Critère de convergence globale
    best_score = max(s.composite for _, s in scored)
    if best_score > 0.80:
        print(f"\n🎯 CONVERGENCE ATTEINTE — score={best_score:.3f} > 0.80")
        print("→ Passer à 1M particules pour validation finale")
        sys.exit(0)

    if tour_number >= 6:
        print(f"\n⏹ Tour {tour_number} : limite de 6 tours atteinte, arrêt")
        sys.exit(0)

    # 4. Générer Tour N+1
    print(f"\n🔄 Génération Tour {tour_number + 1}...")
    next_tour(results, tour_number, optim_dir)
    print(f"\n✅ Pour lancer le Tour {tour_number + 1} :")
    print(f"   python run_tour.py {tour_number + 1} {optim_dir}/tour{tour_number+1}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_tour.py <tour_number> <config_dir>")
        sys.exit(1)
    main(int(sys.argv[1]), Path(sys.argv[2]))
```

---

## Phase 3 — Tour 1 : paramétrage exact

### 3.1 Justification nombre de particules — N = 200 000

| Critère | 100k | **200k** | 500k | 1M |
|---|---|---|---|---|
| Espacement moyen (150 Mpc) | 3.23 Mpc | **2.56 Mpc** | 1.88 Mpc | 1.50 Mpc |
| Résolution filaments (λ=30 Mpc) | marginal | **≥ 12 particules/λ ✅** | bon | excellent |
| Grille PM optimale | 64³ | **128³** | 128³ | 256³ |
| Temps/run RTX 3060 (estimé) | ~15 min | **~25-30 min** | ~2h | ~5h |
| Tour 1 total (3 runs) | ~45 min | **~90 min** | ~6h | ~15h |

**Décision : 200k pour Tour 1.** Suffisant pour détecter ségrégation et filaments grossiers.  
Augmentation progressive : 300k (Tour 2), 500k (Tour 3), 1M (validation finale).

### 3.2 Justification nombre de steps — N_steps = 500

**Plage :** z=5 → z=1.5, soit Δln(a) = ln(0.4/0.1667) = **0.875**

| Steps | Δln(a)/step | Équivalent GADGET | Adapté Janus (λ=30 Mpc) |
|---|---|---|---|
| 200 | 0.00437 | ≈ standard | Trop grossier (λ/v_cross < 1 step) |
| **500** | **0.00175** | **conservateur** | **✅ λ_eff/v_pec ≈ 3-5 steps** |
| 1000 | 0.000875 | très fin | Inutilement coûteux |
| 1500 | 0.000583 | extrêmement fin | Tour 1 : gaspillage |

**Vérification CFL pour le screening :**
- v_pec typique à z=2 : ~200 km/s
- λ_min (halo dense, ρ=100ρ̄) = 30/√100 = 3 Mpc
- Temps traversée screening : 3 Mpc / 200 km/s ≈ 14.7 Gyr  
- Δt cosmologique à z=2, 500 steps : ≈ 0.65 Gyr  
- **Ratio : 14.7/0.65 ≈ 22 >> 1 ✅** — le screening est résolu

**500 steps est donc l'optimum Tour 1.**

### 3.3 Grille des runs Tour 1

| Run | η | λ_base | R_smooth | N | Steps | Config | Attendu |
|---|---|---|---|---|---|---|---|
| A | 0.5 | 30 Mpc | 5 Mpc | 200k | 500 | `config_run_A.yaml` | Répulsion faible, ΛCDM-like |
| B | 1.0 | 30 Mpc | 5 Mpc | 200k | 500 | `config_run_B.yaml` | Référence Janus |
| C | 1.5 | 30 Mpc | 5 Mpc | 200k | 500 | `config_run_C.yaml` | Répulsion forte, voids larges |

**Seed : 42 pour tous les trois** (ICs identiques — différences dues uniquement à (η, λ_base)).

### 3.4 Décision post-Tour 1

```
SI score(C) > score(B) > score(A) :
    → Zoom [1.2, 1.35, 1.5] + exploration λ_base {20, 30, 40}

SI score(B) > score(C) et score(B) > score(A) :
    → Zoom [0.8, 1.0, 1.2] + exploration λ_base {20, 30, 40}

SI score(A) > score(B) :
    → η trop fort partout ? Tester [0.2, 0.35, 0.5]
    OU λ_base trop court → tester λ_base {50, 75, 100} avec η=0.5

SI TOUS les scores < 0.1 :
    → Problème dans le code screening, diagnostiquer AVANT de relancer
```

---

## Phase 4 — Critère de convergence globale et arrêt de la boucle

La boucle s'arrête quand **l'une** de ces conditions est remplie :

```
1. composite_score > 0.80 sur run de 500k particules → lancer validation 1M
2. |η*(Tour N) - η*(Tour N-1)| < 0.01 ET |λ*(Tour N) - λ*(Tour N-1)| < 0.5 Mpc → convergence paramétrique
3. Tour 6 atteint sans convergence → analyser résultats, reformuler critères
4. Comparaison visuelle DESI/Euclid satisfaisante + S(z=0) > 0.4 + filaments > 8 Mpc
```

**Après convergence :** run de validation à 1M particules, 1200 steps, z=5→0 complet.  
Exporter : P(k), S(z), images de densité projetée, r(k) croisé m+/m−.

---

## Phase 5 — Comparaison qualitative DESI/Euclid (post-convergence)

**Déclenchée uniquement après** : composite_score > 0.80 sur run 500k, ou convergence paramétrique Tour ≥ 3.  
**But :** vérifier que (η*, λ*, R*) produit une structure cosmique réaliste, pas seulement un bon score interne.

### 5.1 Métriques de comparaison cibles

| Observable | DESI/Euclid DR1 (observé) | Critère Janus acceptable |
|---|---|---|
| Longueur médiane filaments | 8–15 Mpc | > 7 Mpc |
| Fraction matière en filaments | 18–25% | > 15% |
| Taille médiane des voids | 20–35 Mpc | 18–40 Mpc |
| Densité void (ρ_void / ρ̄) | 0.15–0.25 | 0.10–0.35 |
| Exposant pente P(k) à z=0 | −2.9 à −3.1 | −2.5 à −3.5 |
| Nombre halos > 10¹³ M☉ / (Gpc/h)³ | ~500–1000 | ordre de grandeur |

### 5.2 Script de comparaison visuelle

```python
# optim/compare_desi.py
# Génère 4 figures de comparaison pour validation qualitative

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Références DESI/Euclid (estimées depuis publications publiques)
DESI_REF = {
    "filament_median_mpc": 11.0,
    "fil_matter_fraction": 0.20,
    "void_median_mpc": 27.0,
    "void_density_ratio": 0.20,
    "pk_slope": -3.0,
}

def plot_comparison(run_dir: Path, metrics_final: dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Comparaison Janus vs DESI/Euclid\n"
        f"η={metrics_final['eta']:.2f}, λ_base={metrics_final['lambda_base']:.1f} Mpc",
        fontsize=13
    )

    # ── Panneau 1 : carte de densité projetée (slice z=0, épaisseur 10 Mpc) ──
    ax = axes[0, 0]
    density_slice = load_density_slice(run_dir, z_target=0.0, thickness_mpc=10.0)
    im = ax.imshow(
        np.log10(density_slice + 0.01),
        cmap='inferno', origin='lower',
        extent=[0, 150, 0, 150]
    )
    plt.colorbar(im, ax=ax, label='log₁₀(ρ/ρ̄)')
    ax.set_title("Densité projetée z=0 (slice 10 Mpc)")
    ax.set_xlabel("Mpc"); ax.set_ylabel("Mpc")
    # Référence visuelle : ajouter contour à ρ = 2ρ̄ pour visualiser filaments
    ax.contour(np.log10(density_slice + 0.01), levels=[np.log10(2)],
               colors='cyan', linewidths=0.5, alpha=0.7)

    # ── Panneau 2 : distribution tailles de voids ──────────────────────────
    ax = axes[0, 1]
    void_sizes = load_void_sizes(run_dir)
    ax.hist(void_sizes, bins=30, density=True, color='steelblue',
            alpha=0.7, label='Janus')
    ax.axvline(DESI_REF["void_median_mpc"], color='red', lw=2,
               linestyle='--', label=f'DESI médiane ({DESI_REF["void_median_mpc"]} Mpc)')
    ax.axvline(np.median(void_sizes), color='blue', lw=2,
               linestyle='-', label=f'Janus médiane ({np.median(void_sizes):.1f} Mpc)')
    ax.set_xlabel("Taille void (Mpc)"); ax.set_ylabel("Densité de probabilité")
    ax.set_title("Distribution tailles de voids")
    ax.legend(fontsize=9)

    # ── Panneau 3 : spectre de puissance P(k) ─────────────────────────────
    ax = axes[1, 0]
    k_vals, pk_janus = load_power_spectrum(run_dir, z=0.0)
    pk_lcdm = lcdm_reference_pk(k_vals)  # depuis fichier pré-calculé ou fitting formula
    ax.loglog(k_vals, pk_janus, 'r-', lw=2, label='Janus z=0')
    ax.loglog(k_vals, pk_lcdm, 'b--', lw=1.5, label='ΛCDM z=0')
    ax.set_xlabel("k (Mpc⁻¹)"); ax.set_ylabel("P(k) (Mpc³)")
    ax.set_title("Spectre de puissance")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Panneau 4 : tableau de scores ─────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')
    score_data = [
        ["Métrique", "Janus", "DESI cible", "OK ?"],
        ["Fil. médiane (Mpc)",
         f"{metrics_final.get('filament_mean_mpc', 0):.1f}",
         "7–15", "✅" if metrics_final.get('filament_mean_mpc', 0) > 7 else "❌"],
        ["Matière filaments",
         f"{metrics_final.get('fil_matter_fraction', 0)*100:.0f}%",
         "> 15%", "✅" if metrics_final.get('fil_matter_fraction', 0) > 0.15 else "❌"],
        ["Void médiane (Mpc)",
         f"{metrics_final.get('void_mode_mpc', 0):.1f}",
         "18–40", "✅" if 18 < metrics_final.get('void_mode_mpc', 0) < 40 else "❌"],
        ["Ségrégation S(z=0)",
         f"{metrics_final.get('s_segregation', 0):.3f}",
         "> 0.3", "✅" if metrics_final.get('s_segregation', 0) > 0.3 else "❌"],
        ["P(k) slope",
         f"{metrics_final.get('pk_slope', 0):.2f}",
         "−2.5/−3.5", "✅" if -3.5 < metrics_final.get('pk_slope', 0) < -2.5 else "❌"],
    ]
    table = ax.table(cellText=score_data[1:], colLabels=score_data[0],
                     loc='center', cellLoc='center')
    table.scale(1.2, 1.8)
    ax.set_title("Tableau de validation")

    plt.tight_layout()
    out_path = output_dir / "comparison_desi.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure comparaison : {out_path}")
    return out_path
```

### 5.3 Critère de validation finale

Un run est **validé pour publication** si **toutes** les conditions suivantes sont remplies :

```
✅ S(z=0) > 0.30          (ségrégation robuste)
✅ filament_mean > 7 Mpc  (filaments persistants)
✅ fil_matter_fraction > 0.15  (matière dans filaments)
✅ void_mode entre 18 et 40 Mpc
✅ P(k) slope entre −2.5 et −3.5
✅ r(k) croisé m+/m− < −0.3 à k ~ 0.05 Mpc⁻¹
```

Si validé → **lancer run 1M particules, 1200 steps, z=5→0** pour figures de publication.  
Si validé + 1M → préparer mail JPP avec : images densité projetée, P(k), S(z), r(k), tableau paramètres.

---



```
/mnt/T2/janus-sim/
├── src/
│   ├── metrics.rs          [NOUVEAU — Phase 1.4]
│   ├── early_stop.rs       [NOUVEAU — Phase 1.5]
│   ├── screening.cu        [NOUVEAU — Phase 1.2]
│   └── ... (code existant modifié)
├── optim/
│   ├── metrics.py          [NOUVEAU — Phase 2.1]
│   ├── score.py            [NOUVEAU — Phase 2.2]
│   ├── trichotomy.py       [NOUVEAU — Phase 2.3]
│   ├── run_tour.py         [NOUVEAU — Phase 2.4]
│   ├── compare_desi.py     [NOUVEAU — Phase 5.2]
│   └── tour1/
│       ├── config_run_A.yaml
│       ├── config_run_B.yaml
│       └── config_run_C.yaml
└── output/
    ├── tour1_runA/
    │   ├── metrics.jsonl
    │   ├── run.log
    │   └── snapshot_z*.bin
    ├── tour1_runB/
    └── tour1_runC/
```

---

## Annexe B — Commandes de lancement

```bash
# 0. Vérifier que le binaire compile AVANT tout
cd /mnt/T2/janus-sim
cargo build --release 2>&1 | tail -5

# 1. Générer les configs Tour 1
cd optim
python -c "
from trichotomy import _make_config, TourParams
import yaml
from pathlib import Path
Path('tour1').mkdir(exist_ok=True)
for eta, label in [(0.5,'A'), (1.0,'B'), (1.5,'C')]:
    p = TourParams(eta=eta, lambda_base=30.0, n_particles=200_000, n_steps=500)
    cfg = _make_config(p, f'tour1_run{label}')
    Path(f'tour1/config_run_{label}.yaml').write_text(yaml.dump(cfg))
    print(f'Config {label} : eta={eta}')
"

# 2. Lancer Tour 1
python run_tour.py 1 tour1/

# 3. Tour 2 (après analyse automatique par run_tour.py)
python run_tour.py 2 tour2/

# ... et ainsi de suite
```

---

*Document généré pour Claude CLI — Projet Janus simulation GPU, /mnt/T2/janus-sim/*  
*Paramètres physiques : Jean-Pierre Petit, modèle Janus bimétrique*  
*Validation cible : DESI/Euclid large-scale structure*
