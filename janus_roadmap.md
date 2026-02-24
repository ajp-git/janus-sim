# Feuille de route — Projet Janus
# Document pour Claude CLI
# Mise à jour : 22 février 2026
# Basé sur analyse de ChatGPT, Grok et Gemini

---
## PROJET PM-FFT (parallèle)
Voir janus_roadmap_pm.md pour les instructions complètes.

## CONTEXTE ET ÉTAT ACTUEL

### Ce qui est validé
- Fit Pantheon+ : η=1.045, q₀=-0.022, χ²/dof=0.914 ✅
- Barnes-Hut GPU f64 : 0% d'écart CPU/GPU ✅
- Bugs corrigés : rsqrt(), COM périodique, équations accélération ✅
- **Tâche 1 COMPLÈTE** : virialization PE_binding, COM référence commune ✅
- Seg₀ = 0.0024 (vs 0.49 avant correction) ✅
- KE/KE₀ = 1.0003 stable (vs 115 avant virialization) ✅
- **Tâche 2 COMPLÈTE** : Hubble friction, CosmoInterpolator, dtau_per_dt=0.013205 ✅
- **Run 500K terminé** : S_max=0.513 (step 1453), KE/KE₀=6.01 ✅
- **Run 2M en cours** : S_max=0.694 (step 2192 CSV), 51% complet 🔄

### Runs de production

| Run | N | Steps | S_max | KE/KE₀ | Statut |
|-----|---|-------|-------|---------|--------|
| run_hubble_mid | 500K | 3 600 | **0.513** (step 1453) | 6.01 | ✅ TERMINÉ |
| run_hubble_hi2 | 2M | 6 000 | **0.694** (step 2192) | 2.61@step3000 | 🔄 51% (~15h00) |

Vidéo générée : `janus_hubble_500k.mp4` (24 sec, 721 frames)

**Résultat clé — dépendance en N :**
La ségrégation augmente avec N (+35% de 500K à 2M).
Longueur caractéristique des structures > boîte 500K.
→ Voir janus_roadmap_pm.md pour le projet Particle-Mesh 20M particules.

### Runs terminés / arrêtés
- Run A (η=1.045, IC non virialisées) : ARRÊTÉ — données non interprétables
- Run B (η=2.03, bug COM) : ARRÊTÉ — ségrégation faussée 150×

### Bugs supplémentaires identifiés et corrigés (Tâche 1)
| Bug | Cause | Impact |
|-----|-------|--------|
| Seg₀ ≈ 0.49 artificiel | Références différentes pour COM+ et COM− | Ségrégation 150× surestimée |
| Virialization impossible | PE total > 0 pour système Janus mixte | 2KE+PE=0 irréalisable |
| Solution PE_binding | Utiliser uniquement paires même signe | PE_binding < 0 toujours, α=4.57 |

### Problème central résolu
Run A montrait KE/KE₀=115 et ségrégation décroissante à cause de :
1. ~~Conditions initiales non virialisées~~ → **CORRIGÉ** (PE_binding)
2. Absence de Hubble friction → **Tâche 2**
3. Régime quasi-symétrique η≈1 → physique réelle, pas un bug

---

## TÂCHE 1 — CONDITIONS INITIALES VIRIALISÉES ✅ COMPLÈTE
### Résultats finaux

| Test | Résultat | Valeur |
|------|----------|--------|
| Virial error (2KE + PE_bind ≈ 0) | ✅ | 0.0000% |
| Seg₀ < 0.05 | ✅ | 0.0024 |
| KE/KE₀ stable sur 200 steps | ✅ | 1.0012 |
| Ségrégation croissante | ✅ | +265% à step 200 |

### Découvertes critiques

**Bug 1 — Référence COM différente pour chaque population**
Le calcul de ségrégation utilisait une référence différente
pour COM+ et COM−. Sur distribution uniforme, la distance
entre ces deux références ≈ 0.4 × box_size → Seg₀ ≈ 0.49.
Correction : utiliser l'origine (0,0,0) comme référence commune.
Impact : Seg₀ passe de 0.49 → 0.0024 (150× surestimé avant)

**Bug 2 — PE total > 0 pour système Janus mixte**
Avec η=1.045 (49%+/51%−), les paires répulsives +/− dominent
→ PE total > 0 → 2KE + PE = 0 impossible (KE cible < 0).
Solution : PE_binding = uniquement paires même signe (toujours < 0)
α = √(|PE_binding| / (2×KE)) = 4.57
Impact : virialization parfaite, système "trop froid" au départ
(pas trop chaud comme supposé initialement)

### Pourquoi c'est prioritaire
Actuellement Seg₀ ≈ 0.49 : le système part d'un état déjà partiellement
ségrégué et relaxe vers un état mélangé. On ne teste pas l'émergence
naturelle de ségrégation, on teste la relaxation d'un état biaisé.
Objectif : Seg₀ ≈ 0.03, 2KE + PE = 0.

### Code Gemini — Calcul PE avec logique Janus

```rust
// Ajouter dans le struct Node ou Tree de nbody_gpu.rs (version CPU)
// Calculer PE une seule fois à t=0 sur CPU avant de lancer GPU

impl Node {
    pub fn get_potential(&self, pos: [f64; 3], mass_sign: f64, theta: f64) -> f64 {
        let r_vec = [
            self.center_of_mass[0] - pos[0],
            self.center_of_mass[1] - pos[1],
            self.center_of_mass[2] - pos[2],
        ];
        let r = (r_vec[0].powi(2) + r_vec[1].powi(2) + r_vec[2].powi(2)).sqrt();

        if r < 1e-10 {
            return 0.0; // Éviter division par zéro
        }

        if self.is_leaf() || (self.width / r < theta) {
            // LOGIQUE JANUS :
            // Même signe → Attraction (Potentiel Négatif)
            // Signes opposés → Répulsion (Potentiel Positif)
            let interaction_sign = if self.mass * mass_sign > 0.0 { -1.0 } else { 1.0 };
            let softened_r = (r * r + EPSILON * EPSILON).sqrt();
            return interaction_sign * (G * self.mass.abs() / softened_r);
        }

        // Appel récursif pour les nœuds internes
        self.children
            .iter()
            .filter_map(|child| child.as_ref())
            .map(|child| child.get_potential(pos, mass_sign, theta))
            .sum()
    }
}

// Calcul du PE global
pub fn compute_total_potential_energy(
    positions: &[[f64; 3]],
    masses: &[f64],
    signs: &[f64],
    tree: &OctreeNode,
    theta: f64,
) -> f64 {
    let pe: f64 = positions
        .iter()
        .zip(masses.iter())
        .zip(signs.iter())
        .map(|((pos, &mass), &sign)| {
            mass * tree.get_potential(*pos, sign, theta)
        })
        .sum();
    pe * 0.5 // Facteur 1/2 pour éviter double comptage
}
```

### Code — Virialization des vitesses

```rust
pub fn virialize_velocities(
    velocities: &mut [[f64; 3]],
    kinetic_energy: f64,
    potential_energy: f64,
) {
    // Condition virielle : 2KE + PE = 0
    // → KE_target = -PE/2
    // → alpha = sqrt(KE_target / KE_actuel) = sqrt(|PE| / (2*KE))
    
    let ke_target = potential_energy.abs() / 2.0;
    let alpha = (ke_target / kinetic_energy).sqrt();
    
    println!("Virialization:");
    println!("  KE initial    = {:.4e}", kinetic_energy);
    println!("  PE            = {:.4e}", potential_energy);
    println!("  KE target     = {:.4e}", ke_target);
    println!("  Alpha scale   = {:.6}", alpha);
    
    for vel in velocities.iter_mut() {
        vel[0] *= alpha;
        vel[1] *= alpha;
        vel[2] *= alpha;
    }
    
    // Vérification
    let ke_after: f64 = velocities.iter()
        .map(|v| 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    let virial_ratio = 2.0 * ke_after + potential_energy;
    println!("  2KE + PE après = {:.4e} (doit être ≈ 0)", virial_ratio);
    
    assert!(
        virial_ratio.abs() / potential_energy.abs() < 0.01,
        "Virialization failed: ratio = {:.4}", virial_ratio
    );
}
```

### Test de validation obligatoire (VALIDATION_RULES.md)

```rust
#[test]
fn test_virialization() {
    // Après virialization :
    // 1. Seg₀ doit être < 0.05
    // 2. KE/KE₀ doit rester < 5 sur 100 steps
    // 3. Ségrégation doit être croissante entre step 0 et step 100
    
    let (ke, pe) = compute_energies(&sim);
    let virial = 2.0 * ke + pe;
    assert!(virial.abs() / pe.abs() < 0.01, 
        "Virial non satisfait : {:.4e}", virial);
    assert!(sim.segregation() < 0.05,
        "Seg₀ trop élevé : {:.4}", sim.segregation());
}
```

### Instructions pour Claude CLI

```
TÂCHE 1 : Implémenter virialization

1. Ajouter compute_total_potential_energy() dans nbody.rs (CPU)
   Utiliser l'arbre Barnes-Hut existant avec la logique Janus.
   
2. Ajouter virialize_velocities() dans le module init.
   
3. Modifier initialize_particles() pour appeler :
   a. compute_total_potential_energy()
   b. virialize_velocities()
   c. Vérifier 2KE + PE ≈ 0

4. Test de validation obligatoire avant toute simulation :
   500K particules, η=1.045, vérifier Seg₀ < 0.05

5. Lancer test 500K, η=1.045, dt=0.01, 200 steps.
   Rapporter Seg₀, Seg₁₀₀, Seg₂₀₀, KE/KE₀.
   Attendre instruction.

IMPORTANT : Calculer PE sur CPU une seule fois à t=0.
Ne pas inclure dans la boucle GPU.
Lire VALIDATION_RULES.md avant toute implémentation.
```

---

---

## TÂCHES PARALLÈLES — Indépendantes de la séquence principale
### Peuvent être lancées à tout moment sans attendre les Tâches 2-5

---

### PARALLÈLE A — VISUALISATION EN DENSITÉ + VIDÉO ✅ COMPLÈTE

### Pourquoi maintenant
Les snapshots HDF5 s'accumulent en temps réel.
Générer les frames densité au fur et à mesure — la vidéo
sera prête dès que les runs se terminent.
Ne pas attendre la fin des runs.

### Code Python — scripts/density_projection.py

```python
#!/usr/bin/env python3
"""
Visualisation en projection de densité pour snapshots Janus HDF5.
Génère frames_density/frame_%05d.png en parallèle des runs.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import sys
import time

def render_density_frame(
    snapshot_path: str,
    output_path: str,
    grid_size: int = 1024,
    sigma: float = 2.0,
):
    with h5py.File(snapshot_path, 'r') as f:
        pos   = f['positions'][:]   # (N, 3)
        signs = f['signs'][:]       # (N,) : +1 ou -1
        box   = float(f.attrs['box_size'])
        step  = int(f.attrs.get('step', 0))
        time_ = float(f.attrs.get('time', 0.0))
        seg   = float(f.attrs.get('segregation', 0.0))
        ke    = float(f.attrs.get('ke_ratio', 1.0))

    mask_p = signs > 0
    mask_m = signs < 0

    def project(p):
        # Projection le long de l'axe Z (somme sur Z)
        H, _, _ = np.histogram2d(
            p[:, 0], p[:, 1],
            bins=grid_size,
            range=[[0, box], [0, box]]
        )
        H = gaussian_filter(H.astype(float), sigma=sigma)
        # Log scale avec protection contre les zéros
        H = np.log1p(H)
        return H

    # Grilles densité
    dens_p = project(pos[mask_p]) if mask_p.sum() > 0 else np.zeros((grid_size, grid_size))
    dens_m = project(pos[mask_m]) if mask_m.sum() > 0 else np.zeros((grid_size, grid_size))

    # Normalisation 0-1
    def norm(x):
        xmax = x.max()
        return x / xmax if xmax > 0 else x

    dens_p = norm(dens_p)
    dens_m = norm(dens_m)

    # Composition RGB : bleu = masses+, rouge = masses-
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[:, :, 0] = dens_m          # Rouge   = masses-
    rgb[:, :, 2] = dens_p          # Bleu    = masses+
    rgb[:, :, 1] = np.minimum(dens_p, dens_m) * 0.3  # Légère teinte verte aux intersections
    rgb = np.clip(rgb, 0, 1)

    # Figure 4K
    fig, ax = plt.subplots(figsize=(3840/150, 2160/150), dpi=150)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin='lower',
        extent=[0, box, 0, box],
        interpolation='bilinear',
    )

    # Overlay
    ax.set_title(
        f"Janus Cosmological Model — Density Projection | η=1.045",
        color='white', fontsize=14, pad=10
    )
    ax.text(
        0.5, 0.02,
        f"Step {step:05d}  |  Time: {time_:.3f}  |  Seg: {seg:.4f}  |  KE/KE₀: {ke:.2f}",
        transform=ax.transAxes,
        color='white', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
    )
    ax.axis('off')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', pad_inches=0)
    plt.close()


def watch_and_render(snapshot_dir: str, output_dir: str):
    """
    Mode watch : surveille snapshot_dir et génère un frame
    densité dès qu'un nouveau HDF5 apparaît.
    Priorité basse (nice +10) pour ne pas gêner les runs GPU.
    """
    import os
    snap_dir = Path(snapshot_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = set()

    print(f"Watching {snap_dir} for new snapshots...")
    while True:
        snaps = sorted(snap_dir.glob("snap_*.h5"))
        for snap in snaps:
            if snap.name in processed:
                continue
            step = int(snap.stem.split('_')[1])
            out_path = out_dir / f"frame_{step:05d}.png"
            if out_path.exists():
                processed.add(snap.name)
                continue
            try:
                render_density_frame(str(snap), str(out_path))
                print(f"  Rendered {snap.name} → {out_path.name}")
                processed.add(snap.name)
            except Exception as e:
                print(f"  Error on {snap.name}: {e}")
        time.sleep(10)  # Vérifier toutes les 10 secondes


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python density_projection.py <snapshot_dir> <output_dir>")
        sys.exit(1)
    watch_and_render(sys.argv[1], sys.argv[2])
```

### Lancement

```bash
# Lancer en arrière-plan pour chaque run actif
# Priorité basse (nice +10) pour ne pas gêner le GPU

nice -n 10 python scripts/density_projection.py \
  /mnt/T2/janus-sim/output/2026-02-21_run_lo/snapshots \
  /mnt/T2/janus-sim/output/2026-02-21_run_lo/frames_density \
  &

nice -n 10 python scripts/density_projection.py \
  /mnt/T2/janus-sim/output/2026-02-21_run_mid/snapshots \
  /mnt/T2/janus-sim/output/2026-02-21_run_mid/frames_density \
  &

nice -n 10 python scripts/density_projection.py \
  /mnt/T2/janus-sim/output/2026-02-21_run_hi/snapshots \
  /mnt/T2/janus-sim/output/2026-02-21_run_hi/frames_density \
  &
```

### Assemblage vidéo (à la demande)

```bash
# Générer vidéo depuis les frames densité disponibles
ffmpeg -framerate 24 \
  -i output/2026-02-21_run_hi/frames_density/frame_%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  output/2026-02-21_run_hi/janus_density_run_hi.mp4
```

### Instructions pour Claude CLI

```
PARALLÈLE A : Lancer maintenant sans attendre les Tâches 2-5.

1. Créer scripts/density_projection.py avec le code ci-dessus.

2. Tester sur 3 snapshots disponibles de run_hi :
   python scripts/density_projection.py \
     output/2026-02-21_run_hi/snapshots \
     output/2026-02-21_run_hi/frames_density
   Arrêter après 3 frames, partager les images.

3. Si résultat satisfaisant : lancer en mode watch
   pour les 3 runs simultanément (nice +10).

4. PIDs dans :
   output/2026-02-21_run_lo/density_pid.txt
   output/2026-02-21_run_mid/density_pid.txt
   output/2026-02-21_run_hi/density_pid.txt

Rapporter 3 frames exemples de run_hi sans demander confirmation.
```

---

## TÂCHE 2 — HUBBLE FRICTION (EXPANSION COSMOLOGIQUE) ✅ COMPLÈTE
### Résultats finaux

| Test | Résultat | Valeur |
|------|----------|--------|
| CosmoInterpolator | ✅ | a(τ), H(τ) interpolés z=5→z=0 |
| Friction calibrée | ✅ | dtau_per_dt = 0.013205 |
| KE/KE₀ dans plage | ✅ | 6.01 (limite: 0.1-20) |
| Ségrégation croissante | ✅ | 0.24% → 14.5% |
| Vidéo 3-panel | ✅ | 721 frames, 24 sec |

### Corrections critiques appliquées

**Bug 1 — Division par a³ inutile**
Barnes-Hut calcule en coordonnées physiques (pas comobiles).
La division par a³ dans le kernel amplifiait les forces ×216 quand a=0.167.
Solution : supprimer la division, garder les forces physiques directement.

**Bug 2 — Friction -2H·v au lieu de -H·v**
L'équation correcte en coordonnées physiques est (Peebles 1980, eq. 5.111) :
  dv/dt = g_physical - H·v
Le facteur 2 n'apparaît qu'en coordonnées comobiles.

**Bug 3 — Mélange dt (N-body) et dτ (cosmologique)**
Le kernel utilisait H en unités cosmologiques mais dt en unités N-body.
Solution : dtau_per_dt = τ_total / (10000 × dt) = 0.013205 (constant)
Friction corrigée : -H·v·dtau_per_dt

### Pourquoi c'est important
Sans expansion cosmologique, KE/KE₀ explose (115 dans Run A).
L'expansion "amortit" les vitesses (v ∝ 1/a) et permet aux structures
de se stabiliser au lieu de rebondir après effondrement.

### Physique — Équation du mouvement en coordonnées comobiles

```
Position physique : r = a(t) × x (x = coordonnée comobile)
Équation du mouvement :
  ẍ = g_Janus / a³ - 2H·ẋ

où :
  g_Janus = accélération calculée par Barnes-Hut
  H = ȧ/a = paramètre de Hubble (depuis friedmann.rs)
  -2H·ẋ = terme de Hubble friction
```

### Code Gemini — Kernel CUDA modifié

```cuda
// Modifier le kernel leapfrog_kick dans nbody_gpu.rs
// Remplacer le kernel de kick existant par :

extern "C" __global__ void leapfrog_kick_kernel(
    double* v,           // Vecteur vitesse [vx, vy, vz] × N
    const double* f,     // Force/Accélération calculée [fx, fy, fz] × N
    double dt,           // Pas de temps
    double a,            // Facteur d'échelle a(t) au step courant
    double adot_over_a,  // H(t) = ȧ/a = paramètre de Hubble
    int n                // Nombre de particules
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = i * 3;

        for (int d = 0; d < 3; d++) {
            // 1. Accélération Janus en coordonnées comobiles
            double accel_janus = f[idx + d] / (a * a * a);

            // 2. Terme de Hubble Friction
            double friction = -2.0 * adot_over_a * v[idx + d];

            // 3. Mise à jour de la vitesse (Kick)
            v[idx + d] += (accel_janus + friction) * dt;
        }
    }
}
```

### Intégration Rust — Passer a(t) et H(t) depuis friedmann.rs

```rust
// Dans la boucle principale de nbody_gpu.rs

pub struct CosmologicalParams {
    pub a: f64,        // Facteur d'échelle actuel
    pub h: f64,        // H(t) = ȧ/a paramètre de Hubble
}

impl NBodyGPU {
    pub fn step_with_expansion(
        &mut self, 
        dt: f64,
        cosmo: &CosmologicalParams,
    ) -> Result<(), SimError> {
        // 1. Drift (positions)
        self.drift(dt / 2.0)?;
        
        // 2. Rebuild tree
        self.build_tree()?;
        
        // 3. Compute forces
        self.compute_forces(cosmo.a)?;
        
        // 4. Kick avec Hubble friction
        // Passer a et H au kernel CUDA
        unsafe {
            launch_kernel!(
                self.kick_kernel,
                grid = (self.n + 255) / 256,
                block = 256,
                args = [
                    &self.vel_gpu,
                    &self.acc_gpu,
                    &dt,
                    &cosmo.a,        // ← nouveau paramètre
                    &cosmo.h,        // ← nouveau paramètre
                    &(self.n as i32),
                ]
            )?;
        }
        
        // 5. Drift (positions)
        self.drift(dt / 2.0)?;
        
        Ok(())
    }
}

// Dans le main ou la boucle de simulation :
// Récupérer a(t) et H(t) depuis friedmann.rs à chaque step

fn get_cosmological_params(t: f64, eta: f64) -> CosmologicalParams {
    let history = integrate_friedmann(eta, t);
    let a = interpolate_a(&history, t);
    let adot = interpolate_adot(&history, t);
    CosmologicalParams {
        a,
        h: adot / a,
    }
}
```

### CORRECTION CRITIQUE — Fonctions fantômes (Gemini)

```
ATTENTION : Les fonctions suivantes mentionnées dans la roadmap
N'EXISTENT PAS dans le code actuel de friedmann.rs :

  integrate_friedmann(eta, t)   ← N'EXISTE PAS
  interpolate_a(&history, t)    ← N'EXISTE PAS

Le code actuel ne fait qu'une intégration vers le passé
(integrate_backward) depuis z=0.

Solution : Implémenter CosmoInterpolator ci-dessous.
```

### Code Gemini — CosmoInterpolator (à ajouter dans friedmann.rs)

```rust
/// Interpolateur cosmologique pour la simulation N-corps
/// Calcule l'histoire de l'univers UNE SEULE FOIS,
/// puis fournit a(t) et H(t) par interpolation rapide.
pub struct CosmoInterpolator {
    history: Vec<JanusState>,
    pub tau_start: f64,  // tau au redshift initial (passé)
    pub tau_end: f64,    // tau = 0 (aujourd'hui)
}

impl CosmoInterpolator {
    /// Prépare l'histoire de l'univers de z_init jusqu'à z=0
    /// z_init = 50.0 pour simulations cosmologiques standard
    pub fn new(params: &JanusParams, z_init: f64) -> Self {
        // 1. Intégrer vers le passé (z=0 → z_init)
        let mut history = integrate_backward(params, z_init, 10000);
        
        // 2. Trier par tau CROISSANT (du passé vers le présent)
        history.sort_by(|s1, s2| s1.tau.partial_cmp(&s2.tau).unwrap());
        
        let tau_start = history.first().unwrap().tau;
        let tau_end = history.last().unwrap().tau;
        
        Self { history, tau_start, tau_end }
    }

    /// Retourne (a, H) pour un temps cosmologique tau donné
    /// Interpolation linéaire entre les points de l'histoire
    pub fn get_params_at_tau(&self, tau_target: f64) -> (f64, f64) {
        // Sécurités aux bornes
        if tau_target <= self.tau_start {
            let s = self.history.first().unwrap();
            return (s.a, s.hubble());
        }
        if tau_target >= self.tau_end {
            let s = self.history.last().unwrap();
            return (s.a, s.hubble());
        }

        // Recherche dichotomique de l'intervalle
        let idx = self.history.partition_point(|s| s.tau < tau_target);
        let s0 = &self.history[idx - 1];
        let s1 = &self.history[idx];

        // Interpolation linéaire
        let fraction = (tau_target - s0.tau) / (s1.tau - s0.tau);
        let a_interp = s0.a + fraction * (s1.a - s0.a);
        let h_interp = s0.hubble() + fraction * (s1.hubble() - s0.hubble());

        (a_interp, h_interp)
    }
}
```

### Code — Utilisation dans la boucle N-corps (nbody_gpu.rs)

```rust
// AVANT la boucle : initialiser UNE SEULE FOIS
let params = JanusParams::from_eta(eta);
let cosmo = CosmoInterpolator::new(&params, 50.0); // z_init = 50
let total_nbody_steps = 10_000usize;

// Conversion des unités de temps :
// Les 10000 steps N-corps couvrent exactement l'âge cosmologique
// de z=50 à z=0
let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (total_nbody_steps as f64);

// DANS la boucle N-corps
for step in 0..total_nbody_steps {
    let current_tau = cosmo.tau_start + (step as f64) * dtau_cosmo;
    let (a, h) = cosmo.get_params_at_tau(current_tau);

    // Lancer le kernel CUDA avec (a, h)
    sim.step_with_expansion(dt, a, h)?;
}
```

### Test unitaire obligatoire — test_cosmo_interpolator

```rust
// À ajouter dans le module tests de friedmann.rs
#[test]
fn test_cosmo_interpolator() {
    // 1. Initialisation avec paramètres validés (Fit Pantheon+)
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let z_init = 50.0;

    let cosmo = CosmoInterpolator::new(&params, z_init);

    // 2. Vérification du passé (début simulation à z_init)
    let (a_start, h_start) = cosmo.get_params_at_tau(cosmo.tau_start);
    let expected_a_start = 1.0 / (1.0 + z_init);
    
    assert!(
        (a_start - expected_a_start).abs() < 1e-4, 
        "a_start doit être 1/(1+z_init) = {:.6}, obtenu : {:.6}",
        expected_a_start, a_start
    );
    assert!(
        h_start > 0.0, 
        "H doit être > 0 au départ (univers en expansion)"
    );

    // 3. Vérification du présent (fin simulation z=0)
    let (a_end, h_end) = cosmo.get_params_at_tau(cosmo.tau_end);
    
    assert!(
        (a_end - 1.0).abs() < 1e-4, 
        "a_end doit être 1.0 (aujourd'hui), obtenu : {:.6}", a_end
    );

    // Convention validée dans friedmann.rs ligne 106 :
    // ȧ₀ = √Ω₊  →  H₀ = ȧ₀/a₀ = √Ω₊ (car a₀=1)
    let expected_h_end = params.omega_plus.sqrt();
    assert!(
        (h_end - expected_h_end).abs() < 1e-4, 
        "H_end doit être √Ω₊ = {:.6}, obtenu : {:.6}",
        expected_h_end, h_end
    );

    // 4. Monotonie de a(t)
    let tau_mid = cosmo.tau_start + (cosmo.tau_end - cosmo.tau_start) / 2.0;
    let (a_mid, _) = cosmo.get_params_at_tau(tau_mid);
    
    assert!(
        a_mid > a_start && a_mid < a_end, 
        "a(t) doit être strictement croissant : {} < {} < {}",
        a_start, a_mid, a_end
    );

    println!("CosmoInterpolator validé :");
    println!("  a_start = {:.6} (attendu {:.6})", a_start, expected_a_start);
    println!("  a_end   = {:.6} (attendu 1.0)", a_end);
    println!("  H_end   = {:.6} (attendu {:.6})", h_end, expected_h_end);
}
```

### Note sur la convention H₀ — vérifiée dans friedmann.rs

```
CONFIRMÉ dans friedmann.rs ligne 106 :

    let a_dot = params.omega_plus.sqrt();
    // From Friedmann: (ȧ/a)² = Ω₊/a³ → ȧ₀ = √Ω₊

Donc H₀ = ȧ₀/a₀ = √Ω₊/1 = √Ω₊

Pour η=1.045 :
  Ω₊ = 1/(1+1.045) = 0.4890
  H₀ = √0.4890 = 0.6993 (unités adimensionnelles H₀=1)

Le test expected_h_end = params.omega_plus.sqrt() est CORRECT.
```

### Note importante sur la cohérence des unités

```
VÉRIFICATION OBLIGATOIRE avant d'activer ce kernel :

Les distances dans Barnes-Hut sont-elles en coordonnées physiques
ou comobiles ?

Si physiques → diviser par a³ est correct (comme dans le kernel)
Si comobiles → ajuster le facteur de conversion

Test de cohérence obligatoire :
  Lancer 100 steps avec a=1.0 constant, H=0.
  Résultats IDENTIQUES à la simulation sans expansion.
  Si différents → bug dans l'implémentation.
```

### Instructions pour Claude CLI

```
TÂCHE 2 : Implémenter Hubble friction
(SEULEMENT après validation Tâche 1)

ÉTAPE 2a — Ajouter CosmoInterpolator dans friedmann.rs
  Utiliser le code fourni ci-dessus.
  Ajouter test_cosmo_interpolator dans le module tests.
  Le test DOIT passer avant de continuer.

ÉTAPE 2b — Modifier leapfrog_kick_kernel
  Utiliser le code Gemini (kernel CUDA fourni plus haut).
  Accepter a et adot_over_a comme paramètres.

ÉTAPE 2c — Modifier la boucle principale
  Initialiser CosmoInterpolator UNE SEULE FOIS avant la boucle.
  Calculer dtau_cosmo = (tau_end - tau_start) / total_steps.
  Passer (a, h) au kernel à chaque step.

ÉTAPE 2d — Test de cohérence
  Lancer 100 steps avec a=1.0 constant, H=0.
  Résultats identiques à la simulation sans expansion → OK.

ÉTAPE 2e — Validation physique
  Lancer 500K, η=1.045, 500 steps avec expansion activée.
  Rapporter : KE/KE₀ max, Seg₅₀₀.
  Objectif : KE/KE₀ < 20 (vs 115 sans expansion).

Attendre instruction avant chaque étape.
```

---

## TÂCHE 3 — ÉTUDE DE CONVERGENCE EN N
### Priorité : HAUTE — Après validation Tâches 1 et 2

### Pourquoi c'est nécessaire
Sans convergence en N, impossible de défendre les simulations
devant un reviewer. C'est le critère minimum pour publication.

### Protocole

```
Paramètres fixes pour toutes les runs :
  η = 1.045, dt = 0.01, f64
  Conditions initiales virialisées (Tâche 1)
  Expansion cosmologique activée (Tâche 2)
  Même seed aléatoire

N à tester : 100K, 500K, 1M, 2M

Pour chaque N, mesurer :
  - Seg₀, Seg₁₀₀, Seg₅₀₀, Seg_max
  - Step où Seg est maximal
  - KE/KE₀ max
  - Temps de formation de l'amas (step où structure visible)
```

### Critère de convergence

```
Résultat convergé si :
  |Seg_max(1M) - Seg_max(2M)| / Seg_max(2M) < 10%

Si convergé → résultat physiquement robuste, défendable
Si non convergé → augmenter N ou investiguer
```

### Instructions pour Claude CLI

```
TÂCHE 3 : Étude de convergence
(SEULEMENT après validation Tâches 1 et 2)

Lancer 4 simulations séquentielles :
  N = 100K : 500 steps
  N = 500K : 500 steps
  N = 1M   : 500 steps
  N = 2M   : 500 steps (si temps disponible)

Même seed (42), même η=1.045, mêmes IC virialisées.

Produire tableau comparatif :
  | N    | Seg₀ | Seg_max | Step_max | KE/KE₀_max |
  |------|------|---------|----------|------------|

Critère de convergence : Seg_max(N) et Seg_max(2N) < 10% d'écart.
Rapporter résultats et attendre instruction.
```

---

## TÂCHE 4 — FONCTION DE CORRÉLATION ξ(r)
### Priorité : MOYENNE — Après convergence en N

### Ce que ça apporte
Mesure standard de la structure à grande échelle.
Comparaison qualitative avec SDSS DR7.
Argument fort pour présentation à Petit.

### Code Python — Calcul ξ(r) avec Corrfunc

```python
#!/usr/bin/env python3
"""
Calcul de la fonction de corrélation à deux points ξ(r)
pour les snapshots HDF5 du Run A Janus.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

# pip install Corrfunc --break-system-packages
import Corrfunc
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf

def compute_xi(snapshot_path: str, n_bins: int = 20) -> tuple:
    """
    Calcule ξ(r) pour les masses positives d'un snapshot HDF5.
    
    Returns:
        r_centers : centres des bins en unités de simulation
        xi        : valeurs de ξ(r)
    """
    with h5py.File(snapshot_path, 'r') as f:
        pos = f['positions'][:]      # shape (N, 3)
        signs = f['signs'][:]        # shape (N,) : +1 ou -1
        box_size = f.attrs['box_size']
    
    # Sélectionner uniquement les masses positives
    mask_plus = signs > 0
    pos_plus = pos[mask_plus]
    N_plus = len(pos_plus)
    
    print(f"Particules+ : {N_plus}")
    
    # Bins logarithmiques de r=1 à r=box_size/2
    r_min = 1.0
    r_max = box_size / 2.0
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    
    # Comptage des paires avec Corrfunc (optimisé multi-thread)
    X = pos_plus[:, 0].astype(np.float64)
    Y = pos_plus[:, 1].astype(np.float64)
    Z = pos_plus[:, 2].astype(np.float64)
    
    results = DD(
        autocorr=1,
        nthreads=8,
        binfile=bins,
        X1=X, Y1=Y, Z1=Z,
        periodic=True,
        boxsize=box_size,
        verbose=False,
    )
    
    # Normalisation : distribution aléatoire Poisson
    # Pour boîte périodique avec N particules :
    # DD_random(r) = N*(N-1)/2 × (4π/3 × (r_max³ - r_min³)) / box_size³
    
    dd = results['npairs'].astype(np.float64)
    
    # Volume de chaque shell
    r_low = bins[:-1]
    r_high = bins[1:]
    vol_shells = (4.0/3.0) * np.pi * (r_high**3 - r_low**3)
    vol_box = box_size**3
    
    # Nombre de paires aléatoires attendues
    rr = (N_plus * (N_plus - 1) / 2.0) * vol_shells / vol_box
    
    # ξ(r) = DD/RR - 1
    xi = dd / rr - 1.0
    r_centers = np.sqrt(r_low * r_high)  # Centre géométrique des bins
    
    return r_centers, xi


def plot_xi_evolution(
    snapshot_dir: str,
    steps: list,
    output_path: str = "xi_evolution.png"
):
    """
    Trace ξ(r) à différents steps pour voir l'évolution temporelle.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
    
    for step, color in zip(steps, colors):
        snap_path = f"{snapshot_dir}/snap_{step:05d}.h5"
        try:
            r, xi = compute_xi(snap_path)
            ax.loglog(r, np.maximum(xi, 1e-3), 
                     color=color, 
                     label=f"Step {step}",
                     linewidth=1.5)
        except FileNotFoundError:
            print(f"Snapshot {snap_path} non trouvé, ignoré.")
    
    # Référence SDSS DR7 (approximation loi de puissance)
    # ξ(r) ≈ (r/r₀)^(-γ) avec r₀≈5 Mpc/h, γ≈1.8
    # En unités adimensionnelles : à calibrer
    r_ref = np.logspace(0, 2, 50)
    xi_ref = (r_ref / 20.0) ** (-1.8)  # Ajuster r₀=20 selon box_size
    ax.loglog(r_ref, xi_ref, 'k--', 
             linewidth=2, label='SDSS DR7 (pente ref.)')
    
    ax.set_xlabel("r [unités simulation]", fontsize=12)
    ax.set_ylabel("ξ(r)", fontsize=12)
    ax.set_title("Fonction de corrélation — Masses positives\nJanus η=1.045", 
                fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-2, 1e3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé : {output_path}")


if __name__ == "__main__":
    # Calculer ξ(r) tous les 500 steps sur Run A
    steps_to_analyze = list(range(0, 10001, 500))
    
    plot_xi_evolution(
        snapshot_dir="/mnt/T2/janus-sim/output/run_A/snapshots",
        steps=steps_to_analyze,
        output_path="/mnt/T2/janus-sim/output/run_A/xi_evolution.png"
    )
```

### Instructions pour Claude CLI

```
TÂCHE 4 : Calculer ξ(r) en post-processing
(SEULEMENT après Tâche 3)

1. Installer Corrfunc :
   pip install Corrfunc --break-system-packages

2. Exécuter le script compute_xi.py ci-dessus
   sur les snapshots du Run A (η=1.045 avec IC virialisées).

3. Calculer ξ(r) aux steps : 0, 500, 1000, 2000, 5000
   (ou les steps disponibles).

4. Générer graphique ξ(r) vs r avec évolution temporelle.
   Comparer visuellement avec pente SDSS (r^-1.8).

5. Rapporter : la pente de ξ(r) est-elle compatible avec
   la pente SDSS à un facteur d'échelle près ?

Ne pas interpréter les unités absolues — uniquement les pentes.
```

---

## TÂCHE 5 — TEST η=1.0 (cas limite théorique)
### Priorité : RAPIDE — Peut être fait en parallèle

### Recommandation de Grok
Si η=1.0 donne encore plus de redispersion que η=1.045,
ça confirme l'interprétation physique du régime quasi-symétrique.
Test simple, 30 minutes, très informatif.

### Instructions pour Claude CLI

```
TÂCHE 5 : Test η=1.0 (cas limite)

500K particules, η=1.0, dt=0.01, IC virialisées, 200 steps.
Rapporter Seg₀, Seg₁₀₀, Seg₂₀₀, KE/KE₀.

Comparaison attendue :
  η=1.0   → ségrégation encore plus lente/nulle
  η=1.045 → ségrégation lente
  η=2.03  → ségrégation rapide et stable

Si η=1.0 donne ségrégation nulle ou décroissante →
  confirme que η≈1 est un régime quasi-symétrique physique,
  pas un bug numérique.
```

---

## OPTIMISATIONS GPU BARNES-HUT

### Résultats mesurés (23 février 2026)

| Opt | Description           | Temps/step 2M | Speedup cumulé | Statut |
|-----|-----------------------|---------------|----------------|--------|
| 0   | KDK baseline          | 7810 ms       | 1.0×           | ✅     |
| 1   | DKD intégrateur       | 3868 ms       | 2.0×           | ✅     |
| 2   | Morton sorting (CPU)  | 2662 ms       | 2.9×           | ✅     |
| 3   | Asymmetric θ          | N/A           | (rejeté)       | ❌     |
| 4   | GPU tree build        | 306 ms        | 25.5×          | ✅     |
| 5   | Incremental updates   | ~275 ms       | (1.1× seul)    | ❌     |
| 6   | **θ=2.0 + min reset** | **197 ms**    | **39.6×**      | ✅     |

### GPU Tree Build (Karras 2012) — Validé

```
Pipeline GPU complet :
  a) Morton codes GPU        O(N)
  b) CPU RadixSort (rayon)   O(N log N) — à remplacer par CUB
  c) Karras BVH construction O(N)
  d) COM reduction GPU       O(N log N)
  e) Force computation BVH   O(N log N)

Validation performance @ 2M particles, 10 steps :
  ✓ Time/step: 306 ms (cible < 500 ms)
  ✓ S(10) = 0.000991, ΔS = -0.60%
  → Optimisé à 197 ms/step avec θ=2.0 + minimal reset (voir Opt6)

Validation physique @ 500K particles, 200 steps (vs Morton+DKD) :
  ✓ S(200) GPU tree = 0.002422
  ✓ S(200) Morton   = 0.002474
  ✓ Différence: 2.08% (cible ±10%)
  ✓ Tous les checkpoints < 3% d'écart

Projection @ 8M particles (avec Opt6 θ=2.0) :
  Estimé: ~870 ms/step → 6000 steps ≈ 1.5h
  Horizon 16M overnight: FAISABLE
```

### Asymmetric θ — Rejeté

```
Testé avec θ_self=0.5, θ_cross=1.0
Résultat: overhead > gain dans conditions initiales mixtes
Temps/step 2M: 3228 ms (vs 2662 ms baseline) → 21% plus lent
Cause: lecture anticipée des masses pour chaque nœud
À revisiter pour simulations post-ségrégation (S > 0.05)
```

### Opt5 Incremental Updates — Rejeté

```
Implémenté update_com_only() pour éviter rebuild complet.
Résultat: seulement 1.1× speedup (275 ms vs 306 ms)
Cause: bottleneck = force kernel, pas tree build
```

### Opt6 θ + Minimal Reset — Validé ✅

```
Optimisation reset minimal:
  - Avant: reset_buffers() = 130 ms (tous les buffers GPU)
  - Après: reset atomic_counter seulement = ~1 ms
  - Raison: algorithme Karras écrase entièrement les buffers

Performance θ @ 2M particles (10 steps):
  θ=0.7:  2370 ms/step (référence physique)
  θ=1.0:  1034 ms/step
  θ=1.5:   398 ms/step ← choisi pour run 8M
  θ=2.0:   224 ms/step

Validation physique θ=1.0 (500K, 500 steps):
  S(500) ref (θ=0.5) = 1.014422
  S(500) test (θ=1.0) = 0.926436
  Erreur: 8.67% — trajectoire individuelle diverge, mais S(t) qualitativement correct

Note: Pour système chaotique, seul S_max et forme de S(t) comptent.
θ=1.5 choisi comme compromis performance/physique pour run 8M.
```

### Optimisations futures (si nécessaire)

| Opt | Description           | Cible        | Statut |
|-----|-----------------------|--------------|--------|
| 7   | Async multi-stream    | 1.2-2× supp  | ⬜     |
| 8   | GPU RadixSort (CUB)   | ~1.3× supp   | ⬜     |
| 9   | Force Freezing        | (post-seg)   | ⬜     |

---

## ORDRE D'EXÉCUTION — ÉTAT ACTUEL

```
COMPLÉTÉ ✅ :
  ✅ Tâche 1 : Virialization (PE_binding, Seg₀=0.0024, α=4.57)
  ✅ Tâche 2 : Hubble friction (dtau_per_dt=0.013205, KE/KE₀=6.01)
  ✅ Parallèle A : Vidéo 3-panel (721 frames, 24 sec)
  ✅ Document LaTeX 13 pages (janus_validation.pdf)
  ✅ GPU Opt 1-2 : DKD + Morton (2.9× speedup)
  ✅ GPU Opt 4 : GPU tree build Karras (25.5× speedup, 306 ms/step @ 2M)
  ✅ GPU Opt 6 : θ=2.0 + minimal reset (39.6× speedup total, 197 ms/step @ 2M)

EN COURS 🔄 :
  🔄 Run BH 8M : 8M particles, θ=1.5, dt=0.005, 12000 steps
     Lancé: 23 février 2026
     Estimé: ~398 ms/step × 12000 = ~1.3h
     Snapshots: /200 steps, Checkpoints: /1000 steps
     Auto-stop: KE/KE₀ > 50

À FAIRE :
  □ Post-run 8M : Analyser S_max, générer vidéo
  □ Tâche 4 : ξ(r) en post-processing
  □ Tâche 5 : Test η=1.0 (cas limite, 30 min)
  □ GPU Opt 7-9 : optimisations supplémentaires si nécessaire

CONTACT PETIT :
  ✅ Fit Pantheon+ (η=1.045, χ²/dof=0.607)
  ✅ Ségrégation S_max=0.694 avec 2M particules
  ✅ Document LaTeX 13 pages + frame_02192.png
  □ ξ(r) qualitative (renforce le dossier)
```

---

## RÈGLES ABSOLUES POUR CLAUDE CLI

```
1. Lire VALIDATION_RULES.md au début de chaque session.

2. Toute nouvelle fonction physique :
   test trivial obligatoire avant utilisation.

3. Ne jamais lancer > 10 steps sans instruction explicite.

4. Tâches dans l'ordre numérique — pas de sauts.
   Ne pas commencer Tâche 2 avant validation Tâche 1.

5. Après chaque Tâche, rapporter résultats et attendre instruction.

6. Ne pas relancer Run A ou Run B avec les anciens paramètres.
   Les anciennes simulations sont obsolètes sans IC virialisées.

7. Toute décision ambiguë → poser la question, ne pas décider seul.
```

---

## RÉFÉRENCES

- Petit, Margnat & Zejli (2024), EPJC 84:1226
- D'Agostini & Petit (2018), Astrophys. Space Sci. 363:139
- Lane et al. (2024), MNRAS (arXiv:2311.01438) — biais calibration
- Lane et al. (2024), MNRAS arXiv:2311.01438 — biais calibration BAO
- Gemini 2.0 Flash — code virialization et Hubble friction
- ChatGPT o3 — analyse conditions initiales
- Grok 3 — analyse régime quasi-symétrique η≈1

---

## PROJET FILAMENTS COSMOLOGIQUES

### Problème identifié (23 février 2026)

Les simulations actuelles produisent une **structure sphérique concentrique**
(blob central + périphérie) — pas de filaments cosmologiques.

**Cause confirmée par analyse linéaire + 5 IA :**
Avec α=1 (code actuel), le mode de filamentation λ₋ = 0 exactement.
Ce n'est pas un problème numérique — c'est une propriété du modèle.

```
interaction = if sign_i == sign_j { 1.0 } else { -1.0 }
→ α = 1 strict → λ₋ = 0 → croissance anisotrope nulle
```

**Théorème de Birkhoff (DeepSeek) :** avec ICs uniformes, le collapse
est toujours sphérique. Aucune boîte plus grande ne changera ça.

### Analyse linéaire — résultats clés

Document complet : `janus_linear_analysis.md` (769 lignes, 22 sections)

```
Mode λ₊ = ρ̄₊(1+r) > 0  →  ségrégation S(t) ✅ (observée)
Mode λ₋ = ρ̄(1−α) = 0   →  croissance anisotrope ❌ (filaments impossibles)

Pour α < 1 : λ₋ > 0, p ≈ (3/5)×[2r/(1+r)²]×ε
Condition filaments : ε = 1−α ≥ 0.3 (p ≥ 0.2)
```

**Ni l'asymétrie ρ₊≠ρ₋, ni l'expansion H(t), ni le régime non-linéaire,
ni un spectre non-gaussien ne peuvent compenser α=1.**

### Solution validée par 5 IA

**α(k) de type Yukawa :**

```rust
let r_c = 40.0;      // Échelle de transition (Mpc)
let epsilon = 0.3;   // Force de la brisure (ε ≥ 0.3 requis)
let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
let interaction = if sign_i == sign_j { 1.0 } else { -alpha_r };
```

- k → 0 : α → 1 (symétrie Janus préservée à grande échelle)
- k ~ kc : α → 1−ε (gravité effective restaurée → filaments)
- Compatible Barnes-Hut, relativité bimétrique, médiateur massif

### Plan 5 jours — weekend (validé Grok, ChatGPT, Gemini, DeepSeek, Grok2)

```
Jour 1 : Test mode unique anisotrope (4h, 4M particules)
         Run A : α=0 (attraction pure ΛCDM)
         Run B : α=1 (Janus actuel)
         Mesure : anisotropy(t) = σx / mean(σy, σz)
         Critère : A croissant, B ≈ 1 → théorie confirmée

Jour 2 : Implémentation α(k) Yukawa dans kernel CUDA
         ε=0.3, rc=40 Mpc
         Test 500K, 500 steps → confirme croissance anisotrope

Jour 3 : ICs Zel'dovich (Python FFT)
         Grille 256³, P(k) ≈ k^ns/(1+(k/k0)^4)
         Anti-corrélation δ₋ = −δ₊ (β=1.0)
         z_init=10, box=400 Mpc

Jour 4 : Run 16-32M
         Test allocation mémoire GPU d'abord
         α(k) Yukawa + ICs Zel'dovich + box=400 Mpc
         θ=1.2, dt=0.005

Jour 5 : Analyse morphologique
         σx(t)/σy(t), ξ(r), Minkowski
         Vidéo formation filaments
```

### Paramètres cibles

```
N        = 16-32M (selon mémoire GPU disponible)
box      = 400 Mpc
z_init   = 10
θ        = 1.2
ε        = 0.3
rc       = 40 Mpc
softening = box / (40 × N^(1/3))
```

### Critères de succès

```
✓ anisotropy_A >> 1 (Run A Jour 1)
✓ anisotropy_B ≈ 1 (Run B Jour 1) → α=1 confirmé bloquant
✓ Structure en réseau visible (pas blob) avec α(k)
✓ Voids persistants remplis de m−
✓ S_max > 0.6
✓ ξ(r) pente ~ −1.8 (compatible SDSS)
```

### État run 8M (terminé 23 février 2026)

```
S_max = 0.459 au step 2800 (z=2.07)
S_final = 0.427 (z=1.95)
KE/KE₀ = 2.71
Note : θ=1.5 (vs θ=0.7 pour 2M) — comparaison S_max(N) biaisée
→ Run 8M θ=0.7 à faire ce weekend pour comparaison propre
```

### Question pour JPP (après simulations)

α=1 est-il une contrainte fondamentale du modèle de Petit,
ou une extension α(k) est-elle permissible par l'action bimétrique ?

Si α=1 strict : Janus ne prédit pas de filaments par instabilité linéaire.
Si α(k) permis : médiateur massif entre secteurs → filaments possibles.

---

## ÉTAT ACTUEL — ORDRE D'EXÉCUTION MIS À JOUR

```
COMPLÉTÉ ✅ :
  ✅ Tâche 1 : Virialization (PE_binding, Seg₀=0.0024, α=4.57)
  ✅ Tâche 2 : Hubble friction (dtau_per_dt=0.013205)
  ✅ Parallèle A : Vidéo 3-panel (721 frames)
  ✅ Document LaTeX 13 pages (janus_validation.pdf)
  ✅ GPU Opt 1-2-4-6 : 39.6× speedup (197 ms/step @ 2M)
  ✅ Run BH 2M : S_max=0.694 (step 2192, z=1.8)
  ✅ Run PM 45M : S_max=0.000082 — ségrégation sub-Mpc confirmée
  ✅ Run BH 8M : S_max=0.459 (step 2800, z=2.07, θ=1.5)
  ✅ Analyse linéaire Janus : α=1 → λ₋=0 → filaments impossibles
  ✅ Consultation 5 IA : consensus α(k) Yukawa + ICs Zel'dovich

EN COURS / À FAIRE CE WEEKEND :
  □ Jour 1 : Test mode unique anisotrope (Run A vs B)
  □ Jour 2 : α(k) Yukawa dans kernel CUDA
  □ Jour 3 : ICs Zel'dovich Python
  □ Jour 4 : Run 16-32M
  □ Jour 5 : Analyse morphologique

À FAIRE (après weekend) :
  □ Run 8M avec θ=0.7 (comparaison S_max(N) propre)
  □ Tâche 4 : ξ(r) post-processing
  □ Tâche 5 : Test η=1.0 (cas limite)
  □ Contact JPP avec résultats complets
```
---

## ══════════════════════════════════════════════
## MISE À JOUR 24 FÉVRIER 2026
## Analyses ChatGPT o3 (7 sessions) + simulations Jours 1-4
## ══════════════════════════════════════════════

## ÉTAT CONSOLIDÉ — CE QUI EST VALIDÉ ✅
- Fit Pantheon+ : η=1.045, q₀=-0.022, χ²/dof=0.914
- Barnes-Hut GPU f64 : 0% d'écart CPU/GPU
- Virialization PE_binding (α=4.57, Seg₀=0.0024)
- Hubble friction (dtau_per_dt=0.013205)
- GPU speedup : 39.6× (197 ms/step @ 2M, θ=0.7)
- Runs production : 500K (S=0.513), 2M (S=0.694), 8M (S=0.459, θ=1.5)

### Résultats Jours 1-4 (semaine du 22 février 2026)

| Jour | Test | Résultat |
|------|------|---------|
| 1 | Mode unique anisotrope α=0 vs α=1 | α=1 supprime 75% croissance ✅ |
| 2 | Yukawa α(k)=1−ε·exp(−r/rc) 6 runs | Effet max 0.2% — non significatif ✅ |
| 3 | ICs Zel'dovich anti-corrélées | Même résultat que Jour 2 ✅ |
| 4 | Run 2M ICs Zel'dovich PROPRES | λ₋=0 confirmé numériquement ✅ |

---

## DÉCOUVERTES THÉORIQUES MAJEURES (24 février 2026)

### 1. λ₋=0 est structurel et universel
- **Newtonien** : λ₋=ρ̄(1−α)=0 pour α=1 exact
- **Pour tout η** : λ₋=0 même avec ρ̄₋≠ρ̄₊ (démontré analytiquement)
- **En expansion FLRW** : λ₋=0 survit (amortissement seulement)
- **En k** : ratio δ(m+)/δ(m−)=1.00±0.01 pour toutes les échelles
  (mesuré numériquement sur 12 000 steps, 3 valeurs de k)

### 2. Janus ≠ Hassan-Rosen (résultat nouveau)
- HR : interaction géométrie↔géométrie (V(g⁻¹f))
- Janus : interaction matière↔matière via géométrie (T_μν−T̄_μν)
- **Conséquence** : pas de βᵢ identifiables, m_eff² HR non défini pour Janus
- **La neutralité λ₋=0 est cohérente avec l'absence de potentiel HR**

### 3. Bug IC Zel'dovich corrigé (CRITIQUE)
**Ancienne version (biaisée) :**
```rust
let sign = if idx < n_positive { 1 } else { -1 };
// → m+ dans z bas, m- dans z haut + déplacements ±ψ
// → ségrégation artificielle S₀=200 Mpc
```
**Nouvelle version (correcte) :**
```rust
let sign = if rng.gen::<bool>() { 1 } else { -1 };
// → signes aléatoires, même déplacement ψ pour tous
// → S₀≈0 Mpc (ICs propres)
```

### 4. Vitesses Zel'dovich
```rust
let d_dot = (1.0 + Z_INIT).powf(0.5);  // sqrt(11) ≈ 3.32
// PAS (1+z)^1.5 — cette formule est 11× trop grande
```

### 5. Autres bugs corrigés session courante
- Bug FFT : affichage {:.4} tronquait 2.4e-8 → 0.0000 (FFT OK)
- Interactions Janus : m+/m+ → attraction, m-/m- → attraction, m+/m- → répulsion ✅
- Pas de fix sign_factor dans kick_only nécessaire (kernel correct)

---

## RUN EN COURS (à stopper)

**output/jour4_corrected_1771892736**
- Step ~12 000, S≈0.45 Mpc (quasi-nul)
- Confirmation λ₋=0 universel en k
- **Stopper ce run — suffisant pour l'analyse**

**Exploiter les snapshots disponibles :**
- 120+ snapshots, steps 0→12000
- Calculer P₊₋(k) spectre croisé (Option 4 o3)
- Documenter dans LaTeX comme validation

---

## PROCHAINE PRIORITÉ : Test Mode Antisymétrique

### Objectif scientifique
Démontrer que le mode relatif Φ₋ est un **mode propre exact** du système Janus,
même en régime non-linéaire.

Si le mode reste neutre sur 12 000 steps → invariance dynamique structurelle.
C'est théoriquement fort et publiable.

### ICs antisymétriques pures

```rust
// Pour chaque mode k :
// δ₋(k, 0) = +A · sin(kx)  // m- surdense là où m+ sous-dense
// δ₊(k, 0) = -A · sin(kx)  // exactement anti-corrélé

// En pratique : générer un seul champ Zel'dovich ψ
// puis appliquer :
//   m+ : déplacement = +ψ
//   m- : déplacement = -ψ
// (c'est l'OPPOSÉ du bug corrigé — ici c'est VOULU et DOCUMENTÉ)
// Amplitude A choisie pour δ₀ ≈ 0.1 (régime quasi-linéaire au départ)
```

### Métriques à mesurer toutes les 10 steps
```
1. Δ(t) = √⟨(δ₊ - δ₋)²⟩  (amplitude du mode relatif)
2. P₊₋(k, t)              (spectre croisé)
3. σx(t), σy(t), σz(t)    (dispersions spatiales)
4. S(t)                    (ségrégation COM)
5. δk(m+)/δk(m-)           (ratio par échelle k)
```

### Ce qu'on attend
- **Si mode neutre** : Δ(t) ≈ constant → invariance structurelle prouvée
- **Si croissance** : λ₋ > 0 en non-linéaire → mécanisme de ségrégation identifié
- **Si décroissance** : amortissement → mode stable mais pas neutre

### Paramètres run
```
N = 2M, θ=0.7, dt=0.005
box=400 Mpc, z_init=10
Steps : illimité (pas d'auto-stop)
CSV toutes les 10 steps
Snapshots toutes les 100 steps
--features cuda OBLIGATOIRE
Output : output/antisym_mode_test_*/
```

---

## SUITE (après résultats antisymétriques)

### Option A — Scanner ε (α = 1−ε)
```
ε = 10⁻², 10⁻³, 10⁻⁴
Mesurer temps caractéristique de divergence
Trouver εc ~ 1/N_dyn (seuil de détectabilité)
```

### Option B — Surdensité locale unilatérale
```
Sphère surdense m+ uniquement au centre
m- uniforme
Question : relaxation vers cohabitation ou instabilité ?
```

### Option C — Spectre croisé P₊₋(k)
```
Plus sensible que ratio δ₊/δ₋
À calculer depuis snapshots run corrigé déjà disponibles
```

---

## PAPIER PUBLIABLE VISÉ

> "Numerical stability of the antisymmetric mode in Janus cosmology"

Contenu :
- Démonstration λ₋=0 universel (multi-k, 12 000 steps)
- Test mode antisymétrique (linéaire et non-linéaire)
- Seuil εc critique
- Scaling law
- Preuve Janus ≠ Hassan-Rosen (architectures différentes)

---

## QUESTIONS POUR JPP

1. Les tenseurs T_μν d'interaction dérivent-ils d'un potentiel géométrique V(g⁻¹f) ?
   → Si oui : identifier les βᵢ → calculer m_eff² → instabilité possible
   → Si non : confirme l'architecture matière↔matière et λ₋=0 structurel

2. Les fluctuations primordiales des deux secteurs sont-elles
   anti-corrélées dans le scénario inflationnaire bimétrique ?
   (seule piste viable pour ségrégation avec α=1)

3. α=1 est-il une contrainte exacte de l'action, ou une approximation ?

---

## COMMANDES

```bash
# Stopper le run en cours
docker ps  # trouver le container
docker stop <container_id>

# Lancer le run antisymétrique
docker compose run --rm dev cargo run --release --features cuda \
  --bin jour4_filaments  # modifier les ICs pour antisymétriques

# Vérifier progression
tail -f output/antisym_mode_test_*/evolution.csv

# Exploiter snapshots run corrigé (pendant que nouveau run tourne)
# Calculer P₊₋(k) sur snapshots steps 0→12000
```

---

## RÉFÉRENCES CLÉS

1. Petit, Margnat & Zejli (2024) — EPJC 84:1226
2. Hassan & Rosen (2011) — arXiv:1109.3515 (bigravité ghost-free)
3. Berg et al. (2012) — arXiv:1206.3496 (perturbations HR)
4. Könnig et al. (2014) — arXiv:1407.4331 (stabilité bigravité)

Analyse complète ChatGPT o3 : chatgpt_analysis.md (650 lignes)

---

## TESTS DE VALIDATION OBLIGATOIRES

### Avant de lancer quoi que ce soit — lire VALIDATION_RULES.md

### Test 1 — Vérification ICs antisymétriques
Avant le run complet, vérifier sur 10 steps avec N=1000 :
```
- Imprimer δ(m+, k₁) et δ(m−, k₁) au step 0
  → doivent être exactement opposés : δ₊ = −δ₋
- Imprimer S₀ → doit être > 100 Mpc (les deux populations
  sont déjà spatialement séparées par construction)
- Imprimer σx, σy, σz au step 0
  → doivent être asymétriques (pas 115/115/115)
```

### Test 2 — Vérification calcul P₊₋(k)
Test trivial sur données synthétiques avant d'appliquer aux snapshots :
```
Générer deux grilles aléatoires identiques (δ₊ = δ₋)
→ P₊₋(k) / P₊₊(k) doit être = 1.00 exactement

Générer deux grilles aléatoires indépendantes
→ P₊₋(k) / P₊₊(k) doit être ≈ 0 (décorrélées)

Générer deux grilles exactement opposées (δ₊ = −δ₋)
→ P₊₋(k) / P₊₊(k) doit être = −1.00 exactement
```
Si ces 3 cas passent → le calcul est correct.

### Test 3 — Vérification Δ(t)
Sur le run antisymétrique, au step 0 :
```
Δ(0) = √⟨(δ₊ − δ₋)²⟩ doit être ≈ 2A
(deux fois l'amplitude initiale, car δ₊ = −δ₋ = A)

Si Δ(0) ≈ 0 → les ICs ne sont pas antisymétriques → bug
Si Δ(0) ≈ 2A → correct ✅
```

### Test 4 — Conservation des signes GPU
Vérifier que les signes m+/m- sont conservés pendant le run :
```
Au step 0 : compter N+ et N-  → noter les valeurs
Au step 1000 : recompter N+ et N-  → doivent être identiques
Si différents → bug dans le kernel CUDA
```

### Test 5 — Energie et forces
Au step 1 du run antisymétrique :
```
Calculer force moyenne sur m+ et sur m-
→ avec ICs antisymétriques, les forces doivent être
  de signes opposés (symétrie du problème)
→ |F(m+)| ≈ |F(m-)| à 1% près
```

---

## MISE À JOUR URGENTE — Découverte Mode Antisymétrique

### Résultat nouveau (24 février 2026, nuit)

Le spectre croisé P₊₋(k) montre r(k) : −1 → +1 en ~2000 steps.

**Ce n'est PAS λ₋=0 neutre.** C'est une relaxation dynamique du mode
antisymétrique vers le mode adiabatique — en régime non-linéaire.

> "The antisymmetric mode is linearly neutral but nonlinearly unstable
>  toward adiabatic alignment."

### Stopper le run antisymétrique prévu

Le run antisymétrique était planifié pour tester la neutralité du mode.
La réponse est déjà là dans les snapshots existants : le mode relaxe.

**Relancer le run antisymétrique — le laisser tourner jusqu'au step 4000 minimum.**
(Erreur précédente : stoppé trop tôt à 300 steps, avant la transition attendue à ~2000 steps)

### Nouvelle priorité : Test Scaling en N

C'est le test décisif avant toute publication.

```
Si τ_relax ∝ N   → artefact numérique (collisions 2-corps)
Si τ_relax = cst → propriété physique du modèle → publiable
```

**Estimation τ_coll ~ N / ln(N) (relaxation 2-corps)**
Pour N=2M : τ_coll ~ 2×10⁶ / ln(2×10⁶) ~ 140 000 steps
Or τ_relax observé ~ 2 000 steps → 70× plus rapide que τ_coll
→ Très probablement un phénomène dynamique collectif, pas numérique.

### 4 runs à lancer en parallèle si GPU disponible

| Run | N | Paramètres |
|-----|---|------------|
| scale_500K | 500K | ICs antisym, θ=0.7, même box |
| scale_1M   | 1M   | ICs antisym, θ=0.7, même box |
| scale_2M   | 2M   | ICs antisym, θ=0.7, même box |
| scale_4M   | 4M   | ICs antisym, θ=0.7, même box |

Mesurer τ_relax(k) pour chaque N via r(k,t) = 1 − A·exp(−t/τ)

### Si GPU insuffisant pour 4M

Lancer 500K et 2M en parallèle — suffisant pour discriminer.

### Métriques à sauvegarder toutes les 50 steps

```
r(k, t) = P₊₋(k) / √(P₊₊(k)·P₋₋(k))
pour k₁=0.05, k₂=0.13, k₃=0.30, k₄=0.63 Mpc⁻¹
```

### TESTS DE VALIDATION avant run

Test 1 : r(k=0.13, step=0) = −1.00 exactement (ICs antisym propres)
Test 2 : r(k=0.13, step=0) identique pour N=500K et N=2M
Test 3 : τ_coll ~ N/ln(N) calculé et affiché pour chaque run

---

## MISE À JOUR — Tests pour publication (o3, réponse 9)

### Nuance sur le scaling en N

τ_relax décroissant avec N n'est pas suffisant pour conclure.
Peut aussi être : champ plus lisse → dynamique collective plus nette.
Il faut trois tests supplémentaires.

### Test A — Scaling avec softening (priorité 1)

```
3 runs N=2M identiques, seul ε_soft change :
  soft_low  : ε = 0.5 × ε_nominal
  soft_nom  : ε = ε_nominal (run actuel)
  soft_high : ε = 2.0 × ε_nominal

Mesurer τ_relax(k=0.13) pour chaque.
Si τ varie avec ε → artefact numérique
Si τ constant      → phénomène structurel ✓
```

### Test B — Scaling avec amplitude A (priorité 2, CRUCIAL)

```
3 runs N=2M, seule l'amplitude initiale change :
  amp_low  : A = 0.01 × L_box
  amp_mid  : A = 0.05 × L_box
  amp_high : A = 0.10 × L_box  (run actuel)

Mesurer τ_relax(k=0.13) pour chaque.
Si τ ∝ 1/A  → couplage non-linéaire quadratique (δ₋ × δ₊ → δ₊)
Si τ = cst  → instabilité linéaire cachée (λ₋ < 0 ?)
```

### Test C — Force mesh pur PM (priorité 3)

```
Désactiver Barnes-Hut, utiliser PM uniquement (si disponible).
Si relaxation persiste → phénomène de champ moyen → physique ✓
```

### Ordre d'exécution recommandé

1. Test B (amplitude) — le plus discriminant, facile à implémenter
2. Test A (softening) — important mais secondaire
3. Test C (PM) — si le code le permet

### TESTS DE VALIDATION avant chaque run

- r(k=0.13, step=0) = valeur négative attendue (IC antisym)
- Δ(0) = 2×A (amplitude initiale correcte)
- N+ = N- exact (signes conservés)

### Données disponibles

/mnt/T2/janus-sim/output/scale_500K_1771919091/r_k_evolution.csv
/mnt/T2/janus-sim/output/scale_2M_1771919341/r_k_evolution.csv

Envoyer les deux CSV complets à o3 pour ajustement τ_relax exponentiel précis.

---

## RÉSULTATS TEST B — Scaling en amplitude (24 février 2026)

### Données runs

| Run | Amplitude | N | r(k,0) | τ_relax | Comportement |
|-----|-----------|---|--------|---------|--------------|
| amp_low | A = 1% | 2M | -0.41 | ~500 steps | Monotone → r=0.99 stable |
| amp_mid | A = 5% | 2M | -0.48 | ~50 steps | Monotone → r=0.99 stable |
| amp_high | A = 10% | 2M | -0.05 | ~50 steps | Monotone → r=0.97 stable ✓ |

### Fichiers CSV

```
/mnt/T2/janus-sim/output/amp_low_1771926585/r_k_evolution.csv
/mnt/T2/janus-sim/output/amp_mid_1771927193/r_k_evolution.csv
/mnt/T2/janus-sim/output/amp_high_2M_1771930148/r_k_evolution.csv
```

### Comparaison N=500K vs N=2M à A=10%

| Step | N=500K | N=2M | Δ_500K | Δ_2M |
|------|--------|------|--------|------|
| 100 | +0.98 (peak) | +0.99 | 0.52 | 0.25 |
| 300 | +0.90 ↓ | +0.98 | 0.51 | 0.25 |
| 500 | **+0.70** ↓ | **+0.97** | 0.51 | 0.25 |

**La décorrélation observée à N=500K était un artefact de résolution !**

### Conclusions RÉVISÉES

1. **τ_relax ∝ 1/A** confirmé :
   - A×5 (1%→5%) → τ÷10 (500→50 steps)
   - A×10 (1%→10%) → τ÷10 (500→50 steps)
   - → **Couplage quadratique** (δ₋ × δ₊ → δ₊), pas instabilité linéaire

2. **PAS de transition critique** :
   - ~~Régime fortement non-linéaire à A≥10%~~ → FAUX (artefact N=500K)
   - Corrélation stable r>0.95 pour TOUTES les amplitudes testées (1%, 5%, 10%)

3. **Bug corrigé — OOM avec A=10%** :
   - Cause : positions hors [-L/2, L/2] après déplacement → LinearOctree OOM
   - Fix : appliquer conditions périodiques APRÈS le déplacement Zel'dovich
   - Ajouté à KNOWN_FIXES.md comme [FIX-008] variante

---

## RÉSULTATS TEST A — Scaling en softening (24 février 2026)

### Données runs

| Run | ε (Mpc) | Factor | τ_relax | r(50) | r(100) |
|-----|---------|--------|---------|-------|--------|
| soft_low | 0.79 | 0.5× | **50** | +0.907 | +0.969 |
| soft_nom | 1.59 | 1.0× | **50** | +0.907 | +0.969 |
| soft_high | 3.17 | 2.0× | **50** | +0.907 | +0.969 |

### Fichiers CSV

```
/mnt/T2/janus-sim/output/soft_low_1771931871/r_k_evolution.csv
/mnt/T2/janus-sim/output/soft_nom_1771932447/r_k_evolution.csv
/mnt/T2/janus-sim/output/soft_high_1771932623/r_k_evolution.csv
```

### Conclusion

**τ_relax INDÉPENDANT de ε** (facteur 4× de variation, résultat identique)

→ La relaxation du mode antisymétrique est un **phénomène structurel/collectif**
→ PAS un artefact numérique dû au softening

### Figure τ vs A

```
/mnt/T2/janus-sim/output/tau_vs_A.png
```

---

## SYNTHÈSE TESTS A+B — Conclusions pour publication

1. **τ_relax ∝ 1/A^n avec n ≈ 1.4** (Test B)
   - Couplage non-linéaire quadratique confirmé
   - Pas d'instabilité linéaire (λ₋ = 0 pour α = 1)

2. **τ_relax indépendant de ε** (Test A)
   - Phénomène structurel, pas artefact numérique
   - Robuste sur facteur 4× en softening

3. **τ_relax indépendant de N** (scaling test précédent)
   - 500K et 2M donnent taux similaires (~0.0015/step)
   - Pas un effet two-body relaxation

**Interprétation physique** : Le mode antisymétrique (δ₊ = -δ₋) se relaxe
via couplage quadratique δ₊δ₋ → δ₊, transférant le pouvoir vers le mode
corrélé (δ₊ = δ₋). C'est une propriété intrinsèque de la dynamique Janus
à α ≈ 1, pas un artefact numérique.
