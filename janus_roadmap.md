# Feuille de route — Projet Janus
# Document pour Claude CLI
# Date : 20 février 2026
# Basé sur analyse de ChatGPT, Grok et Gemini

---

## CONTEXTE ET ÉTAT ACTUEL

### Ce qui est validé
- Fit Pantheon+ : η=1.045, q₀=-0.022, χ²/dof=0.914 ✅
- Barnes-Hut GPU f64 : 0% d'écart CPU/GPU ✅
- Bugs corrigés : rsqrt(), COM périodique, équations accélération ✅
- **Tâche 1 COMPLÈTE** : virialization PE_binding, COM référence commune ✅
- Seg₀ = 0.0024 (vs 0.49 avant correction) ✅
- KE/KE₀ = 1.0003 stable (vs 115 avant virialization) ✅

### Run production terminé — Hubble friction (2026-02-21)
| Paramètre | Valeur |
|-----------|--------|
| Run | 2026-02-21_run_hubble_mid |
| N | 500,000 |
| η | 1.045 |
| Steps | 3,600 |
| Runtime | 4h |
| KE/KE₀ final | **6.01** ✅ |
| Seg initiale | 0.24% |
| Seg finale | **14.5%** ✅ |

Vidéo générée : `janus_hubble_500k.mp4` (24 sec, 721 frames)

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

## ORDRE D'EXÉCUTION — ÉTAT ACTUEL

```
COMPLÉTÉ ✅ :
  ✅ Tâche 1 : Virialization (PE_binding, Seg₀=0.24%)
  ✅ Tâche 2 : Hubble friction (dtau_per_dt=0.013205, KE/KE₀=6.01)
  ✅ Parallèle A : Vidéo 3-panel (721 frames, 24 sec)

À FAIRE :
  □ Tâche 3 : Convergence en N (100K → 2M)
  □ Tâche 4 : ξ(r) en post-processing
  □ Tâche 5 : Test η=1.0 (cas limite)

CONTACT PETIT :
  ✅ Fit Pantheon+ (η=1.045, χ²/dof=0.914)
  ✅ Vidéo simulation avec expansion cosmologique
  □ Courbe ξ(r) qualitative
  □ Comparaison convergence en N
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
- Gemini 2.0 Flash — code virialization et Hubble friction
- ChatGPT o3 — analyse conditions initiales
- Grok 3 — analyse régime quasi-symétrique η≈1
