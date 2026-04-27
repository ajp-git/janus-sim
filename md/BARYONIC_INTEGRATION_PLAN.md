# Intégration physique baryonique dans janus_adaptive_zoom.rs
*Plan précis pour CLI — lire entièrement avant de coder*

---

## Contexte

`janus_adaptive_zoom.rs` a la mécanique du split adaptatif mais
pas la physique baryonique. `janus_baryonic_calibrated.rs` a la
physique baryonique mais pas les splits. On fusionne les deux.

---

## Fichiers source de référence

- Physique baryonique : `src/bin/janus_baryonic_calibrated.rs` lignes 356-395
- Interface GpuCooling : `src/cooling_gpu.rs` (module existant)

---

## Modifications à apporter — 5 étapes

### Étape 1 — Ajouter l'import

Dans les imports de `janus_adaptive_zoom.rs`, ajouter :

```rust
#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
```

### Étape 2 — Ajouter les constantes baryoniques

Après les constantes existantes (ligne ~132), ajouter :

```rust
// Physique baryonique
const T_INIT_PLUS: f64 = 10000.0;   // Température initiale m+ [K]
const T_FLOOR: f64 = 100.0;         // Température plancher [K]
```

### Étape 3 — Initialiser GpuCooling après le GPU sim

Après la création de `gpu_sim` (ligne ~797), ajouter :

```rust
// Initialiser la physique baryonique (cooling + SF)
let cuda_device = gpu_sim.device();
let n_plus_init = state.particles.iter().filter(|p| p.sign == 1).count();
let mut gpu_cooling = GpuCooling::new(
    cuda_device,
    n_plus_init,
    args.l_box,
    state.m_plus_base,
).expect("Failed to create GpuCooling");

let signs_plus: Vec<i32> = vec![1i32; n_plus_init];
gpu_cooling.init_from_temperature(T_INIT_PLUS, T_INIT_PLUS, &signs_plus)
    .expect("Failed to init cooling temperatures");
println!("  ✓ Physique baryonique initialisée (T_init = {} K)", T_INIT_PLUS);

let mut n_stars: usize = 0;
let mut sfr: f64 = 0.0;
```

### Étape 4 — Ajouter la physique dans la boucle principale

Après `gpu_sim.step_with_expansion_dkd_gpu(...)` (ligne ~920),
ajouter le bloc baryonique :

```rust
// ═══════════════════════════════════════════════════════
// PHYSIQUE BARYONIQUE (m+ uniquement, chaque step)
// ═══════════════════════════════════════════════════════
{
    let pos = gpu_sim.get_positions().unwrap();
    let signs_data = gpu_sim.signs();

    // Densités locales pour le refroidissement
    let overdensities = compute_local_overdensities(
        &pos, &signs_data, 32, args.l_box
    );

    // Convertir en unités du kernel
    let rho_to_nh = 2e-7 * (1.0 + z).powi(3);
    let densities: Vec<f64> = overdensities.iter()
        .map(|&od| od * rho_to_nh / 3.07e-17)
        .collect();

    // Refroidissement GPU
    gpu_cooling.upload_densities(&densities)
        .expect("Failed to upload densities");
    gpu_cooling.apply_cooling(args.dt_max, z)
        .expect("GPU cooling failed");

    // Formation stellaire
    let new_stars = gpu_cooling.apply_star_formation(args.dt_max)
        .unwrap_or(0);
    n_stars += new_stars as usize;
    sfr = (new_stars as f64) * state.m_plus_base / args.dt_max;

    if new_stars > 0 {
        println!("    ★ Step {}: {} nouvelles étoiles, N★={}", step, new_stars, n_stars);
    }
}
```

### Étape 5 — Adapter GpuCooling après les splits

Quand `n_new > 0` (après un split), le nombre de m+ change.
Il faut réinitialiser GpuCooling avec le nouveau n_plus :

```rust
if n_new > 0 {
    // ... code GPU existant ...

    // Réinitialiser GpuCooling avec le nouveau n_plus
    let n_plus_new = state.particles.iter().filter(|p| p.sign == 1).count();
    let cuda_device = gpu_sim.device();
    gpu_cooling = GpuCooling::new(
        cuda_device, n_plus_new, args.l_box, state.m_plus_base
    ).expect("Failed to recreate GpuCooling after split");
    let signs_plus_new: Vec<i32> = vec![1i32; n_plus_new];
    gpu_cooling.init_from_temperature(T_INIT_PLUS, T_INIT_PLUS, &signs_plus_new)
        .expect("Failed to reinit cooling");
}
```

### Étape 6 — Copier compute_local_overdensities

Copier la fonction `compute_local_overdensities` depuis
`janus_baryonic_calibrated.rs` (chercher avec grep) vers
`janus_adaptive_zoom.rs` — c'est une fonction pure, aucune dépendance.

---

## Ordre d'implémentation

```
1. Copier compute_local_overdensities (aucun risque)
2. Ajouter import + constantes
3. Ajouter init GpuCooling (étape 3)
4. cargo check → corriger les erreurs de compilation
5. Ajouter le bloc baryonique dans la boucle (étape 4)
6. cargo check → corriger
7. Ajouter réinit après split (étape 5)
8. cargo check → corriger
9. Test N_grid=10, 200 steps :
   - N_stars doit augmenter vers z=5
   - SFR > 0 dans le CSV
10. Si OK → GO production N_grid=215
```

---

## Points critiques

```
CRITIQUE 1 : GpuCooling.device()
  Vérifier que GpuNBodySimulation expose bien une méthode device()
  Si non, utiliser CudaDevice directement depuis cudarc.

  grep -n "fn device\|pub device\|CudaDevice" src/nbody_gpu.rs | head -5

CRITIQUE 2 : compute_local_overdensities signature
  La fonction prend (pos: &[f64], signs: &[i32], grid_size, box_size)
  Vérifier que la signature correspond dans baryonic_calibrated.rs :
  grep -n "fn compute_local_overdensities" src/bin/janus_baryonic_calibrated.rs

CRITIQUE 3 : N_stars dans le snapshot v3
  Le header SnapshotHeaderV3 a déjà le champ n_stars.
  Passer n_stars à save_snapshot() — déjà prévu dans la signature.

CRITIQUE 4 : Performance
  GpuCooling tourne sur GPU → négligeable en temps
  La réinit après split ajoute ~1s → acceptable
```

---

## Ne PAS faire

```
✗ Ne pas modifier GpuCooling ou cooling_gpu.rs
✗ Ne pas changer la logique de split
✗ Ne pas modifier le format snapshot v3
✗ Ne pas toucher à nbody_gpu.rs
```

---

*Ce plan donne à CLI les modifications exactes ligne par ligne.*
*Implémenter dans l'ordre des étapes, cargo check après chaque étape.*
