# Scan des Conditions Initiales Janus
## Objectif : Trouver l'amplitude de perturbation initiale qui produit des structures visibles à z=0

---

## Contexte physique

Les simulations Janus avec CI uniformes aléatoires (δρ/ρ ~ 0.002%) restent en régime
linéaire à z=0. L'hypothèse : au rebond Janus (z≈4.5), l'univers n'est pas parfaitement
homogène — il a accumulé des perturbations depuis le régime de Milne (z>4.5).

On cherche l'amplitude δ_init telle que le régime non-linéaire (δρ/ρ ~ 1) soit atteint à z=0.

---

## Paramètres fixes (tous les runs)

```
μ = 19                    # valeur canonique (Ω_tot = 1)
N = 5M particules         # rapide, suffisant pour détecter les structures
Box = 500 Mpc             # compromis résolution/volume
Steps = 2000
G_code = 4.499e-15        # Mpc³ M_sun⁻¹ Gyr⁻²
Ω_b = 0.05
Expansion Janus activée   # t⁺ = α²(μ_p + ½sinh²μ_p)
z_init = 4.0
λ = 0                     # pas d'écrantage
```

---

## Paramètre varié : amplitude des perturbations initiales

Au lieu de positions purement aléatoires (bruit blanc Poisson),
perturber les positions autour d'une grille régulière :

```python
# Grille régulière de base
x_grid = np.linspace(-box/2, box/2, N^(1/3))

# Perturbation gaussienne d'amplitude δ_init
x_perturbed = x_grid + np.random.normal(0, δ_init * cell_size)
```

### Valeurs à tester

| Run | δ_init | Description |
|-----|--------|-------------|
| 1   | 0.002% | Baseline (bruit Poisson pur, actuel) |
| 2   | 0.1%   | ×50 plus fort |
| 3   | 1%     | Perturbations modérées |
| 4   | 5%     | Perturbations fortes |
| 5   | 10%    | Proche non-linéaire au départ |
| 6   | 20%    | Fortement perturbé |
| 7   | 50%    | Quasi non-linéaire au départ |

---

## Implémentation

### Modification des conditions initiales dans le code Rust

Dans `src/initial_conditions.rs`, remplacer la génération uniforme par :

```rust
// Paramètre d'amplitude (0.0 = uniforme pur, 1.0 = une cellule entière)
let delta_init: f64 = 0.01; // 1% — à faire varier

// Taille d'une cellule
let cell_size = box_size / (n_total as f64).cbrt();

// Position de base sur grille + perturbation gaussienne
let x = grid_x + rng.sample::<f64, _>(StandardNormal) * delta_init * cell_size;
let y = grid_y + rng.sample::<f64, _>(StandardNormal) * delta_init * cell_size;
let z = grid_z + rng.sample::<f64, _>(StandardNormal) * delta_init * cell_size;

// Assigner signe m+ ou m- selon μ
// N+ = N / (1 + μ), N- = N × μ / (1 + μ)
```

### Script de lancement

```bash
#!/bin/bash
# scan_ci_amplitude.sh

DELTAS=(0.002 0.1 1.0 5.0 10.0 20.0 50.0)
LABELS=("baseline" "x50" "moderate" "strong" "near_nl" "very_strong" "quasi_nl")

for i in "${!DELTAS[@]}"; do
    DELTA=${DELTAS[$i]}
    LABEL=${LABELS[$i]}
    
    echo "=== Run $((i+1))/7 : δ_init = ${DELTA}% ==="
    
    docker compose run --rm dev cargo run --release -- \
        --mu 19 \
        --n-particles 5000000 \
        --box-size 500.0 \
        --steps 2000 \
        --delta-init ${DELTA} \
        --output-dir output/ci_scan_delta${DELTA} \
        --snapshot-every 50 \
        --z-init 4.0
    
    echo "Run ${LABEL} terminé"
done
```

---

## Outputs par run

Pour chaque valeur de δ_init, générer :

### 1. Frames clés (5 snapshots)
```
z = 4.0, 2.5, 1.5, 0.5, 0.0
Format 7 panels 4K (comme scan μ grossier)
```

### 2. Métriques temporelles
```
time_series.csv :
step | z | t_Gyr | Diff/Pois | Corr(δ+,δ-) | ρ+_max/ρ̄+ | σ8
```

### 3. Observable clé : ρ+_max/ρ̄+
```
Si ρ+_max/ρ̄+ > 10 à z=0 → régime non-linéaire atteint → structures visibles
Si ρ+_max/ρ̄+ < 5  à z=0 → encore linéaire
```

### 4. Panel de synthèse
```
Plot : ρ+_max/ρ̄+(z=0) vs δ_init
→ Identifier le seuil δ* où les structures apparaissent
```

---

## Critère de succès

**Structures visuellement détectables** si à z=0 :
- ρ+_max/ρ̄+ > 10 (overdensité ×10)
- Diff/Pois > 5
- Corr(δ+,δ-) < -0.3
- Blobs ou filaments visibles dans la projection 2D

---

## Estimation temps

```
7 runs × ~25 min (5M, 500 Mpc, 2000 steps) = ~3h
+ Génération frames : ~1h
Total : ~4h
```

---

## Interprétation physique

Si δ* ≈ 1-5% : les perturbations au rebond Janus doivent être
significativement plus grandes que le bruit thermique pur.
Cela contraint la physique du régime de Milne pré-rebond.

Si δ* > 20% : le régime non-linéaire nécessite des CI
quasi-non-linéaires dès le départ — incompatible avec
un univers primordial lisse. Problème théorique.

Si δ* < 0.1% : nos simulations actuelles sont sous-résolues
en amplitude, pas en nombre de particules.

---

## Note pour JPP

Ce scan constitue une **contrainte observationnelle sur les CI au rebond** :
pour reproduire la toile cosmique observée dans le modèle Janus,
les perturbations au rebond z≈4.5 doivent avoir une amplitude δ* ∈ [X%, Y%].
C'est une prédiction testable de la phase de Milne pré-rebond.
