# JANUS — Trichotomie v2 — Boîte 500 Mpc
## Exploration corrigée à la bonne échelle cosmologique
**Pourquoi recommencer :** les paramètres optimaux (η=0.99, λ=9.6 Mpc)
ont été trouvés dans des boîtes de 150-200 Mpc. Dans une boîte de 500 Mpc,
les modes longue longueur d'onde dominent et l'instabilité est beaucoup plus
forte. Les paramètres doivent être re-optimisés à la bonne échelle.

---

## Analyse préliminaire — ce qui change à 500 Mpc

### L'invariant Π recalibré

```
Π = η × (λ / L_filament)²   avec L_filament ~ 12 Mpc

Anciens runs (150 Mpc) : λ/L_box = 9.6/150 = 0.064
Nouveaux runs (500 Mpc) : λ/L_box = λ/500

Pour garder le même ratio λ/L_box = 0.064 :
  λ_equivalent = 0.064 × 500 = 32 Mpc ← proche de notre λ=35 Mpc ✅
```

### λ_Jeans aux nouveaux z_act

```
λ_J(z) = c_s / √(4πGρ_mean(z))   avec c_s ≈ 370 km/s

λ_J(z=1.5) ≈ 9.6 × (4.5/2.5)^(3/2) ≈ 9.6 × 2.5 ≈ 24 Mpc
λ_J(z=2.0) ≈ 9.6 × (4.5/3.0)^(3/2) ≈ 9.6 × 1.8 ≈ 17 Mpc
λ_J(z=2.5) ≈ 9.6 × (4.5/3.5)^(3/2) ≈ 9.6 × 1.4 ≈ 14 Mpc

→ λ ∈ {15, 25, 35} Mpc couvre bien ces valeurs ✅
```

### Note sur η > 1

⚠️ Nos analyses précédentes indiquent H² < 0 pour η > 1 dans les
équations de Friedmann couplées. Cependant, cette contrainte dépend
de l'implémentation exacte des équations de champ de JPP.

Les runs η ∈ [1.0, 1.3] sont inclus par demande explicite.
Si le code détecte H² < 0, il doit logger l'erreur et killer le run
proprement (pas de crash silencieux).

Vérifier dans le code :
```rust
if H_squared < 0.0 {
    log::error!("H² < 0 at z={} for η={}", z, eta);
    return Err("Non-physical Hubble parameter");
}
```

---

## Résolution progressive par tour

```
Tour 1 (exploration η)    : 500k  — topologie grossière, réponse rapide
Tour 2 (grille 3×3×3)    : 1M    — structure visible, filaments grossiers
Tour 3 (trichotomie)     : 2M    — filaments résolus, métriques fiables
Tour 4 (validation)      : 5M    — publication quality
Run final                : 40M   — run de publication

Espacement moyen par résolution dans 500 Mpc :
  500k  → 3.7 Mpc  (topologie OK, filaments < 5 Mpc invisibles)
  1M    → 2.9 Mpc  (filaments > 8 Mpc détectables)
  2M    → 2.3 Mpc  (filaments > 5 Mpc détectables)
  5M    → 1.7 Mpc  (filaments > 3 Mpc détectables)
  40M   → 0.9 Mpc  (publication)
```

---

## Paramètres communs à tous les runs

```yaml
simulation:
  box_size_mpc: 500.0        # FIXE — ne pas changer
  n_particles: 500000        # Tour 1 — voir tableau résolution
  n_steps: 800               # z=5 → ~z=0.5
  z_start: 5.0
  z_end: 0.0
  seed: 42
  theta: 0.7                 # standard cosmologique

physics:
  r_smooth_mpc: auto         # = lambda_base × 0.20
  lambda_floor: 0.01
  hubble_friction: true
  cross_force_asymmetry: 1.0
  cross_force_z_width: 0.5

pm_grid:
  n_cells: 128               # suffisant pour 500k/500Mpc
  k_min: 2

output:
  snapshot_every_steps: 200  # 4 snapshots par run (léger)
  snapshot_redshifts: [3.0, 2.0, 1.0, 0.0]
  save_velocities: false     # économiser espace
  metrics_every_steps: 20    # time_series.csv léger
```

### Gestion des snapshots

```bash
# Après analyse de chaque run :
# Si filament_score < 0.1 → supprimer les snapshots
# Garder uniquement time_series.csv et metrics_summary.json

if [ "$filament_score" -lt "0.1" ]; then
    rm -rf output/tour1/${run_name}/snapshots/
    echo "Snapshots supprimés pour ${run_name} (score faible)"
fi
```

---

## TOUR 1 — Exploration large η (20 runs)

### Objectif
Cartographier l'espace η sur une plage inexplorée [0.70, 1.30],
avec λ et z_act fixés aux valeurs médianes.

### Configuration fixe Tour 1
```yaml
lambda_base_mpc: 25.0    # médiane de {15, 25, 35}
z_act: 2.0               # médiane de {1.5, 2.0, 2.5}
```

### Grille η — 20 valeurs uniformes

| Run | η | Π (estimé) | Note |
|-----|---|------------|------|
| E01 | 0.70 | 0.30 | |
| E02 | 0.73 | 0.31 | |
| E03 | 0.77 | 0.33 | |
| E04 | 0.80 | 0.34 | |
| E05 | 0.83 | 0.36 | |
| E06 | 0.87 | 0.37 | |
| E07 | 0.90 | 0.39 | |
| E08 | 0.93 | 0.40 | |
| E09 | 0.97 | 0.42 | |
| E10 | 1.00 | 0.43 | ⚠️ vérifier H² |
| E11 | 1.03 | 0.44 | ⚠️ vérifier H² |
| E12 | 1.07 | 0.46 | ⚠️ vérifier H² |
| E13 | 1.10 | 0.47 | ⚠️ vérifier H² |
| E14 | 1.13 | 0.49 | ⚠️ vérifier H² |
| E15 | 1.17 | 0.50 | ⚠️ vérifier H² |
| E16 | 1.20 | 0.52 | ⚠️ vérifier H² |
| E17 | 1.23 | 0.53 | ⚠️ vérifier H² |
| E18 | 1.27 | 0.55 | ⚠️ vérifier H² |
| E19 | 1.30 | 0.56 | ⚠️ vérifier H² |
| E20 | 0.99 | 0.43 | référence ancien optimum |

### Early stopping Tour 1

```python
# Vérifier tous les 100 steps
def should_stop_early(metrics, step, box_size=500):
    # Kill si H² < 0 (détecté dans les logs)
    if metrics.h_squared < 0:
        return True, "H² non-physique"

    # Kill si explosion cinétique
    if metrics.ke_ratio > 1e6:
        return True, "Explosion KE"

    # Kill si ΔCOM > 40% de la boîte dès z > 3
    # (instabilité trop rapide, pas de filaments possibles)
    if metrics.z > 3.0 and metrics.dcom_mpc > 0.4 * box_size:
        return True, "Instabilité précoce massive"

    # Flag (pas kill) si pas de structure à z=2
    if metrics.z < 2.1 and metrics.z > 1.9:
        if metrics.n_halos_plus < 2:
            return False, "FLAG: structure pauvre à z=2"

    return False, "OK"
```

### Résultats attendus Tour 1

Après les 20 runs, tracer :
- filament\_score vs η
- ΔCOM\_max vs η
- n\_halos\_plus vs η
- Temps d'apparition de l'instabilité vs η

→ Identifier la plage η productive (filament\_score > 0.2)
→ Cette plage devient le centre du Tour 2

---

## TOUR 2 — Grille 3×3×3 (27 runs)

Centré sur η_optimal trouvé en Tour 1.

### Configuration

```
η     ∈ {η* - δη, η*, η* + δη}   # δη = 0.10
λ     ∈ {15, 25, 35} Mpc
z_act ∈ {1.5, 2.0, 2.5}
```

### Paramètres affinés
```yaml
n_particles: 1000000       # 1M — filaments > 8 Mpc détectables
n_steps: 1000
snapshot_every_steps: 250  # 4 snapshots
```

### Critères filaments stricts (inchangés)
```python
n_cells < 500
aspect_ratio > 3.0
longueur > 8 Mpc              # dans une boîte 500 Mpc
distance_halo > 5 Mpc
correction_periodique: True   # OBLIGATOIRE
```

### Early stopping Tour 2

```python
# Plus strict qu'en Tour 1
if metrics.z > 3.5 and metrics.dcom_mpc > 200:
    return True, "Dipole géant précoce — pas de filaments"

if step == 400 and filament_score_z2 == 0:
    return False, "FLAG: 0 filaments à z~2, continuer prudemment"
```

---

## TOUR 3 — Trichotomie infinie

Même logique que la version précédente, mais avec la correction :

```
```yaml
n_particles: 2000000       # 2M — filaments > 5 Mpc détectables
n_steps: 1500
snapshot_every_steps: 300  # 5 snapshots
```

Zoom factor initial : ±15%
Zoom factor suivant : ×0.65 à chaque tour
η borné entre 0.65 et min(η_max_physique, 1.35)
λ borné entre 10 et 50 Mpc
z_act borné entre 1.0 et 3.5
```

### Condition de convergence
```
Progression filament_score < 5% entre deux passes
OU Δη < 0.02 ET Δλ < 1 Mpc ET Δz < 0.1
OU 8 passes effectuées
```

### Tableau de bord après chaque passe
```
Passe N | η=X, λ=Y Mpc, z_act=Z
Score   : F (progression : +G%)
Top 3   : [liste]
Décision: CONTINUER / CONVERGENCE
```

---

## TOUR 4 — Validation haute résolution

Sur les paramètres convergés (η**, λ**, z**) :

```yaml
simulation:
  box_size_mpc: 500.0
  n_particles: 5000000      # 5M — résolution ×10
  n_steps: 2000
  theta: 0.7

output:
  snapshot_every_steps: 100  # 20 snapshots
  save_velocities: true      # pour mesurer v_relative
```

### Critères pour passer au run 40M
```
n_filaments_real >= 5
length_max_real >= 20 Mpc    # plus exigeant (boîte plus grande)
ΔCOM_max < 200 Mpc           # instabilité contrôlée
void_fraction > 0.20         # vrais voids
```

---

## Règles absolues

```
1. box_size = 500 Mpc FIXE dans tous les tours
   Ne jamais tester dans 150 ou 200 Mpc

2. Correction périodique sur toutes les longueurs de filaments

3. Si H² < 0 → logger + killer proprement (pas ignorer)

4. Supprimer les snapshots des runs avec filament_score < 0.10
   Garder toujours : time_series.csv + metrics_summary.json

5. Scorer sur z=0.5 (pic filaments confirmé)

6. Early stopping sur instabilité précoce (ΔCOM > 40% boîte à z>3)
   → ce régime ne donnera pas de filaments, inutile de continuer

7. Ne pas utiliser les anciens paramètres optimaux (η=0.99, λ=9.6)
   comme référence — ils étaient calibrés pour 150 Mpc
```

---

## Questions ouvertes à résoudre

Ces questions orienteront l'interprétation des résultats :

**Q1 — η > 1 est-il physique dans ce code ?**
Si les runs E10-E19 ne crashent pas avec H² < 0, c'est que
l'implémentation tolère η > 1. À documenter et signaler à JPP.

**Q2 — Existe-t-il un régime sans instabilité précoce ?**
L'instabilité démarrait à z=4.7 dans le run 40M.
Avec z_act = 1.5 et λ = 35 Mpc, le web ΛCDM a plus de temps.
Les filaments ΛCDM survivent-ils assez longtemps pour être détectés ?

**Q3 — Y a-t-il un η_c différent à 500 Mpc ?**
η_c = √δ/(1+√δ) dépend de δ dans les filaments.
Dans une boîte plus grande, les filaments sont-ils plus ou moins denses ?
→ η_c pourrait être différent de 0.990.

---

## Estimation temps GPU

```
Tour 1 (20 runs)  : 20 × ~15 min = ~5h
Tour 2 (27 runs)  : 27 × ~20 min = ~9h
Tour 3 (27 runs × N passes) : N × 9h

Total minimum     : ~1-2 jours GPU

Optimisations :
  - Early stopping agressif → -30% temps
  - Suppression snapshots   → -50% espace disque
  - theta=0.7               → ×3-4 vs theta=0.5
```

---

*Trichotomie v2 — boîte 500 Mpc*
*Lancée après résultats run 40M (instabilité à z=4.7)*
*Chaumes-en-Brie, mars 2026*
