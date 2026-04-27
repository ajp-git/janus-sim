# JANUS — Trichotomie Infinie Filaments
## Document d'exécution pour Claude CLI
**Objectif :** Trouver les paramètres optimaux (η**, λ**, z_act**) maximisant les filaments cosmiques réels dans le modèle Janus  
**Critère unique :** filament_score (pas S_segregation)  
**GPU :** RTX 3060, docker avec --gpus all obligatoire

---

## ⚠️ CONTEXTE — Pourquoi on recommence

Les runs précédents (Phase 1 et Phase 2) tournaient en **CPU** (93.5% CPU, GPU à 0%).  
Tous les résultats sont invalides. On repart de zéro avec GPU vérifié.

---

## ÉTAPE 0 — Vérification GPU (OBLIGATOIRE avant tout run)

```bash
cd /mnt/T2/janus-sim

# 1. Compiler avec CUDA
cargo build --release --features cuda,cufft

# 2. Vérifier que docker a accès au GPU
docker compose run --rm dev nvidia-smi
# → Doit afficher RTX 3060

# 3. Smoke test GPU : 10k particules, 50 steps
docker compose run --rm dev cargo run \
  --release --features cuda,cufft \
  --bin janus_optim -- --config optim/smoke_test.yaml

# Pendant le smoke test, dans un autre terminal :
watch -n 2 nvidia-smi
# → GPU Utilization doit être > 50%
# → Temps/step doit être ~ 40ms (pas 400ms)
```

**Si GPU non confirmé → STOP. Diagnostiquer avant de continuer.**  
**Si GPU confirmé → lancer la trichotomie.**

---

## ÉTAPE 1 — Trichotomie infinie (boucle jusqu'à convergence)

### Principe général

```
Tour 1  : grille large 3×3×3 (27 runs) → identifier le bassin optimal
Tour 2+ : zoom ±20% autour du gagnant → affiner
Tour N  : zoom ±(20% × 0.6^(N-2)) → convergence progressive

La boucle continue TANT QUE :
  - filament_score progresse de > 5% entre deux tours
  - ET l'intervalle de zoom > résolution minimale

La boucle s'arrête quand :
  - filament_score progresse de < 5%  → CONVERGENCE
  - OU filament_score > 0.80          → OBJECTIF ATTEINT
  - OU 10 tours effectués             → LIMITE DE SÉCURITÉ
```

### Scoring filaments — définition figée

```python
def filament_score(metrics):
    """Score ∈ [0, 1]. Critère unique de sélection."""
    s_length  = min(metrics.length_max / 30.0, 1.0)   # objectif > 30 Mpc
    s_count   = min(metrics.n_filaments / 3.0, 1.0)   # objectif > 3 filaments
    s_density = min((metrics.density_mean - 1.0) / 0.5, 1.0)  # objectif > 1.5 ρ̄
    s_purity  = min(metrics.fraction_mplus / 0.8, 1.0)  # objectif > 80% m+
    s_flow    = min(metrics.coherent_flow / 0.5, 1.0)   # objectif > 0.5

    return (0.30 * s_length +
            0.25 * s_count  +
            0.20 * s_density +
            0.15 * s_purity  +
            0.10 * s_flow)
```

### Critères filaments stricts — ne jamais assouplir

```python
def is_real_filament(component):
    if component.n_cells > 500:        return False  # trop grand
    if component.aspect_ratio < 3.0:   return False  # pas allongé
    if component.length_mpc < 8.0:     return False  # trop court
    if component.dist_halo_mpc < 5.0:  return False  # trop proche d'un halo
    return True
```

### Snapshot de scoring

**Toujours scorer sur z=0.5 uniquement** (pic des filaments confirmé empiriquement).

---

## TOUR 1 — Exploration large (27 runs)

### Grille 3×3×3

| Paramètre | Valeur 1 | Valeur 2 | Valeur 3 |
|---|---|---|---|
| η | 0.80 | 0.88 | 0.95 |
| λ_base (Mpc) | 5 | 8 | 12 |
| z_act | 1.5 | 2.0 | 3.0 |

**→ 27 combinaisons, toutes avec Z1 sigmoid activé**

### Config commune Tour 1

```yaml
# Paramètres fixes pour tous les 27 runs Tour 1
simulation:
  n_particles: 500000
  n_steps: 600
  box_size_mpc: 150.0
  z_start: 5.0
  z_end: 0.0
  seed: 42

physics:
  r_smooth_mpc: auto   # = lambda_base × 0.20
  lambda_floor: 0.01
  cross_force_activation:
    mode: sigmoid
    z_width: 0.5
    # z_start = z_act du run

pm_grid:
  n_cells: 128

output:
  snapshot_redshifts: [2.0, 1.0, 0.5, 0.0]
  save_velocities: true
  metrics_every_steps: 50
```

### Commande de lancement Tour 1

```bash
python3 /mnt/T2/janus-sim/optim/trichotomy.py \
  --tour 1 \
  --eta 0.80 0.88 0.95 \
  --lambda 5 8 12 \
  --z-act 1.5 2.0 3.0 \
  --n-particles 500000 \
  --steps 600 \
  --score-at-z 0.5 \
  --output output/trichotomie_gpu/tour1/
```

### Résolution minimale des paramètres

```
Δη_min   = 0.01   (arrêter zoom si intervalle η < 0.01)
Δλ_min   = 0.5    (arrêter zoom si intervalle λ < 0.5 Mpc)
Δz_min   = 0.1    (arrêter zoom si intervalle z < 0.1)
```

---

## TOUR N ≥ 2 — Zoom trichotomie automatique

### Calcul automatique des paramètres

```python
def next_tour_params(winner, tour_number):
    """
    Génère les 9 runs du tour suivant par zoom trichotomie.
    
    winner = (eta_star, lambda_star, z_star, score_star)
    tour_number = numéro du tour à générer (2, 3, 4, ...)
    """
    # Facteur de zoom décroissant
    zoom = 0.20 * (0.60 ** (tour_number - 2))
    
    eta_star, lam_star, z_star, _ = winner
    
    # 9 runs : 3 axes × 3 valeurs, les 2 autres fixés au gagnant
    runs = []
    
    # Axe η (λ* et z* fixés)
    for eta in [eta_star * (1 - zoom), eta_star, eta_star * (1 + zoom)]:
        eta = max(0.50, min(0.99, eta))  # bornes physiques
        runs.append((eta, lam_star, z_star))
    
    # Axe λ (η* et z* fixés)
    for lam in [lam_star * (1 - zoom), lam_star, lam_star * (1 + zoom)]:
        lam = max(2.0, min(20.0, lam))   # bornes raisonnables
        if lam != lam_star:              # éviter doublon
            runs.append((eta_star, lam, z_star))
    
    # Axe z (η* et λ* fixés)
    for z in [z_star * (1 - zoom), z_star, z_star * (1 + zoom)]:
        z = max(1.0, min(4.0, z))        # bornes cosmologiques
        if z != z_star:                   # éviter doublon
            runs.append((eta_star, lam_star, z))
    
    return runs  # 7-9 runs selon les doublons éliminés


def n_particles_for_tour(tour_number):
    if tour_number <= 3:   return 500000,  600
    if tour_number <= 5:   return 1000000, 1200
    return 1000000, 1500
```

### Vérification résolution minimale

```python
def should_stop_zoom(winner_current, winner_previous):
    """Retourne True si le zoom est trop fin pour continuer."""
    delta_eta = abs(winner_current.eta - winner_previous.eta)
    delta_lam = abs(winner_current.lam - winner_previous.lam)
    delta_z   = abs(winner_current.z   - winner_previous.z)
    
    return (delta_eta < 0.01 and 
            delta_lam < 0.5  and 
            delta_z   < 0.1)
```

---

## Condition de convergence (vérifier après CHAQUE tour)

```python
def check_convergence(score_current, score_previous, tour):
    progress = (score_current - score_previous) / score_previous
    
    if score_current > 0.80:
        return "OBJECTIF ATTEINT → validation haute résolution"
    
    if progress < 0.05:
        return "CONVERGENCE → validation haute résolution"
    
    if tour >= 10:
        return "LIMITE SÉCURITÉ → validation haute résolution"
    
    return f"CONTINUER → Tour {tour + 1} (progression={progress:.1%})"
```

---

## Tableau de bord (afficher après CHAQUE tour)

```
═══════════════════════════════════════════════════════
TOUR [N] TERMINÉ
═══════════════════════════════════════════════════════
Gagnant    : η=[X], λ=[Y] Mpc, z_act=[Z]
Score      : [A] (progression vs tour précédent : +[B]%)
Filaments  : n=[C], L_max=[D] Mpc, ρ_mean=[E]×ρ̄
N runs     : [F] runs, [G] avec filaments réels

Top 3 :
  #1 η=[X1] λ=[Y1] z=[Z1] → score=[S1]
  #2 η=[X2] λ=[Y2] z=[Z2] → score=[S2]
  #3 η=[X3] λ=[Y3] z=[Z3] → score=[S3]

Décision   : [CONTINUER / CONVERGENCE / OBJECTIF / LIMITE]
Prochain   : Tour [N+1], zoom ±[zoom%]%, [N_part] particules
═══════════════════════════════════════════════════════
```

**Ne pas lancer le tour suivant sans afficher ce tableau.**

---

## ÉTAPE 2 — Validation haute résolution

### Déclenchée quand : convergence ou objectif atteint

**Run de confirmation (2M particules) :**

```yaml
simulation:
  n_particles: 2000000
  n_steps: 2000
  box_size_mpc: 200.0
  z_start: 5.0
  z_end: 0.0
  seed: 42
  # paramètres : gagnant de la trichotomie

pm_grid:
  n_cells: 256

output:
  snapshot_redshifts: [3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0]
  save_velocities: true
```

**Si filament_score confirmation > 0.50 → Run publication (20M) :**

```yaml
simulation:
  n_particles: 20000000
  n_steps: 3000
  box_size_mpc: 300.0

pm_grid:
  n_cells: 1024    # OBLIGATOIRE pour 20M particules
```

---

## Figures à produire sur le run gagnant final

```
output/trichotomie_gpu/gagnant_final/
├── density_z05.png              # carte densité au pic filaments
├── evolution_temporelle.png     # S(z) et ΔCOM(z)
├── skeleton_filaments_z05.png   # squelette inter-halos coloré
├── filament_detail.png          # profil densité + composition
├── correlation_gr.png           # g++(r), g--(r), g+-(r)
├── velocity_field.png           # flux de matière
└── metrics_summary.json         # toutes les métriques
```

---

## Règles absolues

```
1. GPU obligatoire
   → vérifier nvidia-smi avant chaque tour
   → GPU Utilization > 50% pendant les runs
   → temps/step ~ 40ms (pas 400ms)

2. Critères filaments stricts
   → n_cells < 500, AR > 3, L > 8 Mpc, d_halo > 5 Mpc
   → ne jamais assouplir même si 0 filament

3. Scorer sur z=0.5 uniquement

4. trichotomy.py décide les paramètres
   → ne pas choisir manuellement les valeurs des tours suivants

5. Afficher le tableau de bord après chaque tour
   → ne pas lancer le tour suivant sans l'afficher

6. Ne pas modifier le filament_score
   → les poids sont figés (0.30/0.25/0.20/0.15/0.10)

7. Si 0 filament dans Tour 1 :
   → élargir η vers [0.88, 0.92, 0.96, 0.99]
   → relancer Tour 1 avec z_act fixé à 2.0
   → ne pas changer les critères de détection
```

---

*Document trichotomie infinie filaments — Projet Janus GPU*  
*Paramètres physiques : Jean-Pierre Petit, modèle Janus bimétrique*  
*Critère : filament_score uniquement — pas S_segregation*
