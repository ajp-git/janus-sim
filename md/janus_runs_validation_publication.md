# JANUS — Runs Validation et Publication
## Paramètres validés théoriquement et empiriquement
**η* = 0.990, λ* = 9.6 Mpc, z_act* = 3.0**  
Dérivation analytique : η_c = √δ/(1+√δ) ≈ 0.990 pour δ~10⁴  
Accord simulation/théorie confirmé par 2 IA indépendantes

---

## ÉTAPE 0 — Correction bug périodique (OBLIGATOIRE avant tout run)

Modifier `/mnt/T2/janus-sim/optim/filament_metrics_v2.py` :

```python
def periodic_bbox(cell_coords, box_size):
    """
    Bounding box avec correction périodique.
    Évite la surestimation des filaments traversant les bords.
    """
    lengths = []
    for dim in range(3):
        coords = cell_coords[:, dim]
        d_direct   = coords.max() - coords.min()
        d_periodic = box_size - d_direct
        lengths.append(min(d_direct, d_periodic))
    return lengths

def filament_length_periodic(cell_coords, box_size):
    return max(periodic_bbox(cell_coords, box_size))
```

Rescorer les runs existants avec la correction :
```bash
python3 optim/rescore_periodic.py \
  --runs output/trichotomie_gpu/tour2/ \
  --runs output/trichotomie_gpu/tour3/ \
  --box-size 150.0 \
  --output output/scores_corriges.csv

# Afficher le classement corrigé
# Confirmer que gagnant = η=0.990, λ=9.6, z=3.0
```

---

## RUN 1 — Validation 2M particules

```yaml
simulation:
  box_size_mpc: 200.0
  n_particles: 2000000
  n_steps: 2500
  z_start: 5.0
  z_end: 0.0
  seed: 42
  theta: 0.7

physics:
  eta: 0.990
  lambda_base_mpc: 9.6
  r_smooth_mpc: 1.92          # λ × 0.20
  lambda_floor: 0.01
  hubble_friction: true
  cross_force_activation:
    mode: sigmoid
    z_start: 3.0
    z_width: 0.5

pm_grid:
  n_cells: 512                # 512³ pour 2M particules dans 200 Mpc

output:
  dir: output/validation_2M/
  snapshot_redshifts: [4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0]
  save_velocities: true
  metrics_every_steps: 25
```

### Analyses à produire sur le run 2M

```
1. Carte densité z=0.5 (pic filaments) — m+ et m- séparés
2. Skeleton filaments inter-halos (critères stricts, correction périodique)
3. filament_detail.png — profil densité + composition + ségrégation interne
4. Évolution temporelle : S(z), ΔCOM(z), n_filaments(z)
5. Fonctions de corrélation g(r) — g++, g--, g+-
6. Champ de vitesses inter-halos
7. Profils NFW tous halos > 100k particules
8. Calcul Π' = (G × ρ_mean × λ²) / σ_v² × η
9. Calcul λ_Jeans à z=3.0 — vérifier λ_J ≈ 9.6 Mpc
```

### Critères pour passer au run 40M

```python
# Passer au run 40M si TOUTES ces conditions :
n_filaments_real  >= 5       # filaments avec correction périodique
length_max_real   >= 15.0    # Mpc, correction périodique appliquée
density_mean      >= 0.3     # ρ/ρ̄ dans le filament
aspect_ratio      >= 3.0     # filament allongé
coherent_flow     >= 0.20    # flux vers les halos (vitesses)

# NE PAS passer si :
# - filaments traversent les bords périodiques
# - density_mean < 0.1 (structure fantôme)
# - n_filaments = 0
```

Temps estimé : ~3-4h GPU (RTX 3060)

---

## RUN 2 — Publication 40M particules

**Ne lancer qu'après validation du run 2M.**

```yaml
simulation:
  box_size_mpc: 500.0
  n_particles: 40000000
  n_steps: 3000
  z_start: 5.0
  z_end: 0.0
  seed: 42
  theta: 0.5                  # Barnes-Hut plus précis pour run publication

physics:
  eta: 0.990
  lambda_base_mpc: 9.6
  r_smooth_mpc: 1.92
  lambda_floor: 0.01
  hubble_friction: true
  cross_force_activation:
    mode: sigmoid
    z_start: 3.0
    z_width: 0.5

pm_grid:
  n_cells: 1024               # OBLIGATOIRE pour 40M particules dans 500 Mpc
                              # Résolution : 500/1024 = 0.49 Mpc/cellule

output:
  dir: output/publication_40M/
  snapshot_redshifts: [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
  save_velocities: true
  metrics_every_steps: 50
```

### Vérification VRAM avant lancement

```bash
# 40M particules × 25 bytes = 1.0 GB positions+vitesses
# PM grid 1024³ × 4 bytes  = 4.0 GB
# Buffers forces, arbres    = 3-4 GB
# Total estimé              = 8-9 GB / 12 GB ✅

nvidia-smi --query-gpu=memory.free --format=csv
# Doit afficher > 10000 MiB libre avant lancement
```

### Analyses publication sur le run 40M

```
FIGURES PRINCIPALES (pour le preprint) :

Fig 1 — Carte densité m+ à z=0.5 (500 Mpc × 500 Mpc, projection 50 Mpc)
         + carte densité m- superposée
         → Vue d'ensemble du web Janus

Fig 2 — Zoom sur région filamentaire la plus dense
         Slice 50 Mpc × 50 Mpc × 20 Mpc
         → Détail de la structure

Fig 3 — Évolution temporelle S(z) et ΔCOM(z)
         z=5 → z=0, marqueur z=3 (activation)
         → Dynamique de ségrégation

Fig 4 — Fonctions de corrélation g(r) à z=0.5
         g++, g--, g+- sur 1-250 Mpc
         Comparaison avec ΛCDM (run référence)
         → Signature statistique de la ségrégation

Fig 5 — Spectre de puissance P+(k), P-(k), P_cross(k) à z=0
         Comparaison avec ΛCDM
         → Observable DESI/Euclid

Fig 6 — Diagramme de phase η_c vs λ
         Points empiriques + courbe théorique η_c = √δ/(1+√δ)
         → Validation de la théorie analytique

Fig 7 — Profils NFW des 10 halos les plus massifs
         r_s, concentration c, masse virielle
         → Comparaison avec simulations ΛCDM

MÉTRIQUES PUBLICATION :
  n_filaments_real   (correction périodique obligatoire)
  length_max_real    (Mpc)
  length_mean_real   (Mpc)
  density_mean       (ρ/ρ̄ dans filaments)
  composition        (% m+)
  segregation_internal (ratio m+/m- le long du filament)
  void_fraction      (fraction volume < 0.1 ρ̄)
  S_max et z(S_max)  (pic de ségrégation)
  ΔCOM_final         (Mpc)
  lambda_jeans_z3    (vérification λ_J ≈ λ* = 9.6 Mpc)
```

### Temps estimé run 40M

```
40M particules, 3000 steps, BOX=500 Mpc, PM 1024³
RTX 3060 12GB :
  Étape PM      : ~0.8s/step × 3000 = 40 min
  Étape BH      : ~1.5s/step × 3000 = 75 min
  Total estimé  : ~2h par 1000 steps = ~6-8h total

Lancer la nuit — ne pas interrompre.
Sauvegarder tous les snapshots sur /mnt/T2/ (espace requis ~50 GB)
```

---

## Règles absolues

```
1. Correction périodique active sur TOUS les calculs de longueur
2. GPU vérifié avant lancement (nvidia-smi > 50% utilization)
3. Ne pas lancer 40M sans avoir vu les figures du run 2M
4. Ne pas modifier η, λ, z_act — paramètres théoriquement validés
5. Sauvegarder le run 2M complet avant de lancer 40M
   (les snapshots 2M servent de référence pour vérifier 40M)
```

---

*Paramètres dérivés analytiquement :*  
*η_c = √δ/(1+√δ) ≈ 0.990 pour δ~10⁴*  
*λ_opt = λ_Jeans(z_act) ≈ 9.6 Mpc*  
*z_opt : σ_8(z) ≈ 0.2 → z ≈ 3.0*
