# Tâche : Zoom simulation Janus — région interface m−/m+

## Contexte
Run principal terminé : 15M particules, boîte 500 Mpc, 4000 steps (z=5 → z=0.22).
Ségrégation plateau à Seg=0.40, snap_004000.bin disponible.
Les halos macroscopiques se sont formés mais sans filaments inter-halos visibles
(k_min=3 → modes ≥167 Mpc — filaments à 5-30 Mpc non semés dans les ICs).

Objectif en deux phases :
- **Phase 0** : Explorer 20 runs courts (100K particules, ~30min chacun) pour
  identifier les paramètres optimaux de la zoom simulation
- **Phase 1** : Zoom simulation réel sur snap_001500.bin avec les paramètres validés

## Fichiers disponibles
- Snapshot zoom : `snap_001500.bin` (z=1.634)
  → Point de départ optimal : halos formés, dynamique restante z=1.634→0
    pour développer les filaments dans la sous-boîte
- Snapshot récent : `snap_004000.bin` (z=0.225) — pour identifier la région zoom
  Répertoire : `/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/`
  Format .bin : header u64 N, puis N×28 bytes float32 [x,y,z,vx,vy,vz,mass_sign]
- Code source simulation : `/mnt/T2/janus-sim/` (Rust/CUDA)
- Paramètres run principal : ε=0.25, η=1.045, H=0.01, θ=0.5, BOX=500 Mpc

---

## PHASE 0 — Exploration 20 runs pour valider les paramètres zoom

### Objectif
Avant de lancer le vrai zoom (coûteux), scanner l'espace des paramètres
avec des runs courts 100K particules, boîte 50 Mpc, démarrant depuis z=5
avec ICs Zel'dovich haute fréquence. Identifier quelle combinaison produit
le plus de filaments mesurables.

### Paramètres à explorer (Latin Hypercube Sampling)

| Paramètre    | Plage          | Justification                           |
|--------------|----------------|-----------------------------------------|
| k_min_zoom   | 5 – 50         | Longueur d'onde min des ICs (filaments) |
| epsilon      | 0.02 – 0.20 Mpc| Softening (résolution force)            |
| amplitude_IC | 0.1 – 1.0      | Amplitude perturbations Zel'dovich      |
| box_zoom     | 30 – 80 Mpc    | Taille de la sous-boîte                 |
| N_particles  | 50K – 500K     | Nombre de particules test               |

Paramètres **fixés** pour tous les runs :
```
eta         = 1.045   (Janus strict)
H_friction  = 0.01
theta_bh    = 0.5
z_init      = 5.0    (ICs fraîches, pas de snapshot source)
z_final     = 0.0
dt          = 0.01
steps       = 2000   (~30-60min sur RTX 3060)
```

### Script `run_zoom_exploration.py`

Écrire ce script qui :

1. **Génère 20 configurations LHS** couvrant l'espace des paramètres ci-dessus
   (utiliser `scipy.stats.qmc.LatinHypercube`)

2. **Pour chaque configuration**, génère :
   - ICs Zel'dovich dans une boîte `box_zoom` Mpc avec `k_min_zoom` modes minimum
   - Fichier de config pour le code Rust/CUDA Janus
   - Répertoire de sortie `zoom_exploration/run_XXX/`

3. **Lance les runs séquentiellement** (un seul GPU disponible)
   en loggant le temps par run

4. **Analyse automatique après chaque run** :
   - Fraction de cellules filament (T-web, Hessian densité m−)
   - Ségrégation finale Seg
   - σ_P finale
   - KE/KE₀ (stabilité numérique — rejeter si >3.0)
   - Fraction de cellules vides (voids Janus)

5. **Tableau de résultats trié** par score composite :
   ```
   score = 0.5 × frac_filament + 0.3 × Seg + 0.2 × (1 - frac_void)
   ```

6. **Affiche le top 3** avec tous les paramètres pour Phase 1

### Métrique T-web pour la détection des filaments

Dans chaque run terminé, calculer le Hessian de densité m− sur grille 32³ :
```python
# Hessian de ρ_m− → valeurs propres λ1 ≥ λ2 ≥ λ3
# Classification :
#   void     : λ1 < 0
#   sheet    : λ1 ≥ 0, λ2 < 0
#   filament : λ1 ≥ 0, λ2 ≥ 0, λ3 < 0
#   node     : λ1 ≥ 0, λ2 ≥ 0, λ3 ≥ 0
```

Seuil λ = 0 (ou légèrement positif pour réduire le bruit).

### Sortie attendue Phase 0

```
=== EXPLORATION ZOOM — 20 RUNS ===
Run  | box  | k_min | ε     | amp  | N    | Filaments | Seg  | Score | Status
-----|------|-------|-------|------|------|-----------|------|-------|-------
001  | 50   | 10    | 0.05  | 0.3  | 100K | 42.3%    | 0.38 | 0.32  | OK
002  | 30   | 20    | 0.08  | 0.5  | 200K | 61.7%    | 0.41 | 0.47  | OK
...

TOP 3 CONFIGURATIONS :
1. Run 002 : box=30, k_min=20, ε=0.08, amp=0.5, N=200K → score=0.47
2. ...
```

---

## PHASE 1 — Zoom simulation réel (après validation Phase 0)

À lancer seulement après avoir les paramètres optimaux de Phase 0.

### Étape 1 — Identifier la région zoom sur snap_004000.bin

Écrire `find_zoom_region.py` qui :
1. Charge `snap_004000.bin` (z=0.225, ségrégation maximale → halos les plus nets)
2. Calcule la densité m− et m+ sur une grille 64³
3. Identifie les centres de halos m− et m+ (pics de densité)
4. Sélectionne la paire (halo_m−, halo_m+) la plus proche
5. Définit la région zoom : milieu entre ces deux centres,
   boîte de taille `box_zoom_optimal` (issue de Phase 0)
6. Génère `zoom_region.json`

Sortie :
```
Halo m− centres : [(x1,y1,z1), ...]
Halo m+ centres : [(x2,y2,z2), ...]
Meilleure interface : m−=(x,y,z) ↔ m+=(x,y,z), dist=D Mpc
Région zoom : centre=(cx,cy,cz), taille=BOX_OPTIMAL Mpc
N particules dans la région (snap_001500) : XXXX
```

### Étape 2 — Extraire et enrichir (snap_001500.bin)

Écrire `extract_and_split.py` qui :
1. Charge `zoom_region.json` + `snap_001500.bin` (z=1.634)
   → on extrait depuis z=1.634 pour laisser la dynamique se développer jusqu'à z=0
2. Extrait les particules dans la région zoom
3. Applique **particle splitting** (facteur issu de Phase 0 : ×4 ou ×8) :
   - Chaque particule → N_filles
   - Masse fille = masse mère / N_filles (signe conservé)
   - Position fille = position mère + offset gaussien σ=0.1×ε_split
   - Vitesse fille = vitesse mère (conservation exacte)
4. Ajoute perturbations Zel'dovich haute fréquence :
   ```
   k_min = k_min_optimal  (issu Phase 0)
   amplitude = amp_optimal × σ_Zel(z=1.634)
   ```
5. Sauvegarde ICs zoom au format .bin
6. Vérifie conservation de masse : |Σmasse_avant - Σmasse_après| < 0.01%

### Étape 3 — Configuration et lancement

Créer `zoom_sim_config.toml` avec les paramètres optimaux de Phase 0 :
```
box_size          = <box_optimal>
N_particles       = <N_split>
z_init            = 1.634
z_final           = 0.0
epsilon           = <epsilon_optimal>
eta               = 1.045
H_friction        = 0.01
theta_bh          = 0.5
dt                = <dt_optimal>
snapshot_interval = 100
output_dir        = /mnt/T2/janus-sim/output/zoom_<box>Mpc/
```

### Étape 4 — Validation post-lancement

Vérifier au step 10 :
- KE/KE_init < 2.0 (stabilité)
- N particules conservées (pas de fuite aux bords périodiques)
- σ_v cohérent avec run principal à z=1.634

---

## Notes importantes

- **Bords périodiques** obligatoires dans la boîte zoom
- **Contexte gravitationnel** ignoré (halos voisins à >100 Mpc — acceptable)
- **Si Phase 0 ne montre aucun filament** → tester k_min plus élevé (50-100)
  ou boîte plus petite (20 Mpc)
- **Si KE explose** dans Phase 0 → epsilon trop petit, exclure ces configs
- Documenter toutes les décisions dans `zoom_simulation_log.md`
- Phase 0 : ~10h GPU total (20 runs × 30min)
- Phase 1 : ~5-15h GPU selon N_split et box_optimal
