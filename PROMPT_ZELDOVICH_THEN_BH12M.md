# MISSION : Test Zel'dovich 500K → enchaîner BH 12M production
# Exécuter les deux runs EN SÉQUENCE ce soir sans intervention humaine.
# Lis ce fichier en entier avant toute action.

---

## CONTEXTE INDISPENSABLE

Lire avant de coder :
1. FILAMENTS_ROADMAP.md (en entier)
2. RUNS.md (entrées récentes)

Bugs critiques à ne jamais oublier :
- FIX-015 : box = n_side × 2.15 Mpc  avec  n_side = N^(1/3)
- FIX-016 : dtau_per_dt = tau_range / (TOTAL_STEPS × DT)  ← PAS tau_range/TOTAL_STEPS

ICs validées (run 2M février reproduit) :
- new() positifs d'abord puis négatifs
- virialize() PE full OU virialize_sampled(n=N/200)
- Ces ICs donnent Seg_max=0.45 sur 2M et Seg_max attendu >0.3 sur 12M

---

## RUN 1 — Test Zel'dovich 500K (~30 min)

### Objectif
Tester si les ICs Zel'dovich density-based + ordre février produisent
à la fois une ségrégation forte ET une morphologie filamentaire.
C'est une combinaison inédite : brisure de symétrie de février + structure P(k).

### Paramètres
```
N               = 500_000
Box             = 172 Mpc   (n_side=79, spacing=2.18 Mpc)
Softening       = 0.65 Mpc
θ               = 0.7
dt              = 0.01
z_init          = 5.0
Steps           = 2000
dtau_per_dt     = tau_range / (2000 × 0.01)  ← FIX-016
Snapshots       = NON
Output          : output/zeldovich_test_500k/
```

### ICs — combinaison Zel'dovich + ordre février

Construire les ICs ainsi (ordre CRITIQUE) :

```rust
// 1. Générer le champ de densité Zel'dovich δ(x) via FFT avec P(k)
// 2. Calculer les déplacements ψ(x) depuis δ(x)
// 3. Placer LES PARTICULES POSITIVES EN PREMIER :
//    Pour chaque point de grille où δ(x) > 0 :
//      pos = pos_grille + ψ(x)  → particule POSITIVE
// 4. Placer les particules négatives ensuite :
//    Pour chaque point de grille où δ(x) < 0 :
//      pos = pos_grille + ψ(x)  → particule NÉGATIVE
//    (même déplacement ψ — pas de ±ψ opposés)
// 5. virialize_sampled(n = N/200 = 2500)
// 6. dtau_per_dt correctement calculé (FIX-016)
```

L'idée : les particules + occupent les surdensités Zel'dovich,
les particules − occupent les sous-densités. L'ordre (+ d'abord)
reproduit la brisure de symétrie de février qui amorce λ₊.

### Critères d'évaluation

```
EXCEL    : Seg_max > 0.20  ET  morphologie filamentaire visible  → NOTER
GOOD     : Seg_max > 0.10  ET  KE/KE₀ < 10                      → NOTER
PASS     : Seg_max > 0.05  ET  KE/KE₀ < 20                      → NOTER
FROZEN   : Seg stagne à Seg_0 ± 0.01                             → NOTER
FAIL     : KE/KE₀ > 50 avant step 100                           → STOP
```

### Images à générer (APRÈS le run, avant de lancer le run 2)

Générer 3 images aux steps 0, 500, 1000, 2000 :

**Image A — Densité ρ+ / ρ− (publication)**
```python
# Deux panneaux côte à côte
# Panneau gauche  : densité ρ+ projetée XY, colormap 'Blues', log normalisé
# Panneau droit   : densité ρ− projetée XY, colormap 'Reds', log normalisé
# Grille : 256×256 cellules
# Résolution : 2048×1024
# Fichier : output/zeldovich_test_500k/dens_step_XXXX.png
```

**Image B — Contraste (ρ+ − ρ−) / (ρ+ + ρ−)**
```python
# Colormap divergente bleu/rouge, échelle [-1, +1]
# Résolution : 1024×1024
# Fichier : output/zeldovich_test_500k/contrast_step_XXXX.png
```

**Image C — Vue 2.5D fond noir**
```python
# Fond #0a0a0a
# Masses+ : points bleus, alpha=0.3, s=2
# Masses− : points rouges, alpha=0.3, s=2
# Résolution : 1920×1080
# Fichier : output/zeldovich_test_500k/view25d_step_XXXX.png
```

Uploader les images du step 2000 + time_series.csv avant de lancer le run 2.

---

## RUN 2 — Production BH 12M (~29h)

Lancer IMMÉDIATEMENT après la fin du run 1 et la génération des images.
Ne pas attendre de retour humain — enchaîner automatiquement.

### Paramètres
```
N               = 12_000_000
Box             = 492 Mpc   (n_side=229, spacing=2.15 Mpc)
Softening       = 0.65 Mpc
θ               = 0.7
dt              = 0.01
z_init          = 5.0
Steps           = 20000
ICs             = new() positifs d'abord puis négatifs
Virialization   = virialize_sampled(n=80000)  ← 0.5% de N/2
dtau_per_dt     = tau_range / (20000 × 0.01)  ← FIX-016 IMPÉRATIF
SNAPSHOT_INTERVAL = 20   → 1000 snapshots → ~192 GB
Output          = /mnt/T2/janus-sim/output/production_bh_12m/
```

### Format snapshot binaire (compact)
```rust
// Fichier : snap_XXXXXX.bin  (zero-padded step number)
// Header  : [n_particles: u64, step: u64, reserved: u64]  = 24 bytes
// Data    : n_particles × [x: f32, y: f32, z: f32, sign: f32]  = 16 bytes/particule
//           sign = +1.0 pour masse+, -1.0 pour masse−
// Taille  : 24 + 12M × 16 = ~192 MB/snapshot
// Total   : 1000 snapshots × 192 MB = ~192 GB
```

### Checklist avant lancement run 2
```
□ nvidia-smi → GPU propre (run 1 terminé, 0 MB résiduel)
□ df -h /mnt/T2/ → > 200 GB disponibles
□ dtau_per_dt vérifié : tau_range / (20000 × 0.01)
□ ICs = new() positifs d'abord
□ virialize_sampled(n=80000) — PAS virialize() PE full
□ SNAPSHOT_INTERVAL = 20
□ KE/KE₀ step 5 < 1.05
□ git push origin main
□ Sauvegarder container ID
```

### Milestones à logger dans RUNS.md pendant le run
```
Step 100   (z≈4.8) : KE doit être < 0.95  (friction active)
Step ~2800 (z≈2.2) : onset ségrégation attendu
Step ~4200 (z≈1.7) : pic ségrégation attendu (Seg > 0.2)
Step 20000 (z=0.0) : run complet
```

---

## DOCUMENTATION

### Après run 1, ajouter dans RUNS.md :
```
### Run: zeldovich_500k_combined
Date: 2026-03-04
ICs: Zel'dovich density-based + ordre février (+ d'abord)
Seg_0: X  Seg_max: X @ z=X  KE_max: X
Verdict: [EXCEL/GOOD/PASS/FROZEN/FAIL]
Morphologie: [filamentaire/blob/uniforme]
```

### Après lancement run 2, ajouter dans RUNS.md :
```
### Run: production_bh_12m
Date: 2026-03-04
N=12M, Box=492 Mpc, Steps=20000, Snapshots=1000
ICs: new() positifs d'abord + virialize_sampled(80000)
dtau_per_dt: tau_range / (20000 × 0.01)
Container: [ID]
ETA: ~29h
```

---

## RÈGLES ABSOLUES

```
JAMAIS  : docker stop $(docker ps -q)  — autres containers !
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
TOUJOURS : vérifier KE/KE₀ step 5 avant de continuer
JAMAIS  : modifier dtau sans vérifier FIX-016
```
