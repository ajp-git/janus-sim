# MISSION : Test TreePM + ICs février → objectif 40-50M particules
# Lis ce fichier en entier avant toute action.

---

## CONTEXTE

Le Barnes-Hut pur est limité à ~12M particules sur RTX 3060 (10284 MiB).
Le TreePM divise la VRAM par 3-4 en éliminant l'octree global →
objectif 40-50M particules.

Note historique : "TreePM instable avec Zel'dovich" (roadmap) ne s'applique
PAS aux ICs uniformes aléatoires (ICs février). À retester maintenant que
FIX-016 (bug dtau) est corrigé.

Les deux bugs critiques à ne jamais oublier :
- FIX-015 : box = n_side × 2.15 Mpc  avec  n_side = N^(1/3)
- FIX-016 : dtau_per_dt = tau_range / (TOTAL_STEPS × DT)

---

## ÉTAPE 1 — Test de stabilité TreePM 500K (30 min)

Avant tout, vérifier que TreePM + ICs février est stable.

```
Binaire   : janus_60m_treepm.rs (ou équivalent TreePM existant)
N         = 500_000
Box       = 172 Mpc  (n_side=79, spacing≈2.17 Mpc)
Softening = 0.65 Mpc
θ         = 0.7
dt        = 0.01
z_init    = 5.0
Steps     = 2000
ICs       = new() positifs d'abord puis négatifs (ICs février)
Virialization = virialize() PE full
dtau_per_dt = tau_range / (2000 × 0.01)  ← FIX-016
Snapshots = NON
Output    : output/treepm_500k_test/
```

Critères PASS :
```
Step 5   : KE/KE₀ ∈ [0.95, 1.05]  ← stable
Step 100 : KE/KE₀ décroît (< 0.95) ← friction Hubble active
Step 500 : Seg commence à croître
Step 2000: Seg_max > 0.05
KE/KE₀ ne dépasse jamais 10.0
```

Critère FAIL :
```
KE/KE₀ > 10 avant step 100  → STOP, TreePM instable avec ces ICs
```

Uploader time_series.csv après le run.

---

## ÉTAPE 2 — Mesurer le N_max TreePM (si étape 1 PASS)

Lancer un test 100 steps pour mesurer la VRAM à différents N :

```
Tester dans l'ordre : 20M → 30M → 40M → 50M
Arrêter quand VRAM > 11 GB
Box = n_side × 2.15 Mpc pour chaque N
ICs = new() positifs d'abord puis négatifs
Steps = 100 (juste pour mesurer)
```

| N   | n_side | Box     | VRAM estimée |
|-----|--------|---------|--------------|
| 20M | 271    | 583 Mpc | ~?           |
| 30M | 310    | 667 Mpc | ~?           |
| 40M | 342    | 735 Mpc | ~?           |
| 50M | 368    | 791 Mpc | ~?           |

Mesurer nvidia-smi au step 50 (distribution en cours de structuration →
cas le plus défavorable pour l'octree local TreePM).
Noter VRAM et step_ms pour chaque N testé.

---

## ÉTAPE 3 — Run validation 2M TreePM (si étape 1 PASS)

Valider que TreePM reproduit la physique de février avant le grand run.

```
N         = 2_000_000
Box       = 271 Mpc
Softening = 0.65 Mpc
Steps     = 5000
ICs       = new() positifs d'abord puis négatifs
dtau_per_dt corrigé (FIX-016)
```

Critère PASS : Seg_max > 0.3 avec pic à z≈1.5-2.0
Si PASS → lancer le run production N_max TreePM.

---

## ÉTAPE 4 — Run production N_max TreePM + snapshots

Paramètres à compléter après étape 2 :

```
N               = [N_max mesuré]
Box             = n_side × 2.15 Mpc
Softening       = 0.65 Mpc
θ               = 0.7
dt              = 0.01
z_init          = 5.0
Steps           = 10000
ICs             = new() positifs d'abord puis négatifs
Virialization   = virialize() PE full
dtau_per_dt     = tau_range / (10000 × 0.01)  ← FIX-016 IMPÉRATIF
SNAPSHOT_INTERVAL = 20
Output          = /mnt/T2/janus-sim/output/production_treepm_NM/
```

Vérifier espace disque avant lancement :
```
N × 16 bytes × (10000/20) snapshots
Exemple 40M : 40M × 16 × 500 = 320 GB  ← ajuster SNAPSHOT_INTERVAL si besoin
```

Si espace insuffisant, augmenter SNAPSHOT_INTERVAL à 50 ou 100.

---

## CHECKLIST AVANT CHAQUE LANCEMENT

```
□ nvidia-smi → GPU propre (0 MB utilisé)
□ df -h /mnt/T2/ → espace disque suffisant
□ dtau_per_dt vérifié dans le code (FIX-016)
□ ICs = new() positifs d'abord (PAS signes mélangés)
□ Virialization = virialize() PE full
□ KE/KE₀ au step 5 < 1.05
□ git push origin main
```

---

## RÈGLES ABSOLUES

```
JAMAIS  : docker stop $(docker ps -q)
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
JAMAIS  : lancer production sans valider stabilité à petit N d'abord
```

---

## LIVRABLES ATTENDUS

Après chaque étape, uploader :
- time_series.csv du run
- Log des VRAM mesurées (étape 2)
- Confirmation go/no-go avant étape suivante

Les snapshots .bin seront traités par script Python externe.
CLI ne génère pas d'images.
