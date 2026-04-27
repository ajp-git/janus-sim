# JANUS ZOOM-IN L1 — Spécification technique
## Run baryonique haute résolution
### Version 1.0 — Avril 2026

---

## CONTEXTE

Le run principal (janus_baryonic_calibrated, 500 Mpc, 10M particules)
a produit un collapse gravitationnel non résolu à z≈0.37
(ρ_max+ = 105 412, runaway numérique).

Ce run zoom-in repart depuis le snapshot step 4550 (z=0.459),
AVANT le collapse, avec une résolution ×10 dans la région du halo
dominant et la physique baryonique complète activée.

---

## SOURCE

```
Snapshot : snap_04550.bin
z_init   : 0.459
t_init   : 4.55 Gyr
Centre halo dominant : [-5.329, 11.171, -39.571] Mpc
ρ_max à ce step : ~87
t_ff halo : 3.4 Gyr  ← largement avant collapse
```

---

## ZONES DE RÉSOLUTION

### Zone Haute Résolution (HR)
```
Rayon : r < 8 Mpc autour du centre halo
Particules m+ : subdivisées ×10
  m_part_HR = 5.1×10¹¹ / 10 = 5.1×10¹⁰ M☉
N_HR_originales : ~8 014 m+
N_HR_subdivisées : ~80 140 m+
Particules m- : inchangées (4 dans la sphère)
```

### Zone Background (BG)
```
Rayon : r > 8 Mpc (reste de la boîte)
Particules m+ : inchangées, m_part_BG = 5.1×10¹¹ M☉
Particules m- : inchangées
Comportement : collisionless, gravité pure
→ fournit les forces de marée à grande échelle
```

---

## PARAMÈTRES SIMULATION

```
z_init = 0.459 → z_final = 0.0
dt     = 0.0005 Gyr
steps  = 18500
θ      = 0.7 (Barnes-Hut)
ε_HR   = 0.03 Mpc  (particules HR)
ε_BG   = 0.10 Mpc  (particules BG)
```

---

## PHYSIQUE BARYONIQUE — HR UNIQUEMENT

### ⚠️ CRITIQUE : s'applique UNIQUEMENT aux particules is_HR=1

### Cooling
```
Kernel : S&D93 tabulé CUDA natif (<2% erreur)
T_init : 10 000 K
T_floor : 1000 K  ← CORRIGÉ (était 100 K → SF runaway)
UV background : HM2012
Self-shielding : Rahmati 2013
```

### Formation Stellaire — Schmidt-Kennicutt Probabiliste
```
Critères :
  (1) overdensité locale / ρ_mean > 50
  (2) T < 10 000 K (gaz froid)
  (3) ∇·v < 0 (gaz convergent)

Formule Schmidt-Kennicutt (standard FIRE/EAGLE) :
  t_ff = sqrt(3π / 32Gρ_local)
  p_SF = 1 - exp(-ε_star × dt / t_ff)

  Tirer u = rand_uniform(0, 1)
  Si u < p_SF → former une étoile

ε_star = 0.01 (efficacité par t_ff)
Délai SN = 0.01 Gyr (10 Myr)

Note: Évite le SF runaway observé avec formule déterministe
```

### Feedback SN — CALIBRATION CORRIGÉE
```
Énergie physique par particule HR :
  N_SN = m_part_HR / 100 M☉ = 5×10⁸ supernovae
  E_SN_physique = N_SN × 10⁵¹ erg = 5×10⁵⁹ erg

Mode : cinétique (kicks de vitesse)
  v_kick = 50 km/s  ← CORRIGÉ (était 450 → dispersait tout)
  mass_loading η = 3
    (3× la masse stellaire formée est éjectée en vent)

Delayed cooling :
  Après un événement SN, désactiver le cooling
  sur les particules affectées pendant 0.01 Gyr (10 Myr)
  → Permet à l'énergie de se propager avant refroidissement

Note: 450 km/s était trop fort — dispersait ρ_HR à 0 en 100 steps
```

### Pourquoi ces paramètres (consensus 4 IA)
```
δ > 100 : évite SF dans gaz diffus non résolu
v_kick = 450 km/s : calibré sur énergie physique réelle
                    (4 IA convergent : 400-500 km/s)
η = 3 : standard EAGLE/IllustrisTNG à cette résolution
delayed cooling : empêche overcooling numérique
```

---

## PHYSIQUE BG — GRAVITÉ PURE

```
Particules BG (is_HR=0) :
  Pas de cooling
  Pas de SF
  Pas de feedback
  Gravité uniquement → forces de marée correctes
```

---

## OUTPUTS

```
Snapshots :
  Interval = 10 steps → 1850 snapshots
  Format JSNP v2 avec flag is_HR
  Espace estimé : ~10 MB/snap × 1850 = ~18 GB

Métriques CSV (toutes les 10 steps) :
  Colonnes : step, z, t_Gyr,
             rho_max_HR, rho_max_BG,
             N_stars_HR, SFR_HR,
             v_disp_HR, T_mean_HR,
             ratio_vrms, S_global

STOP automatique :
  Si rho_max_HR > 500 000 ET N_stars_HR = 0
    → SF ne démarre pas → stopper et diagnostiquer
  Si N_stars_HR > 1 000 000
    → SF runaway → stopper et diagnostiquer
```

---

## CRITÈRES DE SUCCÈS

```
step 200 (validation précoce) :
  ρ_max_HR > 10        ← gaz encore présent
  100 < N_stars < 10000 ← SF modérée
  1000 K < T_mean < 10000 K  ← cooling contrôlé
  SFR < 10¹² M☉/Gyr   ← ordre cosmique

z=0.35 : N_stars_HR > 0     ← SF démarre
z=0.25 : N_stars_HR > 1000  ← SF active
z=0.10 : rho_max_HR stable  ← pas de runaway
z=0.00 : galaxies visibles dans les frames
```

---

## COMMANDES DE LANCEMENT

```bash
# 1. Stopper le run gravitationnel actuel
docker stop $(docker ps -q --filter name=janus)

# 2. Compiler avec physique baryonique HR
cargo build --release --features cuda \
  --bin janus_zoom_l1_baryonic

# 3. Lancer
nohup docker compose run --rm dev \
  ./target/release/janus_zoom_l1_baryonic \
  --snap    output/janus_baryonic_calibrated/snapshots/snap_04550.bin \
  --center  -5.329,11.171,-39.571 \
  --r-hr    8.0 \
  --subdiv  10 \
  --out-dir output/janus_zoom_L1/ \
  > output/janus_zoom_L1/run.log 2>&1 &

# 4. Monitorer
tail -f output/janus_zoom_L1/run.log
```

---

## VISUALISATION

Utiliser le renderer 15-panels adapté au zoom :
- Rangée 1 : vue globale 50 Mpc (pas 400 Mpc)
- Rangée 2 : zoom HR r < 8 Mpc avec étoiles
- Rangée 3 : métriques scientifiques

Pas de conditions périodiques dans les vues —
la boîte est petite, pas d'artefact en X.

---

## PROCHAINE ÉTAPE — ZOOM L2 (si nécessaire)

Si collapse détecté dans la zone HR :
  → Identifier la région collapse (r < 1 Mpc)
  → Subdiviser ×10 → m_part_L2 = 5.1×10⁹ M☉
  → Relancer depuis le snapshot pre-collapse L2
  → Même physique baryonique, ε_L2 = 0.003 Mpc

---

## HISTORIQUE

| Date | Événement |
|------|-----------|
| Avril 2026 | Run principal terminé à z=0.36 (collapse runaway) |
| Avril 2026 | Analyse trajectoire halo → step 4550 optimal |
| Avril 2026 | Run L1 gravitationnel lancé par erreur → STOP |
| Avril 2026 | Ce document créé pour éviter l'oubli baryonique |

---

*Version 1.0 — Avril 2026*
*À donner à CLI à chaque nouveau lancement du zoom L1*
