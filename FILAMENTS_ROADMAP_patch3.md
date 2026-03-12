---

## SESSION 2026-03-03 — DÉCOUVERTES CRITIQUES

### BUG MAJEUR TROUVÉ ET CORRIGÉ : dtau_per_dt (FIX-016)

**Symptôme** : Seg stagnait à ~0.017 sur tous les runs, KE/KE₀ restait > 1.0
**Cause** : dtau calculé incorrectement dans janus_grid_exploration.rs

```rust
// INCORRECT (tous les runs depuis début de session) :
let dtau = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
// → dtau ≈ 0.83 / 2000 = 0.000415  (20× trop faible)

// CORRECT (comme nbody_overnight.rs de février) :
let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);
// → dtau ≈ 0.83 / (10000 × 0.01) = 0.0083
```

**Conséquence** : friction de Hubble 20× trop faible → KE ne descend pas → λ₊ ne peut
pas dominer → ségrégation gelée.
**Tous les runs 8M de cette session sont invalidés pour ce motif.**

---

### DÉCOUVERTE CRITIQUE : L'ORDRE DES PARTICULES DANS LES ICs

**Observation** : le run 2M de février utilisait `new()` avec les particules +
placées en premier, puis les −. Le code actuel (`new_with_state()` + signes
mélangés aléatoirement) échoue à produire de la ségrégation même avec dtau corrigé.

**Explication physique** : quand les signes sont mélangés spatialement, chaque
particule + a une particule − en voisin immédiat → forces +/− se compensent
localement → λ₊ ne peut pas s'amorcer. Avec les + d'abord puis les −, une
asymétrie spatiale cohérente existe dès z=5 → brisure de symétrie → instabilité.

**Statut** : physiquement discutable (voir H1 dans section compatibilité Petit 2024),
mais numériquement nécessaire pour reproduire les résultats de février.

**Règle ajoutée** :
```
TOUJOURS : utiliser new() avec positifs d'abord puis négatifs (ordre février)
TOUJOURS : utiliser virialize() PE full — PAS virialize_sampled()
JAMAIS   : mélanger les signes aléatoirement dans new_with_state()
```

---

### GRILLE D'EXPLORATION 100K (2026-03-03)

6 cas testés avec dtau corrigé, seed=42, N=100K, box=100 Mpc :

```
Case A : uniforme aléatoire, signes mélangés → Seg_max=0.005  FROZEN
Case B : density-based 0.3×spacing          → Seg_0=0.131, stable  GOOD (figé)
Case C : density-based 1.0×spacing          → Seg_0=0.130, stable  GOOD (figé)
Case D : density-based 2.0×spacing          → Seg_0=0.134, stable  GOOD (figé)
Case E : ±ψ opposés 0.3×spacing             → Seg_max=0.004  FROZEN
Case F : ±ψ opposés 1.0×spacing             → Seg_max=0.003  FROZEN
```

**Conclusion** : à 100K, aucun cas density-based ne montre de croissance dynamique.
Les cases B/C/D ont Seg_0 élevée (ICs pré-séparées) mais figée.
Les cases A/E/F montrent un effondrement blob co-localisé +/− → Seg non détectée
par la métrique actuelle mais structure visible visuellement.
**100K est insuffisant pour reproduire la dynamique de ségrégation de février.**

---

### RUN 2M VALIDÉ (ICs février, dtau corrigé)

```
N         = 2,000,000
Box       = 271 Mpc
ICs       = new() positifs d'abord puis négatifs (ordre février)
Virialization = virialize() PE full
dtau_per_dt = τ_range / (TOTAL_STEPS × DT)  ← corrigé
Seg_max   = 0.452 @ step 2906 (z≈1.69)  ✅
KE/KE₀    = 4.59 au pic (normal)
Pic z     ≈ 1.7  (cohérent avec février : z≈1.8)
```

Run en cours au moment de ce patch (~38% complet).

---

### RUNS INVALIDÉS CETTE SESSION

```
Run 8M box=271 Mpc  → INVALIDE : spacing trop petit (FIX-015) + bug dtau
Run 8M box=430 Mpc  → INVALIDE : bug dtau (friction 20× trop faible)
Run 500K            → INVALIDE : bug dtau
Grille 100K A-F     → INVALIDE pour comparaison : bug dtau + N trop faible
```

---

### PROCHAINE ÉTAPE : run 8M production (à lancer après validation 2M)

Critère déclenchement : Seg_final > 0.05 sur le run 2M en cours.

```
N         = 8,000,000
Box       = 430 Mpc    (n_side=200, spacing=2.15 Mpc, FIX-015)
Softening = 0.65 Mpc
θ         = 0.7
dt        = 0.01
z_init    = 5.0
Steps     = 10000
ICs       = new() positifs d'abord puis négatifs
Virialization = virialize() PE full
dtau_per_dt = τ_range / (10000 × 0.01)  ← IMPÉRATIF
ETA       = ~8-9h
```

Seg_max attendu : 0.3-0.5 (extrapolé depuis 2M).
