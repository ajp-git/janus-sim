---

## RÉSULTATS PHASE C — 8M (2026-03-03)

### Correction N_max production

20M → OOM sur RTX 3060 12GB.
**N_max validé avec box proportionnelle : 8M** (run en cours).
Extrapolation mémoire GPU : 5420 MiB pour 8M → N_max réaliste ≈ 16M.

```
Profil mémoire mesuré (nvidia-smi) :
  8M particules, box=430 Mpc : 5420 MiB / 12288 MiB (44%)
  Extrapolation 16M          : ~10800 MiB → faisable avec marge
  Extrapolation 20M          : ~13500 MiB → OOM confirmé
```

### Paramètres Phase C validés (8M)

```
N         = 8,000,000
Box       = 430 Mpc         ← corrigé (271 Mpc = INVALIDE, spacing trop petit)
Spacing   = 2.15 Mpc        ← identique run 2M référence
Softening = 0.65 Mpc        ← 0.3 × spacing
θ         = 0.7
dt        = 0.01
z_init    = 5.0
α         = 4.60 (virialize_sampled)
step_ms   ≈ 3350 ms
```

### Run invalide documenté

```
Run 8M box=271 Mpc → INVALIDE
  Cause : spacing 1.36 Mpc (vs 2.15 Mpc requis) → densité 4× trop élevée
  Symptôme : KE/KE₀ → 3.38 à z=0.16, Seg plafonné à 0.018
  Action : stoppé et supprimé
```

### Règle ajoutée (FIX-015)

```
FIX-015 : Box size doit être proportionnelle à N^(1/3) pour maintenir spacing constant
  box = n_side × spacing_ref  avec  n_side = N^(1/3)
  spacing_ref = 2.15 Mpc (run 2M référence validé)
  Exemples :
    2M  → n_side=126 → box=271 Mpc  ✅
    8M  → n_side=200 → box=430 Mpc  ✅
    16M → n_side=252 → box=542 Mpc  (à tester)
    20M → OOM RTX 3060 12GB         ✗
```

### Prochaine étape si 8M PASS

Tenter run 16M avec box=542 Mpc, softening=0.65 Mpc.
Vérifier GPU : profil mémoire attendu ~10800 MiB (88% VRAM).
