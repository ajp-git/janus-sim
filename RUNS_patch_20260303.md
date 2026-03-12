---

## Runs session 2026-03-03

### Run: anticorr_8M_box271 — INVALIDE
Date: 2026-03-03
Status: INVALIDE — stoppé
Cause: spacing=1.36 Mpc (box trop petite) + bug dtau friction 20× trop faible
Symptôme: KE/KE₀ → 3.38 à z=0, Seg figé à 0.018
Action: supprimé

### Run: anticorr_8M_box430_v1 — INVALIDE
Date: 2026-03-03
Status: INVALIDE — complété mais résultats non exploitables
Cause: bug dtau (friction de Hubble 20× trop faible)
Résultats: N=8M, box=430 Mpc, 10000 steps, z=5→0
  KE/KE₀_max=1.030 (stable mais friction insuffisante)
  Seg_max=0.017 (figé — pas de dynamique)
Leçon: stabilité numérique ≠ physique correcte

### Run: grid_exploration_100K_A-F — INFORMATIF
Date: 2026-03-03
Status: complété
N=100K, box=100 Mpc, 2000 steps, 6 variantes ICs
Bug dtau présent mais partiellement corrigé en cours de session
Résultats: voir section EXPLORATION GRID dans FILAMENTS_ROADMAP.md
Conclusion: 100K insuffisant, ICs density-based figées, uniforme aléatoire
  produit effondrement blob co-localisé (Seg métrique trompeuse à cette résolution)

### Run: ref_2M_icsfevrier — EN COURS ✅
Date: 2026-03-03
Status: running (~38% au moment du patch)
N=2,000,000 | Box=271 Mpc | ICs=new() positifs d'abord | virialize() PE full
dtau_per_dt corrigé : τ_range / (TOTAL_STEPS × DT)
Résultats partiels:
  Seg_max=0.452 @ step 2906 (z≈1.69)  ✅
  KE/KE₀=4.59 au pic (normal)
  Pic z≈1.7 cohérent avec run février (z≈1.8)
Verdict attendu: EXCEL si Seg_final > 0.05
Prochaine action: lancer 8M box=430 avec mêmes ICs si EXCEL
