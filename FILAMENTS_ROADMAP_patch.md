---

## COMPATIBILITÉ AVEC PETIT ET AL. 2024 (arXiv:2412.04644v3)
# Vérification effectuée le 2026-03-02

### ✅ Confirmé par le papier

- Lois d'interaction (éq. 107-108) : même signe → attraction, signe opposé → répulsion avec α=1 ✅
- Élimination du runaway via κ=−1 dans l'action (éq. 90-93) ✅
- Structure lacunaire émergente : conglomérats sphéroïdes de masses− qui confinent les masses+ (fig. 12-13) ✅
- η > 1 → E < 0 → accélération du secteur positif (éq. 96-98) ✅

### ⚠️ Hypothèses assumées (non dérivées du papier)

**H1 — ICs anti-corrélées (δ₋ = −δ₊)**
Le papier décrit une émergence spontanée de la structure lacunaire après découplement,
pas une anti-corrélation primordiale imposée. L'ICs anti-corrélée est une hypothèse de
travail pour forcer les filaments à apparaître dans la simulation. Elle est physiquement
motivée (les deux feuillets sont CPT-symétriques, donc les fluctuations pourraient être
anti-corrélées) mais n'est pas dérivée formellement. À assumer explicitement dans toute
publication.

**H2 — Ordre d'émergence des structures**
Le papier (éq. 109) montre que t̄_J < t_J : les masses négatives se structurent en premier,
puis confinent les positives. Les ICs anti-corrélées sautent cette phase dynamique
d'émergence. Acceptable pour un proof-of-concept, mais à noter : dans une simulation
"physiquement correcte", les blobs− devraient apparaître avant les filaments+.

**H3 — λ₋ = 0 (mode filamentaire neutre)**
L'analyse λ± de la matrice de couplage à deux fluides vient de notre analyse interne
(confirmée par o3), pas du papier 2024. Si utilisé dans une publication, dériver
explicitement ou référencer une source indépendante.

### Conséquence pour la simulation

Les phases A/B/C restent valides. La Phase A montrera si l'hypothèse H1 produit
une dynamique stable. Si oui, noter dans RUNS.md que la ségrégation observée est
partiellement imposée par les ICs (Seg_0 non nul) et partiellement dynamique (croissance de Seg après step 0).

La question ouverte pour JPP (amplitude physique de δ₋ ≠ δ₊) reste entière.
