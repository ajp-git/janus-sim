# ZOOM L1 V2 — Spécification complète
*Basé sur la synthèse de 4 IA (ChatGPT, DeepSeek, Grok, Gemini) — Avril 2026*

---

## Contexte

Run principal : 10M particules, 400 Mpc, μ=19, baryonique complet, terminé z=5→z=0.
Zoom L1 actuel : 261K particules, subdivision ×10, m_part=5×10¹⁰ M☉ → trop peu pour vidéo 4K.
Objectif : ~5M particules dans 50 Mpc, physique propre, vidéo 4K publication-ready.

Halo dominant identifié : center = [-5.329, 11.171, -39.571] Mpc, z=0.459.

---

## Méthode retenue : Splitting adaptatif avec subdivision m⁻ (recommandation Gemini)

Consensus 4 IA : les Zoom ICs Lagrangiennes sont le gold standard publication,
mais le splitting adaptatif par seuil de densité est le meilleur compromis
qualité/temps pour Janus, à condition de subdiviser aussi les m⁻.

**Point critique Janus (Gemini, ignoré par les 3 autres IA) :**
Si m⁺ ultra-résolus mais m⁻ environnantes restent grossières →
force de répulsion bimétrique numériquement bruitée (effet "marche d'escalier")
→ courbes de rotation faussées → résultats non publiables.
Solution : subdiviser m⁻ autour du halo au minimum ×8.

---

## Paramètres de subdivision

### Région source
- Snapshot : output/janus_baryonic_calibrated/snapshots/snap_04550.bin (z=0.459)
- Boîte extraite : 50 Mpc centrée sur [-5.329, 11.171, -39.571]
- N_source : ~261 000 particules (m⁺ + m⁻)

### Critères de subdivision m⁺ (baryons haute résolution)

| Zone | Rayon | Subdivision | m_part résultante |
|---|---|---|---|
| Cœur halo | r < 2 Mpc | ×50 | ~10⁹ M☉ |
| Halo intermédiaire | 2 < r < 8 Mpc | ×20 | ~2.5×10⁹ M☉ |
| Halo étendu | 8 < r < 20 Mpc | ×5 | ~10¹⁰ M☉ |
| Région externe | 20 < r < 25 Mpc | ×1 (inchangé) | 5×10¹⁰ M☉ |

### Critères de subdivision m⁻ (CRITIQUE pour Janus)

| Zone | Rayon | Subdivision | Justification |
|---|---|---|---|
| Autour du halo | r < 12 Mpc | ×8 minimum | Force répulsive propre |
| Région externe | r > 12 Mpc | ×1 (inchangé) | Pas d'effet sur SF |

**Règle absolue :** À chaque interface entre zones de résolution différente,
appliquer une zone de transition (rampe sur 1 Mpc) pour éviter
les artefacts gravitationnels aux bords.

---

## Implémentation du splitting

### Positions des filles
Pour chaque particule mère de position x₀ :
- Générer N_split filles disposées en sphère isotrope (Blue Noise)
- Rayon de dispersion : r_disp = h_sph / 3
  où h_sph = longueur de lissage SPH locale de la particule mère
- Si h_sph non disponible : r_disp = ε_softening / 2 = 0.015 Mpc

```
positions_filles = x₀ + r_disp × directions_blue_noise(N_split)
```

**NE PAS utiliser** de dispersion aléatoire uniforme → shot noise.
**Utiliser** Blue Noise pour garantir une distance minimale entre filles.

### Vitesses des filles
```
v_fille = v_mère + Δv_thermique
Δv_thermique ~ Gaussienne(σ = 1–3 km/s)
```
Très faible perturbation pour ne pas chauffer le gaz artificiellement.

### Masse des filles
```
m_fille = m_mère / N_split
```
Conservation exacte de la masse totale.

### Température / énergie interne
```
T_fille = T_mère  (conservée exactement)
u_fille = u_mère  (énergie interne conservée)
```

---

## Relaxation après splitting

**Obligatoire** pour éviter les chocs artificiels au démarrage.

```
Durée relaxation : 50 steps
dt_relaxation : 0.0001 Gyr (×5 plus petit que dt normal)
Formation stellaire : DÉSACTIVÉE pendant la relaxation
Feedback : DÉSACTIVÉ pendant la relaxation
Refroidissement radiatif : ACTIF (pour dissiper l'énergie de dispersion)
```

Critère de fin de relaxation :
- ΔE_total / E_total < 0.01 (conservation énergie à 1%)
- v_rms stable (variation < 2% sur 10 steps consécutifs)

---

## Paramètres du run v2

```
Source snapshot : snap_04550.bin (z=0.459)
Center          : [-5.329, 11.171, -39.571] Mpc
R_extract       : 50 Mpc
R_HR            : 8 Mpc (cœur haute résolution)
N_total_cible   : ~5 000 000 particules
dt              : 0.0003 Gyr
steps           : 18500 (z=0.459 → z=0)
θ (Barnes-Hut)  : 0.7
ε_softening     : adaptatif selon zone (0.01–0.03 Mpc)
```

### Softening adaptatif par zone

| Zone | ε |
|---|---|
| r < 2 Mpc (cœur) | 0.010 Mpc |
| 2–8 Mpc | 0.020 Mpc |
| 8–25 Mpc | 0.030 Mpc |

---

## Outputs

```
Snapshots     : toutes les 10 steps → ~1850 snapshots
Format        : JSNP v2 avec flags is_HR, is_star, zone_id
CSV métriques : toutes les 5 steps
Colonnes CSV  : step, z, t_Gyr, N_stars, rho_max_HR,
                SFR, v_disp, T_mean_HR, M_stars_total

Output dir    : output/janus_zoom_L1_v2/
```

---

## Critères de succès

```
Post-relaxation (step 50)  : ΔE/E < 1%, pas de choc visible
z=0.35                     : N_stars > 0 (SF démarre)
z=0.20                     : N_stars > 1000
z=0.10                     : ρ_max stable, pas de runaway SF
z=0.00                     : structures visibles, N_stars > 5000 cible
```

---

## STOP automatiques

```
Si ΔE/E > 10% pendant relaxation → STOP, diagnostic splitting
Si N_stars > 2 000 000          → STOP, SF runaway
Si ρ_max_HR > 10^7 ET N_stars=0  → STOP, SF ne démarre pas
```

---

## Visualisation

Utiliser zoom_l1_renderer_v2.py (validé) avec :
- Renderer fond noir pour vidéo
- Renderer fond blanc pour document
- render_2p5D_publication.py pour figure standalone

Vidéo finale : z=0.459 → z=0 (le run principal z=5→z=0.459
sera concaténé séparément avec renderer unifié — ZOOM_L1_V2_PART2.md)

---

## Commandes de lancement

```bash
# 1. Compiler le module splitting
cargo build --release --features cuda \
  --bin janus_zoom_l1_v2_adaptive_split

# 2. Lancer le splitting + relaxation + run
nohup docker compose run --rm dev \
  ./target/release/janus_zoom_l1_v2_adaptive_split \
  --snap     output/janus_baryonic_calibrated/snapshots/snap_04550.bin \
  --center   -5.329,11.171,-39.571 \
  --r-extract 50.0 \
  --r-hr     8.0 \
  --split-core 50 \
  --split-mid  20 \
  --split-ext  5 \
  --split-minus 8 \
  --relax-steps 50 \
  --steps    18500 \
  --dt       0.0003 \
  --out-dir  output/janus_zoom_L1_v2/ \
  > output/janus_zoom_L1_v2/run.log 2>&1 &

# 3. Monitorer
tail -f output/janus_zoom_L1_v2/run.log

# 4. Vérifier après relaxation (step 50)
grep "step=50" output/janus_zoom_L1_v2/run.log
```

---

## Historique

| Date | Événement |
|---|---|
| Avril 2026 | Zoom L1 v1 terminé : 261K particules, N★=2124, z=0→-0.11 |
| Avril 2026 | Analyse 4 IA (ChatGPT, DeepSeek, Grok, Gemini) |
| Avril 2026 | Décision : splitting adaptatif + subdivision m⁻ (Gemini) |
| Avril 2026 | Ce document créé |

---

*Version 1.0 — Avril 2026*
*Donner ce fichier à CLI avant tout lancement du zoom L1 v2*
