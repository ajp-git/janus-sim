# EXPLORATION GRID — Recherche du bon paramètre ICs Janus
# Objectif : identifier les ICs qui produisent Seg_max > 0.1
# Durée totale estimée : 4-6h (6 runs 100K + récap + 1 run 2M)
# Dernière mise à jour : 2026-03-03

---

## CONTEXTE

Le run 8M box=430 Mpc a montré Seg_max=0.018 figé dès z=4.5.
Le run 2M référence (ICs uniformes aléatoires) avait S_max=0.694.
Hypothèse : les ICs density-based pré-satisfont l'état final → pas de dynamique.
But de cette grille : tester systématiquement les variantes d'ICs à faible coût.

---

## PARAMÈTRES COMMUNS À TOUS LES RUNS

```
N         = 100_000
Box       = 100 Mpc
Spacing   = 2.154 Mpc  (= 100 / 100^(1/3) × ajustement grille)
Softening = 0.65 Mpc   (0.3 × spacing)
θ         = 0.7
dt        = 0.01
z_init    = 5.0
Steps     = 2000        (z=5 → z=0)
virialize_sampled(N/1000)
Seed      = 42 (identique pour tous → comparaison équitable)
```

---

## LES 6 CAS

### Case A — ICs uniformes aléatoires (CONTRÔLE)
```
ICs      : positions aléatoires uniformes dans [-box/2, +box/2]
Signes   : aléatoires 50/50
Zel'dov  : AUCUN
Amplitude: —
Output   : output/grid_A_100k/
```
**Pourquoi** : c'est ce qui donnait S_max=0.694 sur le run 2M référence.
C'est le cas de référence absolu. Si A ne donne pas Seg > 0.1 à 100K,
le problème est ailleurs (résolution, virialization, η).

### Case B — Density-based amplitude standard (ACTUEL)
```
ICs      : grille + déplacement Zel'dovich
Signes   : basés sur δ(x) local (+ si surdensité, − si sous-densité)
Amplitude: 0.3 × spacing = 0.646 Mpc
Output   : output/grid_B_100k/
```
**Pourquoi** : reproduire le comportement du run 8M actuel à petite échelle.

### Case C — Density-based amplitude forte
```
ICs      : grille + déplacement Zel'dovich
Signes   : basés sur δ(x) local
Amplitude: 1.0 × spacing = 2.154 Mpc
Output   : output/grid_C_100k/
```
**Pourquoi** : tester si l'amplitude est trop faible dans B.

### Case D — Density-based amplitude très forte
```
ICs      : grille + déplacement Zel'dovich
Signes   : basés sur δ(x) local
Amplitude: 2.0 × spacing = 4.308 Mpc
Output   : output/grid_D_100k/
```
**Pourquoi** : pousser jusqu'au bord de la stabilité.
STOP si KE/KE₀ > 10 au step 5.

### Case E — ±ψ opposés amplitude standard (roadmap original)
```
ICs      : grille + déplacement Zel'dovich ±ψ selon signe
Signes   : aléatoires 50/50
Amplitude: 0.3 × spacing = 0.646 Mpc
Output   : output/grid_E_100k/
```
**Pourquoi** : tester l'approche originale de la roadmap (abandonnée trop tôt ?).

### Case F — ±ψ opposés amplitude forte
```
ICs      : grille + déplacement Zel'dovich ±ψ selon signe
Signes   : aléatoires 50/50
Amplitude: 1.0 × spacing = 2.154 Mpc
Output   : output/grid_F_100k/
```
**Pourquoi** : ±ψ avec plus de déplacement initial.

---

## ORDRE D'EXÉCUTION

```
1. nvidia-smi → GPU propre (0 MB utilisé) — OBLIGATOIRE
2. Lancer A → attendre fin → vérifier KE step 5
3. Lancer B → attendre fin
4. Lancer C → attendre fin
5. Lancer D → vérifier KE step 5 → si > 10 : STOP case D, passer à E
6. Lancer E → attendre fin
7. Lancer F → attendre fin
8. Générer les images pour chaque case (voir section VISUALISATION)
9. Sauvegarder tous les time_series.csv dans output/grid_summary/
10. Donner les 6 CSV à AJP pour récap
```

---

## CRITÈRES DE SUCCÈS PAR CASE

```
PASS   : Seg_max > 0.05  ET  KE/KE₀_max < 5.0
GOOD   : Seg_max > 0.10  ET  KE/KE₀_max < 3.0
EXCEL  : Seg_max > 0.30  ET  KE/KE₀_max < 2.0  → lancer 2M immédiatement
FAIL   : KE/KE₀ > 10 au step 5  → STOP, ne pas attendre la fin
FROZEN : Seg stagne à Seg_0 ± 0.005 sur tout le run  → noter, passer au suivant
```

---

## VISUALISATION — À GÉNÉRER POUR CHAQUE CASE

### Script : scripts/render_grid_case.py

Générer 3 types d'images pour chaque case, aux steps 0, 500, 1000, 2000 :

**Image 1 — Format 2.5D (vue grand public)**
```python
# Projection isométrique 2.5D, fond noir
# Masses+ : points bleus lumineux, taille ∝ densité locale
# Masses− : points rouges lumineux, taille ∝ densité locale
# Résolution : 1920×1080
# Fichier : output/grid_X_100k/frames/step_XXXX_25d.png
#
# Paramètres suggérés :
#   - Scatter plot avec alpha=0.3, s=1.5
#   - Colormap bleu : matplotlib 'Blues_r'
#   - Colormap rouge : matplotlib 'Reds_r'
#   - Fond : #0a0a0a
#   - Titre : "Janus 100K — Case X | z=X.XX | Seg=X.XXXX"
```

**Image 2 — Format densité ρ+ / ρ− (publication)**
```python
# Deux panneaux côte à côte, fond blanc
# Panneau gauche  : densité ρ+ projetée XY, colormap 'Blues'
# Panneau droit   : densité ρ− projetée XY, colormap 'Reds'
# Grille : 128×128 cellules, log(1 + ρ/ρ̄) normalisé
# Résolution : 2048×1024
# Fichier : output/grid_X_100k/frames/step_XXXX_dens.png
#
# Titre panneau : "Case X — Step XXXX | z=X.XX | Seg=X.XXXX | KE=X.XXX"
```

**Image 3 — Contraste différentiel (diagnostic)**
```python
# Panneau unique : (ρ+ − ρ−) / (ρ+ + ρ−) normalisé
# Colormap divergente : bleu=+, rouge=−, blanc=0
# Échelle : [-1, +1]
# Résolution : 1024×1024
# Fichier : output/grid_X_100k/frames/step_XXXX_contrast.png
```

### Récapitulatif visuel final

Générer une image mosaïque :
```python
# output/grid_summary/recap_seg_evolution.png
# 6 lignes (cases A→F), 5 colonnes (steps 0, 500, 1000, 1500, 2000)
# Format densité (Image 2) uniquement
# Résolution totale : 4096×2048
# Ajouter Seg et KE en titre de chaque vignette
```

---

## RÉCAP CSV À PRÉPARER POUR AJP

Après les 6 runs, créer `output/grid_summary/recap.csv` :

```
case,ic_type,amplitude_mpc,seg_0,seg_max,seg_max_z,ke_max,ke_final,verdict
A,uniform_random,0,X,X,X,X,X,PASS/FAIL/FROZEN
B,density_based,0.646,X,X,X,X,X,...
C,density_based,2.154,X,X,X,X,X,...
D,density_based,4.308,X,X,X,X,X,...
E,pm_opposed,0.646,X,X,X,X,X,...
F,pm_opposed,2.154,X,X,X,X,X,...
```

Uploader recap.csv + les images mosaïques pour décision.

---

## PHASE SUIVANTE — 2M (déclencher si ≥ 1 case GOOD)

Prendre le meilleur case (Seg_max le plus élevé avec KE < 3).
Lancer run 2M avec :

```
N         = 2,000,000
Box       = 271 Mpc   (spacing = 2.15 Mpc, identique référence)
Softening = 0.65 Mpc
θ         = 0.7
dt        = 0.01
z_init    = 5.0
Steps     = 5000
ICs       = [meilleur case de la grille]
Output    : output/filaments_2M_best/
```

Critère pour lancer 8M production : Seg_max > 0.3 sur le run 2M.

---

## RÈGLES ABSOLUES (rappel)

```
JAMAIS : docker stop $(docker ps -q)
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
TOUJOURS : vérifier KE au step 5 avant de continuer
TOUJOURS : softening = 0.3 × spacing minimum
TOUJOURS : virialize_sampled(N/1000)
```

---

## DOCUMENTATION

Après chaque run, ajouter dans RUNS.md :
```
### Run: grid_X_100k
Date: 2026-03-XX
ICs: [type] amplitude=[X] Mpc
Seg_0: X  Seg_max: X @ z=X  KE_max: X
Verdict: PASS/FAIL/FROZEN/EXCEL
```

Après la grille complète, mettre à jour FILAMENTS_ROADMAP.md avec un patch
résumant les résultats et la décision pour la Phase C finale.
