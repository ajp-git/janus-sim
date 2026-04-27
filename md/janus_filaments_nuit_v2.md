# JANUS — Exploration Filaments v2 : Nuit 20-21 mars 2026
## Plan révisé après analyse par 4 IA (Gemini, ChatGPT, Grok, DeepSeek)
**Durée disponible :** 8h GPU  
**Matériel :** RTX 3060, 12 Go VRAM  
**Objectif unique :** filaments cosmiques > 6 Mpc, coexistant avec ségrégation S > 0.05

---

## Synthèse des 4 IA — Ce qui a changé

### Consensus fort (4/4)
- **Z1 (activation tardive z=2) est le meilleur candidat.** 3/4 IAs lui donnent >70% de chances. À mettre en Phase A, pas en hors-boîte.
- **A5 (BOX=500 Mpc, 200k) est trop peu résolu.** Espacement ~9.3 Mpc : un filament de 10 Mpc = 1 particule. Remonter à 400k minimum.
- **Ajouter un run ΛCDM de référence** (η=0, pas de force croisée). Donne le filament_score naturel comme baseline.

### Contribution unique de chaque IA

**ChatGPT — invariant physique Π (le plus utile) :**
```
Π = η × (λ_base / L_filament)²   avec L_filament ~ 10-15 Mpc

Π >> 1  →  dipole dominant        (pas de filaments)
Π ~  1  →  régime de transition
Π << 1  →  filaments possibles
```
Application immédiate :
```
A3 (η=0.30, λ=40) : Π = 0.30 × (40/12)² = 3.3  ❌ dipole
A5 (η=0.88, λ=40) : Π = 0.88 × (40/12)² = 9.8  ❌ dipole géant
A1 (η=0.88, λ=3)  : Π = 0.88 × (3/12)²  = 0.06 ✅ filaments probables
A4 (η=0.50, λ=15) : Π = 0.50 × (15/12)² = 0.78 ⚠️ transition
```
→ **A3 et A6 sont éliminés.** Π trop élevé, temps GPU gaspillé.

**Gemini — régularisation du floor (Y4) :**
```
λ_eff(x) = λ_base / √(ρ_local/ρ_mean + ε)

Au lieu du floor brut (discontinuité de force) :
  → λ_eff_max = λ_base / √ε   (dans les voids profonds)
  → ε = 0.10 → λ_eff_max = 3.16 × λ_base
  → Empêche les voids de "souffler" les filaments adjacents
```
C'est une modification mineure du code (un paramètre), impact potentiel majeur.

**Grok — asymétrie 2× et Z1 en Phase A :**
- Z1 doit être dans Phase A, pas hors-boîte
- Asymétrie F_cross : m− reçoit 2× la force que m+ → m− expulsé plus vite, filaments m+ préservés
- Phase C : un seul run 2000 steps plutôt que deux

**DeepSeek — résolution et squelette 3D :**
- 500k/600 steps pour Phase A (pas 200k/500)
- Détection filaments par squelette 3D (`skimage.morphology.skeletonize_3d`) plus fiable que slices 2D

---

## Invariant Π — filtre de décision

**Règle avant de lancer tout run :**
```python
def filament_feasible(eta, lambda_base, l_filament=12.0):
    pi = eta * (lambda_base / l_filament) ** 2
    if pi > 3.0:
        return "❌ DIPOLE — ne pas lancer"
    elif pi > 0.8:
        return "⚠️ TRANSITION — lancer si temps disponible"
    else:
        return "✅ FILAMENTS POSSIBLES — priorité haute"
```

Application à tous les runs de la nuit :

| Run | η    | λ_base | Π     | Verdict     |
|-----|------|--------|-------|-------------|
| A0  | 0.00 | 0 Mpc  | 0.00  | ✅ ΛCDM pur |
| A1  | 0.88 | 3 Mpc  | 0.06  | ✅ Priorité |
| A2  | 0.88 | 8 Mpc  | 0.43  | ✅ Priorité |
| A4  | 0.50 | 15 Mpc | 0.78  | ⚠️ Limite  |
| A5  | 0.88 | 40 Mpc | 9.8   | ❌ Éliminé  |
| A7  | 0.88 | 20 Mpc | 2.44  | ⚠️ Limite  |
| Z1  | 0.88 | 40 Mpc | —     | ✅ Spécial  |
| Y4  | 0.88 | 20 Mpc | 2.44→0.8 | ✅ Régularisé |

Note Z1 : Π non applicable (la force croisée est désactivée pendant la formation des filaments).

---

## Phase A révisée — 6 runs × ~25 min = 2h30

**Paramètres communs :** N=500k, steps=600, seed=42, z=5→0

### A0 — Référence ΛCDM (ajout IA)
```yaml
eta: 0.0               # PAS de masse négative
lambda_base_mpc: 0.0   # PAS de force croisée
r_smooth_mpc: 0.0
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
```
**Objectif :** mesurer filament_score naturel. Attendu : longueur_moy ~15-20 Mpc, score ~0.5-0.7. C'est le plafond qu'on ne dépassera probablement pas avec Janus.

---

### A1 — λ très court (Π=0.06)
```yaml
eta: 0.88
lambda_base_mpc: 3.0
r_smooth_mpc: 1.5
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
```
**Prédit par Gemini :** filaments probables, ségrégation faible. Les m− forment des "micro-bulles" dans les voids d'un web ΛCDM-like.

---

### A2 — λ intermédiaire court (Π=0.43)
```yaml
eta: 0.88
lambda_base_mpc: 8.0
r_smooth_mpc: 3.0
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
```
**Zone de transition.** Si A1 a des filaments et A4 n'en a pas, A2 localise la frontière.

---

### A4 — Compromis η/λ (Π=0.78)
```yaml
eta: 0.50
lambda_base_mpc: 15.0
r_smooth_mpc: 5.0
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
```
**Prédit par Gemini comme régime de coexistence :** η ≈ 0.5-0.7 + λ ≈ 10-15 Mpc.

---

### A7 — Asymétrie 2× (Grok)
```yaml
eta: 0.88
lambda_base_mpc: 20.0
r_smooth_mpc: 5.0
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
# NOUVEAU PARAMÈTRE :
cross_force_asymmetry: 2.0   # m- reçoit 2× la force que m+
```
**Logique :** les m− sont expulsés 2× plus vite → libèrent l'espace pour que les filaments m+ se consolident. Nécessite 1 ligne de code dans le kernel.

**Implémentation dans le kernel CUDA :**
```cuda
// Force sur m+ : standard
float f_on_plus = force_mag * 1.0f;
// Force sur m- : amplifiée
float f_on_minus = force_mag * config.cross_force_asymmetry;
```

---

### Z1 — Activation tardive z=2 (consensus 4/4)
```yaml
eta: 0.88
lambda_base_mpc: 40.0
r_smooth_mpc: 8.0
box_size_mpc: 150.0
n_particles: 500000
n_steps: 600
# NOUVEAU PARAMÈTRE :
cross_force_activation:
  mode: "sigmoid"          # progression douce (ChatGPT)
  z_start: 2.0             # début de l'activation
  z_width: 0.5             # demi-largeur de la transition
```

**Implémentation sigmoid (ChatGPT) :**
```rust
// Facteur d'activation de la force croisée
fn cross_force_factor(z: f64, params: &JanusParams) -> f64 {
    let z0 = params.cross_force_z_start;  // 2.0
    let dz = params.cross_force_z_width;  // 0.5
    // sigmoid : 0 pour z >> z0, 1 pour z << z0
    1.0 / (1.0 + ((z - z0) / dz).exp())
}
```
**Résultat attendu :** web cosmique formé entre z=5 et z=2, répulsion Janus activée progressivement entre z=2 et z=1.5, ségrégation aux nœuds et interface filaments/voids.

---

## Phase B — Zoom (4 runs × ~90 min = 3h)

**Générés automatiquement par trichotomy.py sur les métriques Phase A.**

Selon le gagnant Phase A, 3 scénarios :

**Si Z1 gagne :**
```
B1 : Z1 mais z_activation = 3.0  (plus tôt)
B2 : Z1 mais z_activation = 1.5  (plus tard)
B3 : Z1 + η=0.50                 (moins de répulsion)
B4 : Z1 + λ=20 Mpc               (screening plus court)
```

**Si A1 ou A2 gagnent (λ petit) :**
```
B1 : η=0.88, λ=2 Mpc
B2 : η=0.88, λ=5 Mpc   ← centre
B3 : η=0.88, λ=8 Mpc
B4 : η=0.60, λ=5 Mpc   ← même λ, η réduit
```

**Si A4 gagne (η petit) :**
```
B1 : η=0.40, λ=12 Mpc
B2 : η=0.55, λ=15 Mpc  ← centre
B3 : η=0.70, λ=18 Mpc
B4 : η=0.55, λ=8 Mpc   ← même η, λ réduit
```

---

## Phase C — Validation finale (1 run × ~2h)

**Un seul run sur le gagnant Phase B** (consensus Grok + DeepSeek) :
```yaml
n_particles: 1000000
n_steps: 2000
z_start: 5.0
z_end: 0.0
snapshots: [3.0, 2.0, 1.0, 0.5, 0.2, 0.0]
```

**Analyses à produire :**
1. Squelette 3D (skimage.morphology.skeletonize_3d) sur grille densité m+
2. Comparaison visuelle avec A0 (ΛCDM)
3. filament_score final vs A0_baseline
4. S(z) + ΔCOM(z) pour vérifier coexistence filaments/ségrégation

---

## Runs hors-boîte — si Phase A donne filament_score < 0.1 partout

**Y4 — Régularisation ε (Gemini) :**
```yaml
eta: 0.88
lambda_base_mpc: 20.0
r_smooth_mpc: 5.0
lambda_regularization_epsilon: 0.10  # NOUVEAU — remplace lambda_floor
# λ_eff = λ_base / √(ρ/ρ̄ + ε)
# λ_eff_max = 20/√0.1 = 63 Mpc (au lieu de 200 Mpc avec floor=0.01)
```

**Implémentation (remplace compute_lambda_eff_grid dans screening.cu) :**
```cuda
// AVANT (floor brut) :
const float rho_ratio = fmaxf(density[idx]/rho_mean, lambda_floor);

// APRÈS (régularisation continue) :
const float rho_ratio = density[idx]/rho_mean + epsilon;
// Plus de discontinuité, gradient de force lisse
```

**A_Boost — spectre initial boosté (Gemini) :**
```yaml
eta: 0.88
lambda_base_mpc: 15.0
r_smooth_mpc: 5.0
initial_conditions:
  sigma8: 1.5      # au lieu de 0.81 standard
  # Force les petites structures à se former AVANT la répulsion
```

---

## Métriques filaments — définition figée v2

```python
# optim/filament_metrics_v2.py
# SEUILS REHAUSSÉS (Grok + DeepSeek)

FILAMENT_MIN_LENGTH_MPC = 6.0    # au lieu de 3.0
FILAMENT_MIN_COUNT = 3            # au moins 3 filaments pour scorer
FILAMENT_MIN_FIL_FRACTION = 0.12  # au lieu de 0.15

def filament_score(m, lcdm_baseline=None):
    """
    Score ∈ [0, 1].
    Si lcdm_baseline fourni : score normalisé par ΛCDM.
    """
    s1 = min(m.length_mean / 10.0, 1.0)
    s2 = min(m.fil_fraction / 0.12, 1.0)
    s3 = min(m.connectivity / 5.0, 1.0)
    s4 = 1.0 if m.n_filaments >= FILAMENT_MIN_COUNT else 0.0

    raw = 0.35*s1 + 0.30*s2 + 0.20*s3 + 0.15*s4

    if lcdm_baseline and lcdm_baseline > 0:
        # Score relatif : 1.0 = aussi bien que ΛCDM
        return raw / lcdm_baseline
    return raw
```

**Score relatif attendu :**
```
A0 (ΛCDM) : 1.00  (baseline)
Janus Run 8 actuel : ~0.00 (deux blobs, pas de filaments)
Objectif cette nuit : > 0.30 (30% de la structure ΛCDM)
Rêve : > 0.60
```

---

## Conditions d'arrêt

```python
# HARD (kill immédiat, vérifiées tous les 10 steps)
if v_max > 5000:   ABORT("runaway")
if ke_ratio > 1e6: ABORT("explosion")
if nan_detected:   ABORT("NaN GPU")

# SOFT (flag, vérifiées tous les 50 steps)
if step == 300 and filament_score < 0.02:
    FLAG("structure plate — probablement dipole")
    # Ne pas killer — laisser finir pour confirmer

if delta_com > 0.7 * box_size:
    FLAG("dipole dominant — filaments impossibles")
    # Killer seulement si filament_score == 0 à ce stade
```

---

## Chronologie de la nuit

```
00h00  : Lancement Phase A (6 runs séquentiels)
         A0 → A1 → A2 → A4 → A7 → Z1

02h30  : Analyse Phase A
         - Afficher filament_score + carte densité pour chaque run
         - Calculer Π pour confirmer prédiction
         - trichotomy.py génère Phase B

02h45  : Lancement Phase B (4 runs)

06h15  : Analyse Phase B
         - Identifier gagnant
         - Générer config Phase C

06h30  : Lancement Phase C (1 run 1M)

08h30  : Analyse finale
         - Squelette 3D sur gagnant
         - Figures pour JPP
```

---

## Instructions Claude CLI

### Avant de démarrer

```
1. Vérifier status Run 2M et Run 300 Mpc :
   tail -3 /mnt/T2/janus-sim/output/run_2M/run.log
   tail -3 /mnt/T2/janus-sim/output/validation_300mpc/run.log

2. Mesurer VRAM libre :
   nvidia-smi | grep MiB

3. Implémenter les NOUVEAUX PARAMÈTRES dans le code :
   a. cross_force_asymmetry (float, défaut=1.0) dans config.rs
   b. cross_force_activation.mode/z_start/z_width dans config.rs
   c. lambda_regularization_epsilon (float, défaut=0.01) dans screening.cu
   
   → Ces 3 paramètres sont optionnels (défaut = comportement actuel)
   → Compilation test : cargo build --release
   → Smoke test : 10k particules, 50 steps, vérifier pas de régression

4. Générer les 6 configs Phase A avec les paramètres ci-dessus
5. Créer filament_metrics_v2.py
6. Lancer Phase A
```

### Règles absolues cette nuit

```
1. filament_score prime sur S_segregation
2. Appliquer le filtre Π avant tout run supplémentaire
3. Ne pas modifier les paramètres Phase B à la main → trichotomy.py décide
4. Si filament_score = 0 partout en Phase A → lancer Y4 (régularisation ε)
5. Rapporter au matin : gagnant + filament_score + une image densité par run
```

---

## Question ouverte pour les IA (à soumettre avec les résultats du matin)

> "Dans le modèle Janus avec répulsion croisée Yukawa, l'invariant Π = η×(λ/L_fil)²
> permet-il de prédire le seuil filament/dipole ?
> Si oui, quelle valeur critique Π_c prédit la théorie de la formation de structure
> perturbée par une répulsion Yukawa ?"

---

## Prédictions révisées après les 4 IA

| Run | Π    | Filament_score prédit | Confiance |
|-----|------|-----------------------|-----------|
| A0  | 0.00 | ~0.70 (baseline ΛCDM) | Certaine  |
| Z1  | —    | ~0.40-0.60            | 70%       |
| A1  | 0.06 | ~0.30-0.50            | 60%       |
| A2  | 0.43 | ~0.15-0.35            | 50%       |
| A4  | 0.78 | ~0.05-0.20            | 40%       |
| A7  | 2.44 | ~0.10-0.25 (asym.)    | 35%       |

**Gagnant prédit :** Z1 (activation progressive z=2) ou A1 (λ=3 Mpc).  
**Si ni l'un ni l'autre :** lancer Y4 (ε-régularisation) en urgence.

---

*Plan v2 — Nuit 20-21 mars 2026*  
*Révisé après analyse Gemini + ChatGPT + Grok + DeepSeek*  
*Objectif : premier univers Janus avec filaments cosmiques confirmés*
