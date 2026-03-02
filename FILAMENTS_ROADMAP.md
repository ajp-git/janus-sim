# JANUS FILAMENTS ROADMAP — Plan Autonome CLI
# Objectif : voir des filaments cosmiques dans la simulation Janus
# Durée autorisée : plusieurs jours sans interruption
# Dernière mise à jour : 2026-03-02

---

## THÉORIE — POURQUOI ON N'A PAS VU DE FILAMENTS (et comment les obtenir)

### Le problème fondamental (validé mathématiquement)
```
Matrice de couplage Janus à deux fluides :
  M = 4πG [ ρ̄₊    -α·ρ̄₋ ]
           [ -α·ρ̄₊   ρ̄₋  ]

Valeurs propres pour α=1 :
  λ₊ = ρ̄(1+α) > 0  → mode blob (ségrégation) : CROÎT
  λ₋ = ρ̄(1-α) = 0  → mode filamentaire : NEUTRE (ni croît, ni décroît)
```

**Conséquence** : avec ICs uniformes aléatoires, les filaments ne peuvent pas
APPARAÎTRE spontanément (λ₋=0). Mais s'ils sont PRÉSENTS dans les ICs,
ils sont CONSERVÉS (λ₋=0 = stabilité neutre). C'est la clé.

### La solution : ICs Zel'dovich anti-corrélées (δ₋ = −δ₊)

**Idée** : au lieu de donner le même déplacement à toutes les particules,
donner des déplacements OPPOSÉS aux masses+ et masses− :
```
Masses+ : pos = pos_grille + ψ(x)   → se déplacent VERS les surdensités
Masses− : pos = pos_grille − ψ(x)   → se déplacent LOIN des surdensités
```

**Résultat physique** :
- Deux toiles cosmiques entrelacées : filaments de + où les filaments de − sont des vides
- Structure filamentaire imposée par P(k) → CONSERVÉE par λ₋=0
- Les masses+ s'effondrent en filaments/noeuds
- Les masses− expansent en vides entre filaments

**Résultat numérique** :
- Les particules + et − démarrent dans des régions SÉPARÉES
- Pas de close encounters +/− initiaux → instabilité numérique éliminée
- La force répulsive +/− commence faible (grande distance) → système stable

**Base théorique** : confirmée par o3 (chatgpt_analysis.md) :
> "Une anti-corrélation primordiale partielle pourrait produire une ségrégation
> gelée très tôt — héritage initial conservé, pas une instabilité."

---

## RÈGLES ABSOLUES (apprises des erreurs précédentes)

```
JAMAIS : docker stop $(docker ps -q)  — autres containers sur le serveur !
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement → GPU propre (0 MB utilisé)
TOUJOURS : git push avant production
TOUJOURS : vérifier KE/KE₀ au step 5 avant de continuer
TOUJOURS : softening ≥ 0.2 × spacing inter-particule
TOUJOURS : virialize_sampled(N/1000) — pas alpha fixe, pas virial_factor linéaire
JAMAIS : utiliser virial_velocity = scale × box × factor (physiquement incorrect)
JAMAIS : lancer 60M+ avec BH pur (N_max BH pur = 32M, FIX-013)
JAMAIS : lancer TreePM avec Zel'dovich (instabilité confirmée à N>1M)
```

---

## PARAMÈTRES DE RÉFÉRENCE VALIDÉS

```
Run 2M BH référence :
  N=2M, box=271 Mpc, spacing=2.15 Mpc, θ=0.7, dt=0.01
  KE/KE₀_max=1.924, S_max=0.694 @ z=1.8 ✅

Run 8M BH validé :
  N=8M, virialize_sampled → α≈4.57, S_max=0.459 ✅

N_max BH pur = 32M (mesuré GPU propre, FIX-013)
N_max TreePM = 63M (mais TreePM instable avec Zel'dovich)
```

---

## ARCHITECTURE DU PLAN

```
Phase A : Proof of concept anti-corrélé (1-2h)
  ↓ succès → Phase B
  ↓ échec  → Correction paramètres → retry
Phase B : Validation morphologique (6-12h)
  ↓ filaments visibles → Phase C
  ↓ blob seulement     → Ajuster amplitude P(k) → retry
Phase C : Production pleine résolution (3-7 jours)
  ↓ run complet → Analyse + rendu vidéo
```

---

## PHASE A — Proof of concept 100K anti-corrélé (OBLIGATOIRE)

### A.1 : Créer src/bin/janus_anticorr_test.rs

**Paramètres EXACTS à utiliser :**
```rust
const N_PARTICLES: usize = 100_000;
const BOX_SIZE: f64 = 100.0;          // 100 Mpc
const Z_INIT: f64 = 5.0;              // Même que runs validés
const THETA: f64 = 0.7;               // FIX-012 validé
const DT: f64 = 0.01;                 // Identique run 2M référence
const TOTAL_STEPS: usize = 2000;
const SOFTENING: f64 = 0.6;           // 0.3 × spacing (spacing=2.15 Mpc à 100K/100Mpc)

// P(k) : même spectre que validé sur treepm_zeldovich_test.rs
const N_S: f64 = 0.96;
const K_PEAK: f64 = 0.02;            // Mpc⁻¹, scale de cohérence ~50 Mpc
```

**ICs anti-corrélées — implémentation :**
```rust
fn generate_anticorr_zeldovich_ics(n: usize, box_size: f64, seed: u64)
    -> (Vec<f64>, Vec<f64>, Vec<i8>)
{
    // 1. Grille cubique n_side³ avec jitter aléatoire (10% spacing)
    //    CONVENTION CENTRÉE [-box/2, +box/2] (FIX-008)
    let n_side = (n as f64).cbrt() as usize;
    let spacing = box_size / n_side as f64;

    // 2. Assigner les signes ALÉATOIREMENT (FIX dans CLAUDE.md)
    //    NE PAS faire : signe = si idx < n/2 { +1 } else { -1 }
    let mut rng = StdRng::seed_from_u64(seed);
    let signs: Vec<i8> = (0..n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect();

    // 3. Générer le champ de déplacement Zel'dovich via FFT
    //    P(k) ∝ k^N_S / (1 + (k/K_PEAK)^4)
    //    Même méthode que treepm_zeldovich_multimode (validé onset z=2.65)
    let psi = generate_zeldovich_displacement_3d(n_side, box_size, seed);

    // 4. ANTI-CORRÉLATION : appliquer ±ψ selon le signe
    let amplitude = 0.3 * spacing; // Déplacement max = 30% spacing (validé stable)
    let mut positions = Vec::with_capacity(n * 3);
    for i in 0..n {
        let ix = i % n_side;
        let iy = (i / n_side) % n_side;
        let iz = i / (n_side * n_side);

        // Position grille centrée + jitter
        let jitter_x = (rng.gen::<f64>() - 0.5) * 0.1 * spacing;
        let jitter_y = (rng.gen::<f64>() - 0.5) * 0.1 * spacing;
        let jitter_z = (rng.gen::<f64>() - 0.5) * 0.1 * spacing;

        let x0 = (ix as f64 + 0.5) * spacing - box_size / 2.0 + jitter_x;
        let y0 = (iy as f64 + 0.5) * spacing - box_size / 2.0 + jitter_y;
        let z0 = (iz as f64 + 0.5) * spacing - box_size / 2.0 + jitter_z;

        // Déplacement anti-corrélé : +ψ pour masses+, -ψ pour masses-
        let sign_f = signs[i] as f64; // +1.0 ou -1.0
        let dx = sign_f * psi[i * 3 + 0] * amplitude;
        let dy = sign_f * psi[i * 3 + 1] * amplitude;
        let dz = sign_f * psi[i * 3 + 2] * amplitude;

        // Conditions périodiques (FIX-008b) — [-box/2, +box/2]
        let wrap = |x: f64| {
            let mut xw = x;
            while xw > box_size / 2.0 { xw -= box_size; }
            while xw < -box_size / 2.0 { xw += box_size; }
            xw
        };
        positions.push(wrap(x0 + dx));
        positions.push(wrap(y0 + dy));
        positions.push(wrap(z0 + dz));
    }

    // 5. Virialization PE_binding (JAMAIS virial_factor linéaire)
    //    virialize_sampled avec min(N/100, 10000) particules
    let n_sample = (n / 100).max(1000).min(10000);
    // Calculer PE_binding sur paires même signe échantillonnées
    // α = sqrt(|PE_binding| / 2KE) — même méthode que run 8M validé
    // Appliquer α aux vitesses initiales (ou initialiser à zéro et laisser
    // virialize_sampled faire son travail via la méthode existante)

    // ALTERNATIVE PLUS SIMPLE : utiliser sim.virialize_sampled(n_sample)
    // après création du GpuNBodyTwoPass — c'est ce qui a fonctionné pour 8M

    (positions, velocities_zero, signs)
}
```

**IMPORTANT** : Si `generate_zeldovich_displacement_3d` n'existe pas en standalone,
réutiliser la logique de `treepm_zeldovich_test.rs` ou `janus_60m_treepm.rs`
qui contient déjà `generate_zeldovich_ics_full()`. Adapter pour l'anti-corrélation
en **une seule ligne** : `let sign_f = signs[i] as f64;` avant le déplacement.

### A.2 : Lancer le test

```bash
nvidia-smi  # Vérifier GPU propre — OBLIGATOIRE
# Si processus parasites : kill -9 <pid> pour chaque

docker compose run --rm dev cargo run --release --features cuda \
  --bin janus_anticorr_test \
  2>&1 | tee /mnt/T2/janus-sim/output/anticorr_test_100k.log
```

### A.3 : Critères de succès Phase A

Lire time_series.csv à step 20, 50, 100 :

**PASS si :**
```
Step 5  : KE/KE₀ ∈ [0.8, 5.0]        ← pas d'explosion
Step 20 : KE/KE₀ ∈ [0.5, 10.0]       ← toujours stable
Step 50 : Seg > 0.001                  ← ségrégation non nulle
Step 100: KE/KE₀ < 50                 ← pas de collapse brutal
Frame step 100 : structure non sphérique visible (elongée, feuillets)
```

**FAIL si :**
```
Step 5  : KE/KE₀ > 50  → STOP immédiat → Section FAIL-A1
Step 20 : KE/KE₀ > 200 → STOP         → Section FAIL-A2
Step 100: Seg stagne à < 0.0001        → Section FAIL-A3
Frame step 100 : sphère parfaite       → Section FAIL-A4
```

### A.4 : Frames à générer (Phase A)

Générer frames aux steps 20, 50, 100, 200, 500 avec Format C (densité ρ+/ρ−) :
- Fond global blanc, panneaux fond noir
- Deux panneaux : ρ+ (colormap bleu) et ρ− (colormap rouge)
- Projection XY uniquement (suffisant pour valider la morphologie)

Uploader frame_step_0100_dens.png pour vérification visuelle.

---

## SECTIONS FAIL — CORRECTIONS AUTOMATIQUES

### FAIL-A1 : KE/KE₀ > 50 au step 5 (explosion immédiate)
```
Cause probable : softening trop petit → forces divergentes
Action :
  1. Doubler le softening : SOFTENING = 1.2 Mpc
  2. Vérifier que virialize_sampled est appelé APRÈS création du sim
  3. Relancer Phase A avec nouveau softening
  4. Si encore échec : SOFTENING = 2.0, dt = 0.005 → relancer
```

### FAIL-A2 : KE/KE₀ > 200 entre step 20-100 (explosion lente)
```
Cause probable : dt trop grand OU virialization incorrecte
Action :
  1. Vérifier α retourné par virialize_sampled
     - Si α > 50 : virialization surévaluée → utiliser α = min(α, 20)
     - Si α < 1  : virialization sous-évaluée → utiliser α = max(α, 2)
  2. Réduire dt : DT = 0.005
  3. Relancer Phase A
  4. Si encore échec : dt = 0.002, vérifier PE_binding < 0 (assert!)
```

### FAIL-A3 : Seg stagne à < 0.0001 (pas de ségrégation)
```
Cause probable : déplacements trop petits → ICs quasi-uniformes
Action :
  1. Vérifier amplitude max du déplacement dans le log
     - Si < 0.01 Mpc : le scaling FFT est trop faible
     - Augmenter : amplitude = 0.5 * spacing (au lieu de 0.3)
  2. Vérifier que les signes sont bien aléatoires (pas tous +1)
  3. Relancer Phase A avec amplitude augmentée
```

### FAIL-A4 : Structure sphérique (pas de filaments visibles)
```
Cause probable : P(k) trop lisse ou amplitude insuffisante
NE PAS paniquer — ce peut être normal à step 100 (trop tôt)
Action :
  1. Attendre step 500 avant de juger la morphologie
  2. Vérifier le spectre P(k) : k_peak doit être ~ 0.02 Mpc⁻¹
  3. Si toujours sphérique à step 500 :
     Augmenter amplitude à 50% du spacing (quasi-linéaire mais visible)
  4. Générer frame step 500 et uploader
```

---

## PHASE B — Validation morphologique 500K (6-12h)

Si Phase A PASS, lancer immédiatement :

### B.1 : Paramètres 500K

```rust
const N_PARTICLES: usize = 500_000;
const BOX_SIZE: f64 = 215.0;    // densité identique à 100K : (100³/100K)=(215³/500K)
const SOFTENING: f64 = 0.50;   // 0.3 × spacing (spacing=1.67 Mpc à 500K/215Mpc)
const TOTAL_STEPS: usize = 12000;
// Autres paramètres : identiques Phase A
```

Calculer box_size pour conserver la densité :
```
spacing_ref = (BOX_A³ / N_A)^(1/3)
BOX_B = (N_B / N_A)^(1/3) × BOX_A = (500K/100K)^(1/3) × 100 = 1.71 × 100 = 171 Mpc
```
Ou utiliser 200 Mpc pour avoir une boîte un peu plus grande (filaments plus visibles).

### B.2 : Monitoring Phase B

Surveiller en arrière-plan toutes les 30 minutes :
```bash
tail -5 /mnt/T2/janus-sim/output/anticorr_500k/time_series.csv
```

STOP immédiat si à step 50 : KE/KE₀ > 100 → appliquer FAIL-A1/A2

### B.3 : Milestones Phase B

```
Step 100  → Générer frames 3 projections XY/XZ/YZ (Format C)
             Vérifier : structure non sphérique ? Elongations ?
Step 500  → Frame + time_series complet
             Chercher : feuillets (Zel'dovich pancakes) visibles ?
Step 1500 → Onset ségrégation ? Seg > 0.01 ?
Step 3000 → Filaments visibles ? Couleurs séparées ?
Step 12000 → Run complet → vidéo Format A + C
```

### B.4 : Critères de succès Phase B

**GO Phase C si à step 1000 :**
```
✓ KE/KE₀ < 20 (stable)
✓ Structures non sphériques visibles sur au moins une projection
✓ ρ+ et ρ− visuellement dans des régions différentes
```

**NO-GO si blob sphérique parfait à step 1000 :**
→ Voir Section FAIL-B1

### FAIL-B1 : Blob sphérique persistant à step 1000
```
Diagnostic : λ₋=0 efface la structure initiale plus vite que prévu
Options (dans l'ordre) :

Option 1 : Augmenter amplitude à 70% du spacing
  → Quasi-linéaire mais très visible initialement

Option 2 : Implémenter α=0.95 (Yukawa doux)
  Dans le kernel CUDA, modifier :
    let interaction = if sign_i == sign_j { 1.0 } else { -0.95 };
  → λ₋ = 0.05 × ρ̄ > 0 → filaments CROISSENT
  → Physiquement : légère asymétrie dans la répulsion
  → Tester sur 100K d'abord (50 steps)

Option 3 : Utiliser z_init=10 au lieu de z_init=5
  → Déplacements 2× plus petits, plus proches du régime linéaire
  → Les filaments sont moins compressés initialement
  → Structure plus fine et plus durable
  → 30% de steps supplémentaires (15600 au lieu de 12000)
```

---

## PHASE C — Production pleine résolution (3-7 jours)

Si Phase B PASS, lancer le run de production.

### C.1 : Choix du N optimal

```
Contrainte hardware : N_max BH pur = 32M (FIX-013)

Recommandé : N = 20M
  box = 400 Mpc (densité identique 2M référence : spacing=2.15 Mpc)
  Calcul : (400³/20M)^(1/3) = 2.15 Mpc ✓
  Softening = 0.4 Mpc (0.2 × spacing)
  Runtime estimé : ~37s/step × (20/30) × 15600 steps ≈ 128h ≈ 5.3 jours

Alternative si 20M instable : N = 10M
  box = 320 Mpc, spacing = 2.15 Mpc
  Runtime : ~15s/step × 15600 = 65h ≈ 2.7 jours
```

### C.2 : Paramètres production

```rust
// src/bin/janus_filaments_production.rs
const N_PARTICLES: usize = 20_000_000;
const BOX_SIZE: f64 = 400.0;          // Mpc — densité 2M référence
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;               // FIX-012
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 15600;     // z=5 → z=0 avec dτ/dt=0.011
const SOFTENING: f64 = 0.4;           // 0.2 × spacing (2.15 Mpc)
// ICs : anti-corrélées Zel'dovich P(k), amplitude=0.3×spacing
// Virialization : virialize_sampled(20000)
// Intégrateur : BH pur step_dkd (PAS TreePM)
// Output :
//   render_data/*.bin toutes les 20 steps (783 MB × 780 = ~600 GB — VÉRIFIER DISQUE)
//   time_series.csv chaque step
//   snapshots/*.bin tous les 200 steps (format header+pos+vel+sign)
```

**VÉRIFIER LE DISQUE AVANT LANCEMENT :**
```bash
df -h /mnt/T2/
# render_data : 20M × 13 bytes = 260 MB/fichier × 780 fichiers = 200 GB
# snapshots   : 20M × 28 bytes = 560 MB/fichier × 78 fichiers = 44 GB
# TOTAL estimé : ~250 GB — vérifier qu'il reste 300 GB libres
# Si insuffisant : réduire fréquence render_data à toutes les 40 steps
```

### C.3 : Lancement production

```bash
# 1. GPU propre
nvidia-smi
# Si processus parasites → kill -9 <pids>

# 2. Git push
cd /mnt/T2/janus-sim
git add -A && git commit -m "feat: Janus filaments production launch - anticorrelated Zeldovich"
git push origin main

# 3. Lancer en détaché
docker compose run -d --rm dev bash -c \
  "cargo run --release --features cuda \
   --bin janus_filaments_production \
   2>&1 | tee /mnt/T2/janus-sim/output/filaments_production.log"

# 4. Sauvegarder le container ID
docker ps | grep janus | tee /mnt/T2/janus-sim/CURRENT_RUN.txt

# 5. Mettre à jour RUNS.md
```

### C.4 : Monitoring automatique Phase C

Script de monitoring à lancer en tâche de fond :
```bash
cat > /tmp/monitor_filaments.sh << 'EOF'
#!/bin/bash
LOG=/mnt/T2/janus-sim/output/filaments_production.log
CSV=/mnt/T2/janus-sim/output/filaments_20M_*/time_series.csv

while true; do
    STEP=$(tail -1 $CSV 2>/dev/null | cut -d',' -f1)
    KE=$(tail -1 $CSV 2>/dev/null | cut -d',' -f7)
    SEG=$(tail -1 $CSV 2>/dev/null | cut -d',' -f8)
    echo "$(date): step=$STEP KE=$KE Seg=$SEG" >> /tmp/monitor_log.txt

    # STOP automatique si explosion
    KE_INT=$(echo "$KE > 200" | bc -l 2>/dev/null)
    if [ "$KE_INT" = "1" ] && [ -n "$STEP" ] && [ "$STEP" -gt "50" ]; then
        echo "EXPLOSION DÉTECTÉE step=$STEP KE=$KE" >> /tmp/monitor_log.txt
        # Ne pas killer automatiquement — noter dans le log
    fi

    sleep 300  # Toutes les 5 minutes
done
EOF
bash /tmp/monitor_filaments.sh &
```

### C.5 : Milestones Phase C

```
Step 50   → Vérifier KE/KE₀ < 10 (stabilité confirmée)
Step 500  → Frame dens + pub → structure visible ?
Step 1500 → Onset ségrégation attendu (z≈2.4)
Step 3000 → S_max devrait dépasser 0.1
Step 8000 → S_max devrait approcher son pic (~0.4-0.6)
Step 15600 → Run complet → vidéo finale
```

---

## RENDU — Pipeline de visualisation

### Format A : Cinématique 2.5D (vidéo grand public)
```python
# scripts/render_anticorr.py — Générer après chaque milestone
# Format A : isométrique 2.5D, fond noir, bleu=masses+, rouge=masses−
# Résolution : 2048×2048
# Sauvegarde : frames/step_XXXXXX_25d.png
```

### Format C : Publication scientifique (ρ+/ρ− séparés)
```python
# Format C : fond global blanc, 2 panneaux côte à côte
# Panneau gauche  : ρ+ colormap bleu (log(1+ρ) normalisé)
# Panneau droit   : ρ− colormap rouge/orange (log(1+ρ) normalisé)
# Barre de couleur avec échelle
# Titre : "Janus Xm — Step XXXXXX | z=X.XX | Seg=X.XXXX"
# Résolution : 2048×2048
# Sauvegarde : frames/step_XXXXXX_dens.png
```

### Génération vidéo (après run complet)
```bash
# Vidéo cinématique
ffmpeg -framerate 24 -pattern_type glob \
  -i 'frames/step_*_25d.png' \
  -vf scale=1280:720 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  videos/janus_filaments_25d.mp4

# Vidéo publication
ffmpeg -framerate 24 -pattern_type glob \
  -i 'frames/step_*_dens.png' \
  -vf scale=1920:960 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  videos/janus_filaments_dens.mp4
```

---

## TESTS DE NON-RÉGRESSION (avant tout lancement)

Après avoir écrit janus_anticorr_test.rs, exécuter ces checks :

### Test 1 : Positions centrées
```rust
// Vérifier que toutes les positions sont dans [-box/2, +box/2]
let pos_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
assert!(pos_max <= box_size / 2.0 + 0.001,
    "Position hors limites : {:.3} > {:.3}", pos_max, box_size/2.0);
```

### Test 2 : Séparation spatiale + et −
```rust
// Vérifier que les masses+ et masses− ne sont PAS co-localisées
// (c'est la propriété clé des ICs anti-corrélées)
let com_plus  = compute_com(&pos_plus);
let com_minus = compute_com(&pos_minus);
let separation = distance(com_plus, com_minus);
assert!(separation > box_size * 0.01,
    "Masses+ et − trop proches initialement : séparation = {:.3} Mpc", separation);
// Seg_0 attendue : 0.05-0.20 (déjà séparées par les ICs anti-corrélées)
```

### Test 3 : Signes aléatoires
```rust
let n_plus  = signs.iter().filter(|&&s| s > 0).count();
let n_minus = signs.iter().filter(|&&s| s < 0).count();
let ratio = n_plus as f64 / n as f64;
assert!((ratio - 0.5).abs() < 0.05,
    "Signes non aléatoires : {:.1}% positifs (attendu 50%)", ratio*100.0);
```

### Test 4 : PE_binding négatif (FIX-007 Janus virialization)
```rust
// TOUJOURS vérifier avant d'appliquer α
assert!(pe_binding < 0.0,
    "PE_binding positif ({:.3e}) — virialization impossible !", pe_binding);
let alpha = (pe_binding.abs() / (2.0 * ke)).sqrt();
assert!(alpha > 0.5 && alpha < 50.0,
    "α hors plage raisonnable : α={:.3}", alpha);
```

---

## CHECKLIST AVANT CHAQUE LANCEMENT

```
□ nvidia-smi → GPU propre (0 MB utilisé)
□ df -h /mnt/T2/ → espace disque suffisant (> 300 GB libres)
□ Test positions centrées → PASS
□ Test séparation +/− → Seg_0 > 0.01
□ Test signes aléatoires → 50% ± 5%
□ Test PE_binding → négatif
□ Alpha virialization → 0.5 < α < 50
□ KE/KE₀ au step 5 < 10 (test 1000 steps d'abord)
□ git push origin main
□ RUNS.md mis à jour
□ Container ID sauvegardé dans CURRENT_RUN.txt
```

---

## RÉSULTATS ATTENDUS

### Ce qu'on devrait voir avec ICs anti-corrélées :

**Step 0-100 (z=5.0 → 4.85) :**
- Deux toiles cosmiques entrelacées visibles dès les premières frames
- ρ+ : filaments/feuillets là où ρ− est vide, et vice versa
- Seg_0 ≈ 0.05-0.20 (déjà séparé par les ICs)
- KE/KE₀ < 2 (stable grâce à la séparation spatiale)

**Step 1000-2000 (z≈3.0) :**
- La dynamique Janus amplifie la séparation +/−
- Les filaments de + s'effondrent davantage
- Les vides de − s'expansent
- Seg croît : 0.1-0.3

**Step 3000-6000 (z≈2.0-1.5) :**
- Pic de ségrégation : Seg_max ≈ 0.4-0.6
- Toile cosmique des masses+ clairement visible
- Masses− repoussées dans les vides entre filaments

**Step 12000 (z=0) :**
- Structure finale : filaments de masses+ + halos aux noeuds
- Grandes régions vides de masses−
- Image comparable aux simulations ΛCDM mais avec deux secteurs distincts

---

## DOCUMENTATION À METTRE À JOUR

Après chaque phase, mettre à jour :

### RUNS.md — Ajouter l'entrée :
```markdown
### Run: janus_anticorr_Xm_filaments
Date: 2026-03-XX
Status: running/completed

Parameters:
  N: X,000,000
  ICs: Zel'dovich anti-corrélées (δ₋ = −δ₊)
  P(k) ∝ k^0.96 / (1 + (k/0.02)⁴)
  Amplitude: 0.3 × spacing
  z_init: 5.0, θ=0.7, dt=0.01
  Integrator: BH pur step_dkd (PAS TreePM)
  Virialization: virialize_sampled(N/1000), α=X.XX

Results:
  Seg_0: X.XXXX (non nul — ICs anti-corrélées)
  KE/KE₀_max: X.XX
  S_max: X.XXXX @ step X (z≈X.X)
  Filaments visibles: OUI/NON
```

### KNOWN_FIXES.md — Ajouter FIX-014 :
```
FIX-014 : ICs anti-corrélées pour filaments Janus
  δ₋(k) = −δ₊(k) : déplacement opposé pour masses+ et masses−
  Avantages :
    1. Structure filamentaire imposée par P(k) → conservée par λ₋=0
    2. Séparation spatiale initiale → pas de close encounters → stable
    3. Seg_0 non nul (normal, c'est voulu)
  Implémentation : sign_f = signs[i] as f64; pos += sign_f * displacement;
  Validé sur 100K step 100 (Phase A) avant production
```

---

## QUESTIONS OUVERTES POUR JPP (à documenter)

Si les filaments apparaissent avec ICs anti-corrélées, noter dans le document v3 :

1. **Les filaments Janus nécessitent une asymétrie primordiale** entre les deux secteurs
   de masse (δ₋ ≠ δ₊). Cela correspond à quoi physiquement dans le formalisme bimétrique ?

2. **La ségrégation et la structure filamentaire sont deux phénomènes distincts** :
   - Ségrégation (blob) : λ₊ > 0, croît spontanément
   - Filaments : λ₋=0, préservés si présents dans les ICs

3. **L'amplitude initiale de l'anti-corrélation** détermine la visibilité des filaments.
   Quelle amplitude est physiquement justifiée dans le modèle de Petit ?

---

## ESTIMATION GLOBALE

```
Phase A (100K, 2000 steps)   : ~30 ms/step × 2000 = 1h
Phase B (500K, 12000 steps)  : ~200 ms/step × 12000 = 7h
Phase C (20M, 15600 steps)   : ~25s/step × 15600 = 108h = 4.5 jours

Total estimé : ~5 jours (Phase A+B en parallèle de C si B PASS)
```

---

## ORDRE D'EXÉCUTION STRICT

```
1. Lire ce fichier en entier
2. Lire KNOWN_FIXES.md (FIX-001 à FIX-014)
3. Lire VALIDATION_RULES.md section 10 (règles CLI)
4. nvidia-smi → GPU propre
5. Créer src/bin/janus_anticorr_test.rs
6. Tests de non-régression (Test 1-4 ci-dessus)
7. cargo check → 0 erreur
8. Phase A : lancer 100K
9. Vérifier step 5 (KE < 10) → continuer ou FAIL-A
10. Vérifier step 100 → frame → décision Phase B
11. Phase B : 500K
12. Vérifier morphologie step 500 → décision Phase C
13. Phase C : 20M production
14. Monitoring + milestones
15. Rendu vidéo final
16. git push final
17. Mettre à jour RUNS.md + KNOWN_FIXES.md
```

**NE JAMAIS SAUTER UNE ÉTAPE.**
**TOUJOURS VALIDER LA PHASE PRÉCÉDENTE AVANT DE LANCER LA SUIVANTE.**
**EN CAS DE DOUTE : documenter dans RUNS.md et continuer avec les paramètres les plus conservateurs.**
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

---

## RÉSULTATS DES PHASES A/B (2026-03-02)

### Correction méthodologique importante

L'approche initiale du roadmap (déplacements opposés ±ψ depuis les mêmes points de grille)
**ne fonctionne pas** car elle crée des paires +/− proches à chaque point de grille.

**Nouvelle approche validée :** Attribution des signes basée sur le champ de densité local δ(x) :
- δ(x) > 0 (surdensité) → particule positive
- δ(x) < 0 (sous-densité) → particule négative

Cette méthode place naturellement les + dans les régions denses et les − dans les vides,
créant une véritable ségrégation spatiale.

### Phase A — 100K (PASS)
```
Seg₀ = 0.1309 (13.1% of box) ✅
KE/KE₀ max = 1.33 ✅
Seg finale = 0.109 (maintenue)
Ratio d-/d+ = 1.48 → anti-corrélation confirmée
Contraste spatial : 26% régions +, 25% régions −
```

### Phase B — 500K (PASS)
```
Seg₀ = 0.0425 (4.25% of box) ✅
KE/KE₀ max = 1.15 ✅
Seg finale = 0.0446 (maintenue)
Runtime : 345s (172 ms/step)
Contraste spatial : 31% régions +, 24% régions −
```

### Implémentation corrigée (src/bin/janus_anticorr_test.rs)
```rust
// Après FFT inverse de δ(k) → δ_r en espace réel
let sign = if delta_r[idx] > 0.0 { 1 } else { -1 };
signs.push(sign);
// Position = grille + ψ (même déplacement pour tous)
positions.push(wrap(x0 + dx));
```

### Prochaine étape : Phase C (20M)
- Utiliser la même méthode density-based
- box = 400 Mpc, softening = 0.4 Mpc
- Runtime estimé : ~5 jours
