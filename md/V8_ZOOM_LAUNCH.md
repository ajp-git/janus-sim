# v8 — Production Janus avec zoom spatial 150 Mpc (N=5M)

## Décision finale

Run de référence v8 avec :
- **N=5M particules** initial (n_grid=170), z=10→0
- μ=19 (cosmologie plate)
- Tous fixes Phase 1-13 actifs
- **Zoom spatial** : cube [-75, +75]³ Mpc centré, splits forcés dans la zone
- Split level max = 2 (strict, sinon explosion mémoire/temps)

**Rationale du N=5M** : donne une marge VRAM confortable (max attendu 
~8M avec splits, bien en dessous des 12 GB RTX 3060). Perte de résolution 
hors zone (2.9 Mpc au lieu de 2.3 Mpc) acceptable puisque le fond 
cosmologique sert de contexte, pas de mesure fine. La zone zoom reste à 
0.72 Mpc de résolution avec splits niveau 2.

**Objectif** : preprint Janus avec vidéo montrant la formation de halos dans 
la zone zoom, structure alvéolaire à grande échelle hors zone.

**Estimations** :
- N max attendu : 7-8M (réaliste), 13M (pessimiste)
- Temps GPU : 20-40h selon déclenchement splits L2
- VRAM max : ~3-4 GB

## Étape 0 — Audit de l'existant (30 min)

Avant tout, CLI doit **m'informer** de l'état exact du code concernant le 
splitting. J'ai besoin de réponses précises aux 5 questions :

### Q1 — Le splitting existe-t-il dans `janus_adaptive_zoom` ?

```bash
cd /mnt/T2/janus-sim
grep -rn "split" src/bin/janus_adaptive_zoom.rs | head -20
grep -rn "split" src/nbody_gpu.rs | head -20
```

Rapporter :
- Les fonctions qui implémentent le splitting
- Les paramètres CLI existants (--split-threshold, --max-split-level, etc.)
- Les conditions de déclenchement (densité ? spatial ? autre ?)

### Q2 — Trigger spatial existe-t-il déjà ?

Chercher des mentions de "zoom", "zone", "region", "bbox" dans les conditions 
de split :

```bash
grep -rn "zoom\|zone\|bbox\|region" src/bin/janus_adaptive_zoom.rs
grep -rn "in_zoom\|spatial" src/
```

Si oui → documenter la signature.
Si non → il faudra ajouter un flag CLI `--zoom-box MIN MAX` et une condition.

### Q3 — Les derniers runs avaient combien de splits ?

```bash
# Regarder v7b et v7
grep -E "split|N_hr" /mnt/T2/janus-sim/output/janus_adaptive_v7b*/run.log | \
  head -20
```

### Q4 — Per-particle mass support est-il toujours actif ?

Vérifier dans `src/nbody_gpu.rs` que les kernels utilisent `masses[i]` et 
pas `1.0` :

```bash
grep -n "masses\[" src/nbody_gpu.rs | head -10
```

### Q5 — Quel format exact doit prendre le flag `--zoom-box` ?

Je propose : `--zoom-box-min "-75,-75,-75" --zoom-box-max "75,75,75"`
Ou plus simple : `--zoom-cube-size 150` (cube centré sur origine).

Choisir la convention la plus simple à implémenter.

**Me rapporter les 5 réponses AVANT de coder quoi que ce soit.**

## Étape 1 — Modification du code (3-4h)

### Changement conceptuel majeur

**Le splitting actuel utilise ρ_totale comme critère, mais c'est inadapté à 
Janus.** Les m+ fuient les régions denses m- (répulsion), donc même un m+ 
qui s'effondre localement voit une ρ_totale modeste. Conséquence : en v7b, 
0 splits déclenchés.

**Correction** : séparer le calcul de densité en `ρ_plus` et `ρ_minus`, et 
utiliser `ρ_plus` comme critère de split (puisqu'on ne splitte que les m+).

### Changements à apporter

**1. Modifier `compute_densities()` dans `janus_adaptive_zoom.rs`** :

```rust
// AVANT (simplifié)
fn compute_densities(particles: &[ParticleV3], box_size: f64) -> Vec<f64> {
    let mut grid = vec![0.0; n_cells];
    for p in particles {
        grid[idx] += p.mass as f64;
    }
    grid
}

// APRÈS (séparation m+/m-)
fn compute_densities_split(particles: &[ParticleV3], box_size: f64) 
    -> (Vec<f64>, Vec<f64>) {
    let mut grid_plus = vec![0.0; n_cells];
    let mut grid_minus = vec![0.0; n_cells];
    for p in particles {
        let m = p.mass as f64;
        if p.sign == 1 {
            grid_plus[idx] += m;
        } else {
            grid_minus[idx] += m;
        }
    }
    (grid_plus, grid_minus)
}
```

**2. Ajouter flags CLI** :

```rust
#[arg(long, default_value = "0.0")]
zoom_cube_size: f64,  // Taille du cube zoom en Mpc (0 = désactivé)

#[arg(long, default_value = "2")]
max_split_level: u32,  // Limite stricte de niveaux de split

#[arg(long, default_value = "6.78e10")]
delta_split_l1: f64,  // Threshold L1 en M☉/Mpc³ (10× ρ_plus_mean)

#[arg(long, default_value = "6.78e11")]
delta_split_l2: f64,  // Threshold L2 en M☉/Mpc³ (100× ρ_plus_mean)
```

**3. Modifier `adaptive_split_check_with_thresholds()`** :

```rust
// Pseudo-code de la nouvelle condition
fn should_split(particle, rho_plus_local, split_level, zoom_size, max_level) -> bool {
    // Seulement les m+
    if particle.sign != 1 { return false; }
    
    // Condition spatiale : dans la zone zoom si activée
    if zoom_size > 0.0 {
        let in_zoom = particle.pos.x.abs() < zoom_size / 2.0
                   && particle.pos.y.abs() < zoom_size / 2.0
                   && particle.pos.z.abs() < zoom_size / 2.0;
        if !in_zoom { return false; }
    }
    
    // Limite niveau strict
    if split_level >= max_level { return false; }
    
    // Trigger selon niveau (utilise rho_plus au lieu de rho_total)
    let threshold = if split_level == 0 { delta_split_l1 } else { delta_split_l2 };
    rho_plus_local > threshold
}
```

**4. Compiler** :

```bash
cargo build --release --features cuda 2>&1 | tail -20
```

Si warnings non-bloquants → continuer.

### Références physiques pour les seuils

- ρ_crit = 3H₀²/(8πG) = 1.356×10¹¹ M☉/Mpc³ (avec H₀=69.9)
- ρ_plus_mean = Ω_b × ρ_crit = 0.05 × 1.356e11 = **6.78×10⁹ M☉/Mpc³**
- Seuil L1 = 10× ρ_plus_mean = 6.78×10¹⁰ (déclenche en quasi-linéaire)
- Seuil L2 = 100× ρ_plus_mean = 6.78×10¹¹ (déclenche sur surdensités réelles)

## Étape 2 — Sanity check (1h GPU)

Avant la production 30-40h, test court pour confirmer que le zoom 
fonctionne :

```bash
mkdir -p /app/output/sanity_v8_zoom
./target/release/janus_adaptive_zoom \
  --n-grid 170 --l-box 500 --z-init 10.0 --z-final 5.0 \
  --snap-interval 50 --steps-check 100 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --zoom-cube-size 150 \
  --max-split-level 2 \
  --delta-split-l1 6.78e10 \
  --delta-split-l2 6.78e11 \
  --out-dir /app/output/sanity_v8_zoom \
  --run-label sanity_v8_zoom \
  2>&1 | tee /app/output/sanity_v8_zoom/run.log
```

**Critères de validation** (à rapporter) :
- v_rms stable (pas d'explosion après splits)
- Aucun NaN
- N_total au final entre 5M (pas de splits) et 7M (niveau 1 actif)
- Spectre angulaire sur snap final : max_axes(m-) < 1.5 (Phase 13 tient)
- **ρ_plus_max observé** (nouvelle métrique à logger) : devrait monter 
  progressivement, pour voir si les seuils sont bien calibrés

Si **N_total reste à 5M et ρ_plus_max < 6.78e10** → aucun split déclenché 
car densité pas encore atteinte. Essayer à z plus bas (z=3 ou z=2).

Si **N_total reste à 5M mais ρ_plus_max > 6.78e10** → bug dans la condition 
de split, à diagnostiquer.

Si **v_rms explose** (>2000 km/s) après un split → bug per-particle mass 
(à corriger en priorité).

## Étape 3 — Lancement production

### Préparation

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTDIR="/app/output/janus_production_v8_zoom_${TIMESTAMP}"

mkdir -p "$OUTDIR"
touch "$OUTDIR/PRODUCTION_ACTIVE.lock"

echo "=== v8 ZOOM PRODUCTION ===" > "$OUTDIR/README.txt"
echo "Started: $(date)" >> "$OUTDIR/README.txt"
echo "Zoom: cube 150 Mpc centré" >> "$OUTDIR/README.txt"
echo "Max split level: 2" >> "$OUTDIR/README.txt"
echo "DO NOT rename this directory while LOCK file exists!" >> "$OUTDIR/README.txt"
```

### Commande

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 170 --l-box 500 --z-init 10.0 --z-final 0.0 \
  --snap-interval 20 --steps-check 50 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --zoom-cube-size 150 \
  --max-split-level 2 \
  --delta-split-l1 6.78e10 \
  --delta-split-l2 6.78e11 \
  --out-dir "$OUTDIR" \
  --run-label "v8_production_zoom" \
  2>&1 | tee "$OUTDIR/run.log" &

PRODUCTION_PID=$!
echo "$PRODUCTION_PID" > "$OUTDIR/PID"
echo "Production PID: $PRODUCTION_PID"
```

### Kill-switch (tmux séparé)

```bash
tmux new -d -s v8_zoom_monitor "bash -c '
OUTDIR=\"$OUTDIR\"
PID=\$(cat \"\$OUTDIR/PID\")

while true; do
  # Process encore vivant ?
  if ! kill -0 \$PID 2>/dev/null; then
    echo \"[\$(date +%H:%M)] Process terminé\" >> \"\$OUTDIR/monitor.log\"
    break
  fi
  
  # Dernière ligne du CSV
  last=\$(tail -1 \"\$OUTDIR/time_series.csv\" 2>/dev/null)
  if [ -z \"\$last\" ]; then sleep 120; continue; fi
  
  step=\$(echo \$last | cut -d, -f1)
  z=\$(echo \$last | cut -d, -f3)
  N=\$(echo \$last | cut -d, -f5)   # supposé colonne N_total
  rho_max=\$(echo \$last | cut -d, -f8)
  v_rms=\$(echo \$last | cut -d, -f9)
  
  # Seuils catastrophiques
  v_critical=\$(awk -v v=\"\$v_rms\" \"BEGIN { print (v+0 > 20000) ? 1 : 0 }\")
  rho_critical=\$(awk -v r=\"\$rho_max\" \"BEGIN { print (r+0 > 1e17) ? 1 : 0 }\")
  
  # NOUVEAU : seuil sur N total (explosion splits)
  # Avec N_init=5M, on s'attend à max ~8M. Alerte à 15M = large marge.
  N_critical=\$(awk -v n=\"\$N\" \"BEGIN { print (n+0 > 15000000) ? 1 : 0 }\")
  
  if [ \"\$v_critical\" = \"1\" ] || [ \"\$rho_critical\" = \"1\" ] || [ \"\$N_critical\" = \"1\" ]; then
    reason=\"v=\$v_rms rho=\$rho_max N=\$N\"
    echo \"CRITICAL \$reason at step=\$step z=\$z\" > \"\$OUTDIR/KILLED.txt\"
    kill \$PID
    break
  fi
  
  if grep -q \"nan\\|NaN\" \"\$OUTDIR/time_series.csv\" 2>/dev/null; then
    echo \"NaN detected\" > \"\$OUTDIR/KILLED.txt\"
    kill \$PID
    break
  fi
  
  echo \"[\$(date +%H:%M)] step=\$step z=\$z N=\$N v=\$v_rms rho=\$rho_max\" >> \"\$OUTDIR/monitor.log\"
  sleep 1800
done
'"
```

### RÈGLES STRICTES

1. **NE JAMAIS** renommer, supprimer, ou toucher à `$OUTDIR` tant que 
   `PRODUCTION_ACTIVE.lock` existe.

2. **NE JAMAIS** lancer un script autonome qui fait `mv` ou `rm -rf` 
   pendant que le run tourne.

3. Si besoin d'arrêter proprement : `kill $(cat $OUTDIR/PID)`, puis 
   attendre 5s, puis supprimer le lock.

4. Le kill-switch a maintenant **3 seuils** :
   - v_rms > 20000 km/s (explosion cinétique)
   - ρ_max > 10¹⁷ (explosion gravitationnelle)
   - **N_total > 15M (explosion de splits)** ← seuil révisé pour N=5M initial

## Étape 4 — Monitoring

Tu peux checker périodiquement :

```bash
# État actuel
tail -1 "$OUTDIR/time_series.csv"

# Évolution N_total (important avec splits)
awk -F, 'NR>1 {print $1, $3, $5}' "$OUTDIR/time_series.csv" | \
  awk 'NR%50==0'  # une ligne sur 50

# Derniers snapshots
ls -lth "$OUTDIR/snapshots/" | head -5
```

## Étape 5 — Analyse post-run

### 5.1 Validation grille à 5 z

Script Python à écrire :

```python
import numpy as np
import subprocess
import os

SNAPDIR = os.environ['OUTDIR'] + '/snapshots'
TARGETS = [5.0, 2.0, 1.0, 0.5, 0.0]

snaps = sorted(os.listdir(SNAPDIR))
for target_z in TARGETS:
    # Trouver le snap le plus proche
    best_snap = None
    best_diff = float('inf')
    for s in snaps:
        z = np.fromfile(f"{SNAPDIR}/{s}", dtype=np.float32, count=4)[3]
        if abs(z - target_z) < best_diff:
            best_diff = abs(z - target_z)
            best_snap = s
    
    if best_snap:
        print(f"=== z≈{target_z} : {best_snap} ===")
        subprocess.run(["python3", "scripts/validate_ics.py",
                       "--snap", f"{SNAPDIR}/{best_snap}",
                       "--nbin", "128"])
```

### 5.2 Comparaison zoom vs hors zoom

Le point **crucial** : les structures dans la zone [-75, +75] doivent être 
bien plus résolues qu'à l'extérieur. Générer un plot comparatif :

```python
# Pour le snap final z=0 :
# - Projection XY de -250 à +250 (tout)
# - Cadre rouge autour zone zoom
# - Inset zoom de la zone
```

### 5.3 Vidéo

La vidéo doit montrer :
1. **Vue globale** : box 500 Mpc, évolution z=10→0
2. **Zoom progressif** : zoom sur la zone centrale à mesure que les 
   structures se forment
3. **Split markers** : couleurs différentes selon split_level 
   (0=blanc/gris, 1=orange, 2=rouge)

```bash
ffmpeg -r 10 -i "$OUTDIR/frames/frame_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p \
  "$OUTDIR/video_v8_zoom.mp4"
```

### 5.4 Phase 9 deep analysis sur zone zoom

Restreindre l'analyse Phase 9 à la zone zoom pour avoir la meilleure stat :

```bash
python3 scripts/phase9_deep_analysis.py \
  --snap "$OUTDIR/snapshots/snap_FINAL.bin" \
  --restrict-box 150 \
  --out-md "$OUTDIR/phase9_zoom_z0.md"
```

### 5.5 Comparaison Petit

Utiliser le document existant `/mnt/user-data/outputs/janus_petit_comparison/comparison_petit_v7b.md` 
et mettre à jour avec résultats v8 (zone zoom).

## Échéancier

| Jour | Tâche | Durée |
|------|-------|-------|
| 0 matin | Audit code + modif compute_densities + flags + compile | 4h |
| 0 midi | Sanity check zoom | 1h GPU |
| 0 soir | Lancement production v8 | - |
| 1-2 | Production tournante | 20-40h GPU |
| 2-3 | Analyse 5.1 à 5.5 | 8h |
| 3 | Rapport final + vidéo | 4h |

**Total : 3-4 jours**

**Budget GPU détaillé** :
- Sanity check : 1h
- Production : 20-40h selon nombre de splits L2 déclenchés
- Total : 21-41h

## Rapport attendu après le run

`FINAL_REPORT_v8.md` structuré :

1. État du run (durée, steps, z atteint, N_final, nb splits)
2. Validation grille (tableau max_axes vs z)
3. Splits : nombre par niveau, localisation
4. Comparaison zoom vs hors zoom (visuel + métriques)
5. 7 prédictions Petit confrontées
6. Limitations et caveats
7. Figures et vidéo

## Contingences

**Si N_total grimpe trop vite** (>10M avant z=3) :
- Augmenter `delta-split-l1` à 2e11 (30× ρ_plus_mean)
- Kill et relance avec nouveau threshold

**Si aucun split ne se déclenche** (N stable à 5M pendant toute la sim) :
- Vérifier dans run.log que ρ_plus_max augmente bien vers les seuils
- Si ρ_plus_max < 6.78e10 à z=0, baisser `delta-split-l1` à 3e10
- Vérifier que la zone zoom contient bien des m+ (peut-être que m+ évite 
  complètement le centre)

**Si v_rms explose après splits** :
- Bug per-particle mass revenu
- Vérifier dans src/nbody_gpu.rs que les masses sont bien passées au BH
- Symptôme : v_rms saute de 100→10000 km/s en un step après un split massif

**Si v_rms explose après splits** :
- C'est un bug connu (per-particle mass support)
- Vérifier que les masses sont bien passées au BH (Q4)

**Si la zone zoom est vide** :
- Certains μ font que m+ évite complètement le centre
- Changer la position du zoom : `--zoom-center "50,50,50"` 
  (nécessite flag supplémentaire)

## Décision après le run

On décide ensemble :
- Si 7 prédictions Petit matchent + vidéo propre → **preprint prêt**
- Si certaines prédictions ne matchent pas → investiguer avant preprint
- Si le run crashe tôt → ajuster et relancer
