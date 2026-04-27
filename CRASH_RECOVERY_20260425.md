# CRASH RECOVERY — État complet du projet Janus
**Date**: 2026-04-25 06:45 UTC
**Session**: Claude CLI (Opus 4.5)

---

## 1. SIMULATION EN COURS

### Production v8 (ACTIVE)
```
Directory: /mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/
Container: janus-sim-dev-run-e15b4149e495 (Up 8+ hours)
Status: z ≈ 0.08 → 0.00 (presque terminé)
```

**Commande lancée:**
```bash
docker compose run --rm dev cargo run --release --features cuda --bin janus_adaptive_zoom -- \
  --n-grid 170 --l-box 500 --z-init 10.0 --z-final 0.0 \
  --snap-interval 20 --steps-check 50 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --zoom-cube-size 150 \
  --max-split-level 2 \
  --delta-split-l1 6.78e10 \
  --delta-split-l2 6.78e11 \
  --out-dir /app/output/janus_production_v8_zoom_20260424_2136 \
  --run-label "v8_production_zoom"
```

**Résultat préliminaire:**
- N = 4,913,000 constant (AUCUN SPLIT déclenché)
- ρ+_max ≈ 5.8e10 M☉/Mpc³ (toujours sous seuil 6.78e10)
- v_rms: 40 → 165 km/s (croissance normale)
- ~400 snapshots générés (snap_00000.bin → snap_08xxx.bin)

### Render Daemon (ACTIVE)
```bash
nohup /tmp/plotenv/bin/python /mnt/T2/janus-sim/scripts/render_daemon_adaptive_v2.py > /tmp/render_daemon_v8.log 2>&1 &
```
- Génère frames 10-panel et 2.5D automatiquement
- ~400 frames générées

---

## 2. MODIFICATIONS v8 APPLIQUÉES

### Fichier: src/bin/janus_adaptive_zoom.rs

**A. Nouvelle fonction compute_densities_split() (ligne ~530-580):**
```rust
fn compute_densities_split(particles: &[ParticleV3], box_size: f64) -> (Vec<f64>, Vec<f64>, f64, f64) {
    // Retourne (densities_plus, densities_minus, rho_plus_max, rho_minus_max)
    // Grille 64³, sépare m+ et m-
}
```

**B. Nouveaux CLI flags (lignes 110-125):**
```rust
#[arg(long, default_value = "0.0")]
zoom_cube_size: f64,  // Taille cube zoom en Mpc

#[arg(long, default_value = "2")]
max_split_level: u8,  // Limite stricte niveaux split

#[arg(long, default_value = "6.78e10")]
delta_split_l1: f64,  // Seuil L1 en M☉/Mpc³

#[arg(long, default_value = "6.78e11")]
delta_split_l2: f64,  // Seuil L2 en M☉/Mpc³
```

**C. Condition spatiale dans adaptive_split_check_with_thresholds():**
```rust
if zoom_cube_size > 0.0 {
    let px = p.pos[0] as f64;
    let py = p.pos[1] as f64;
    let pz = p.pos[2] as f64;
    if px.abs() > zoom_half || py.abs() > zoom_half || pz.abs() > zoom_half {
        continue;  // Hors zone zoom → pas de split
    }
}
```

**D. Logging ρ+_max dans CSV et console:**
- Colonne `rho_plus_max` ajoutée au time_series.csv
- Affichage dans les logs step: `ρ+_max=X.XXeYY`

---

## 3. ÉTAT GIT

```
Branch: main
Dernier commit: a32ea76 (Phase 13 - random Morton offset)
```

**Fichiers modifiés non commités:**
- `src/bin/janus_adaptive_zoom.rs` (modifications v8)
- `scripts/render_daemon_adaptive_v2.py` (chemins v8)
- Cargo.toml (modifications mineures)

**Pour commiter v8:**
```bash
git add src/bin/janus_adaptive_zoom.rs
git commit -m "feat(v8): Split trigger using ρ_plus + spatial zoom cube

- compute_densities_split() separates m+ and m- densities
- CLI flags: --zoom-cube-size, --max-split-level, --delta-split-l1/l2
- Spatial condition: only particles in [-size/2, +size/2]³ can split
- Logging: rho_plus_max added to CSV and console"
```

---

## 4. MODULES ACTIVÉS DANS v8

| Module | État | Détails |
|--------|------|---------|
| Gravité N-body GPU | ✅ ON | Barnes-Hut θ=0.7 |
| Cooling (GpuCooling) | ✅ ON | T_init=10000K, hardcodé |
| VSL dynamique | ✅ ON | CoupledFriedmann, η=1.045 |
| SPH traditionnel | ❌ OFF | Grille CIC, pas de SPH |
| Particle splitting | ✅ ON | Mais jamais déclenché (ρ+ trop faible) |

---

## 5. OBSERVATION CRITIQUE

**Problème: Aucun split sur tout le run z=10→0**

Cause identifiée:
- ρ+_max fluctue entre 5.0e10 et 7.2e10 M☉/Mpc³
- Seuil L1 = 6.78e10 M☉/Mpc³ (10× ρ_mean_plus)
- Les m+ sont trop dilués par répulsion des m- (μ=19 → 95% m-)
- Les pics de densité m+ ne dépassent jamais durablement le seuil

**Solutions potentielles pour prochains runs:**
1. Baisser `--delta-split-l1` à 3e10 ou 4e10
2. Réduire box size (100-200 Mpc au lieu de 500)
3. Augmenter résolution (n_grid > 170)
4. Tester avec μ plus faible (3-10)

---

## 6. FICHIERS CLÉS

### Code source
```
/mnt/T2/janus-sim/src/bin/janus_adaptive_zoom.rs  # Binary principal v8
/mnt/T2/janus-sim/src/nbody_gpu.rs                # GPU N-body (Phase 13 fix)
/mnt/T2/janus-sim/src/cooling_gpu.rs              # Refroidissement radiatif
/mnt/T2/janus-sim/src/vsl_dynamic.rs              # VSL CoupledFriedmann
/mnt/T2/janus-sim/src/snapshot_v3.rs              # Format snapshot v3
```

### Scripts Python
```
/mnt/T2/janus-sim/scripts/render_daemon_adaptive_v2.py  # Rendu frames
/mnt/T2/janus-sim/scripts/phase9_deep_analysis.py       # Analyse Janus
/mnt/T2/janus-sim/scripts/validate_ics.py               # Validation grille
```

### Output v8
```
/mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/
├── snapshots/           # ~400 fichiers snap_XXXXX.bin
├── frames_10panel/      # ~400 PNG 10-panel
├── frames_2p5d/         # ~400 PNG 2.5D
├── time_series.csv      # Métriques par step
├── run.log              # Log complet
├── README.txt           # Métadonnées run
└── PRODUCTION_ACTIVE.lock
```

### Rapports précédents
```
/mnt/T2/janus-sim/output/janus_adaptive_v7b_zmin077/FINAL_REPORT.md
/mnt/T2/janus-sim/md/V8_ZOOM_LAUNCH.md
```

---

## 7. ENVIRONNEMENT

### Python venv pour plotting
```bash
/tmp/plotenv/bin/python  # Matplotlib, numpy, scipy
```

### Docker
```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release --features cuda
```

### GPU
```
NVIDIA RTX 3060 12GB (sm_86)
CUDA 12.3
```

---

## 8. COMMANDES DE DIAGNOSTIC

```bash
# État simulation
tail -20 /mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/run.log

# Dernières métriques
tail -5 /mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/time_series.csv

# Container actif
docker ps | grep janus

# Tuer proprement
docker stop $(docker ps -q --filter name=janus-sim)

# Frames générées
ls /mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/frames_10panel/ | wc -l

# Vérifier snapshot
/tmp/plotenv/bin/python -c "
import struct, os
path = '/mnt/T2/janus-sim/output/janus_production_v8_zoom_20260424_2136/snapshots/snap_08000.bin'
with open(path, 'rb') as f:
    h = f.read(408)
    n = struct.unpack('<Q', h[16:24])[0]
    a = struct.unpack('<d', h[24:32])[0]
    print(f'N={n}, z={1/a-1:.3f}')
"
```

---

## 9. PROCHAINES ÉTAPES

1. **Attendre fin v8** (z=0, ~30 min restantes)
2. **Générer FINAL_REPORT_v8.md** avec:
   - Validation grille (max_axes)
   - Métriques Phase 9
   - Comparaison Petit
3. **Décision**: Si aucun split → ajuster seuils et relancer
4. **Vidéo**: ffmpeg sur frames_10panel

---

## 10. CONTACTS / RÉFÉRENCES

- **Petit, Margnat & Zejli (2024)** — EPJC 84:1226
- **D'Agostini & Petit (2018)** — Astrophys. Space Sci. 363:139
- **Pantheon+ (Scolnic 2022)** — ApJ 938:113

---

*Document généré automatiquement pour récupération après crash.*
