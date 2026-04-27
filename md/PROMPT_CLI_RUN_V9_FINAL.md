# Phase 2D — Production Run v9 — Multi-μ campaign

**Date** : 26 avril 2026  
**Status** : Lancement après validation Jeans Swindle (déjà fait par CLI)  
**Budget** : 2 nuits GPU × ~18-19h chacune

---

## Configuration finale (validée par 4 IA externes + AJP)

```
PARAMÈTRES PHYSIQUES :
  L            = 200 Mpc
  N_init       = 10M (n_grid=215, 215³ = 9,938,375)
  z_init       = 10.0
  z_final      = 0.0
  dt           = 0.001 (Gyr)
  Hubble       = 1.0 (vraie physique)
  Ω_b          = 0.05
  H₀           = 69.9 km/s/Mpc
  η_VSL        = 1.045
  ε_plus       = 0.05 × spacing
  ε_minus      = 0.10 × spacing
  
INITIAL CONDITIONS :
  Method       = Zel'dovich FFT
  δ_rms        = 0.15  (compromis : >0.1 v8, <0.2 limite linéaire)
  seed         = 42
  
REFINEMENT :
  max_split_level = 0   (APR rejeté par 4 IA, uniforme)
  
SORTIES :
  snap_interval        = 20 steps    (~450 snapshots par run)
  csv_logging_interval = 10 steps    (haute résolution time series)
  
DIAGNOSTIC AJOUTÉ (idée ChatGPT) :
  E_pp / E_pm          = ratio énergie attractive intra-m+ / répulsive m+/m-
  Calculé toutes les 50 steps, loggé dans time_series.csv
```

---

## ⚠️ Modifications nécessaires dans le code AVANT lancement

### Modification 1 — Paramètre `--delta-rms-ic`

Vérifier dans `janus_adaptive_zoom.rs` que ce paramètre existe. Si non, l'ajouter :

```rust
#[arg(long, default_value = "0.1")]
delta_rms_ic: f64,
```

Et l'utiliser dans la génération des ICs Zel'dovich (`generate_zeldovich_ic` ou équivalent).

**Si le paramètre n'existe pas et est compliqué à ajouter** : modifier en dur la constante dans le code (DELTA_RMS = 0.15), recompiler.

### Modification 2 — Paramètre `--enable-energy-diagnostic`

Nouveau paramètre booléen. Quand activé, calcule toutes les 50 steps :

```rust
fn compute_energy_diagnostic(particles: &[Particle], box_size: f64, t_step: u64) -> EnergyDiagnostic {
    let mut e_pp_attract = 0.0;  // énergie attractive intra-m+
    let mut e_pm_repulse = 0.0;  // énergie répulsive m+/m-
    
    // Sample-based pour économiser : 10K paires aléatoires
    // au lieu de toutes les paires (O(N²) prohibitif à N=10M)
    
    let n_samples = 10000;
    let mut rng = StdRng::seed_from_u64(t_step);  // reproductible
    
    for _ in 0..n_samples {
        let i = rng.gen_range(0..particles.len());
        let j = rng.gen_range(0..particles.len());
        if i == j { continue; }
        
        let r = distance_periodic(particles[i].pos, particles[j].pos, box_size);
        if r < 0.1 { continue; }  // skip pairs trop proches
        
        let m_i = particles[i].mass;
        let m_j = particles[j].mass;
        let s_i = particles[i].sign;
        let s_j = particles[j].sign;
        
        let e = m_i * m_j / r;  // énergie pondérée (pas le facteur G ni le signe physique)
        
        if s_i == 1 && s_j == 1 {
            e_pp_attract += e;
        } else if (s_i == 1 && s_j == -1) || (s_i == -1 && s_j == 1) {
            e_pm_repulse += e;
        }
        // m-/m- : pas tracké pour ce diagnostic
    }
    
    EnergyDiagnostic {
        e_pp_attract,
        e_pm_repulse,
        ratio: e_pm_repulse / e_pp_attract.max(1e-30),
    }
}
```

Logger dans `time_series.csv` :
```
step, z, ..., e_pp_attract, e_pm_repulse, ratio_pm_pp
```

### Estimation dev

- Modif 1 (delta_rms) : 30 min (paramètre simple)
- Modif 2 (energy diagnostic) : 1h-1h30 (sample-based, attention seed reproductible)
- Compilation + test : 30 min

**Total dev : ~2h**

Si CLI préfère ne pas faire ces modifications, **le run v9 peut quand même tourner** avec :
- `δ_rms = 0.1` par défaut (au lieu de 0.15) — perte mineure de signal attendu
- Pas de diagnostic E_pp/E_pm — juste manque d'info interprétative

À voir avec CLI selon le temps dispo.

---

## Stratégie en 2 nuits GPU

### Nuit 1 — Run μ=8

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN1_DIR="/app/output/janus_v9_mu8_${TIMESTAMP}"
mkdir -p "$RUN1_DIR"
touch "$RUN1_DIR/PRODUCTION_ACTIVE.lock"

cat > "$RUN1_DIR/README.txt" << EOF
=== JANUS v9 — Run 1/2 — μ=8 ===
Started: $(date)
Configuration:
  μ = 8 (Petit conservative value)
  L = 200 Mpc, N_init = 10M, z = 10 → 0
  δ_rms = 0.15 (ICs)
  Hubble friction = 1.0
  E_pp/E_pm diagnostic enabled

Expected: m+ fraction = 11.1% (N+ ≈ 1.1M particles)
mass_factor = 0.05 × 9 / 0.3 = 1.50
EOF

./target/release/janus_adaptive_zoom \
  --n-grid 215 \
  --l-box 200 \
  --z-init 10.0 \
  --z-final 0.0 \
  --mu 8.0 \
  --h0 69.9 \
  --omega-b 0.05 \
  --eps-plus 0.05 \
  --eps-minus 0.10 \
  --dt-max 0.001 \
  --theta 0.7 \
  --delta-rms-ic 0.15 \
  --snap-interval 20 \
  --steps-check 50 \
  --max-split-level 0 \
  --enable-energy-diagnostic \
  --out-dir "$RUN1_DIR" \
  --run-label "v9_mu8_L200_N10M" \
  2>&1 | tee "$RUN1_DIR/run.log"

echo "Run 1 terminé : $(date)"
```

### Nuit 2 — Run μ=30

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN2_DIR="/app/output/janus_v9_mu30_${TIMESTAMP}"
mkdir -p "$RUN2_DIR"
touch "$RUN2_DIR/PRODUCTION_ACTIVE.lock"

cat > "$RUN2_DIR/README.txt" << EOF
=== JANUS v9 — Run 2/2 — μ=30 ===
Started: $(date)
Configuration:
  μ = 30 (zone of n_overdense peak in Phase 2C screening)
  L = 200 Mpc, N_init = 10M, z = 10 → 0
  δ_rms = 0.15 (ICs, same seed as Run 1)
  Hubble friction = 1.0
  E_pp/E_pm diagnostic enabled

Expected: m+ fraction = 3.2% (N+ ≈ 323K particles)
mass_factor = 0.05 × 31 / 0.3 = 5.17
EOF

./target/release/janus_adaptive_zoom \
  --n-grid 215 \
  --l-box 200 \
  --z-init 10.0 \
  --z-final 0.0 \
  --mu 30.0 \
  --h0 69.9 \
  --omega-b 0.05 \
  --eps-plus 0.05 \
  --eps-minus 0.10 \
  --dt-max 0.001 \
  --theta 0.7 \
  --delta-rms-ic 0.15 \
  --snap-interval 20 \
  --steps-check 50 \
  --max-split-level 0 \
  --enable-energy-diagnostic \
  --out-dir "$RUN2_DIR" \
  --run-label "v9_mu30_L200_N10M" \
  2>&1 | tee "$RUN2_DIR/run.log"

echo "Run 2 terminé : $(date)"
```

---

## Surveillance pendant les runs

### Critères d'arrêt automatique

```bash
# Surveiller toutes les 1-2h pendant le run

RUN_DIR=$(ls -td /app/output/janus_v9_*/ | head -1)
TS_CSV="$RUN_DIR/time_series.csv"

# Dernière ligne
tail -1 "$TS_CSV"

# Critères KILL si :
#  - N_total > 12M (impossible sans splits, indique bug)
#  - v_rms > 50000 km/s (relativiste = bug)
#  - "nan" ou "NaN" dans CSV
grep -c "nan\|NaN" "$TS_CSV"

# Espace disque
df -h /app

# Évolution N_total et métriques
awk -F, 'NR%500==1 {print "step="$1, "z="$3, "N="$5, "vrms="$8, "rho+_max="$9}' "$TS_CSV"
```

### Ce qui doit se passer (référence v8 extrapolée)

À z=5 :
- v_rms ≈ 50-150 km/s (selon μ)
- ρ+_max/ρ+_mean ≈ 5-15
- Pas de NaN

À z=2 :
- v_rms ≈ 100-300 km/s
- ρ+_max/ρ+_mean ≈ 10-30

À z=0 :
- v_rms ≈ 200-500 km/s (μ=8) à 500-1500 km/s (μ=30)
- ρ+_max/ρ+_mean ≥ 20 (si Janus marche)
- ρ+_max/ρ+_mean ≤ 10 (si pas de signal physique)

---

## Important pour CLI

**PAS DE RAPPORT INTERPRÉTATIF.**

Juste lancer, attendre, m'envoyer les data. Toutes les conclusions seront tirées par AJP/Claude sur la base des CSV et des snapshots, après les analyses Python (FoF, P(k), r(k)) et les vidéos qu'AJP fera lui-même.

Les rapports interprétatifs précédents de CLI avaient des conclusions trop hâtives ("CASE B → TRUE JANUS EFFECT" sans test Poisson). Cette fois, on fait l'analyse proprement, en aval.

---

## Que livrer à la fin des 2 runs

À la fin de la campagne, AJP envoie à Claude :

1. **Les 2 CSV time_series** complets (Run μ=8 et Run μ=30)
2. **Le snap final z=0** des deux runs
3. **5-10 snapshots intermédiaires** par run (z=5, 3, 2, 1, 0.5, 0)
4. **Les logs run.log** des 2 runs

AJP fera lui-même les analyses Python (FoF, P(k), r(k)) et les 2 vidéos MP4 avec ses scripts existants.

Claude fera la **synthèse comparative** μ=8 vs μ=30 sur la base des CSV et des résultats des analyses Python d'AJP.

---

## Récap budget

| Item | Temps |
|------|-------|
| Modifications code (delta_rms + diagnostic) | ~2h dev |
| Compilation et test rapide | ~30 min |
| Run 1 (μ=8) | ~18-19h GPU |
| Run 2 (μ=30) | ~18-19h GPU |
| **Total côté CLI** | **~40h** |

Étalé sur 2-3 nuits GPU.

**Disque** : 
- Snapshot taille : 10M × 36 bytes ≈ 360 MB
- 450 snapshots/run × 360 MB ≈ 162 GB par run
- 2 runs : ~324 GB sur 1.5 TB disponibles (marge ~80%)
