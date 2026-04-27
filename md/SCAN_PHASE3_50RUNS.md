# Phase 2C extension finale — Scan 50 runs μ ∈ [10, 60]

## Objectif

Caractériser finement la **transition dynamique Janus** dans la zone 
identifiée par les 4 IA externes (ChatGPT, Grok, Gemini, DeepSeek) comme 
physiquement intéressante : μ ∈ [10, 60].

50 runs au total répartis en :
- **25 runs zone critique** [10, 35] log-espacés
- **15 runs zone saturation** [36, 60] log-espacés  
- **10 runs CONTRÔLES** pour valider que le signal n'est pas un artefact

## Note importante

Pas d'analyse interprétative à produire par CLI. Juste lancer, attendre, 
m'envoyer le CSV. **Moi** je ferai la synthèse statistique avec test 
Poisson.

L'objectif final est une **vidéo** avec halos/amas/filaments depuis le 
run de production qui suivra (pas ce screening-ci).

---

## Configuration technique

Réutiliser la structure de `janus_screening.rs`. Créer une nouvelle 
variante `janus_screening_phase3.rs` avec ces modifications.

### 1. Liste des 50 configurations

```rust
// === PHASE 3 SCAN — 50 runs μ ∈ [10, 60] ===

// Zone critique [10, 35] : 25 valeurs log-espacées
const SCAN_CRITICAL: [f64; 25] = [
    10.00, 10.54, 11.10, 11.70, 12.32, 12.98, 13.68, 14.41, 
    15.18, 16.00, 16.85, 17.76, 18.71, 19.71, 20.77, 21.88,
    23.05, 24.29, 25.59, 26.96, 28.40, 29.93, 31.53, 33.22, 35.00
];

// Zone saturation [36, 60] : 15 valeurs log-espacées
const SCAN_SATURATION: [f64; 15] = [
    36.00, 37.34, 38.73, 40.16, 41.66, 43.21, 44.81, 46.48,
    48.20, 49.99, 51.85, 53.78, 55.78, 57.85, 60.00
];

// Contrôles
// Type A : couplage répulsif OFF (5 runs)
const CONTROL_A_MU: [f64; 5] = [2.0, 10.0, 19.0, 40.0, 60.0];

// Type B : mass_factor FIXE = 3.33 (5 runs)
const CONTROL_B_MU: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

// L commun
const SCAN_L: f64 = 100.0;  // Optimum Phase 1
```

### 2. Paramètres communs

```rust
const N_GRID: usize = 58;              // 195K particules
const Z_INIT: f64 = 4.0;
const Z_FINAL: f64 = 0.5;
const DT: f64 = 0.002;
const THETA: f64 = 0.7;
const H0: f64 = 69.9;
const OMEGA_B: f64 = 0.05;
const HUBBLE_FRICTION: f64 = 1.0;
const SEED_IC: u64 = 42;               // Single seed
const ETA_VSL: f64 = 1.045;
```

### 3. Implémentation des contrôles

**Contrôle A — Couplage répulsif OFF** : pour ces 5 runs, le couplage 
cross-espèce m+/m- doit être désactivé. Cherche dans le code 
`janus_adaptive_zoom.rs` comment activer/désactiver ce couplage. Si 
nécessaire, ajouter un paramètre booléen `cross_repulsion_enabled` :

```rust
gpu_sim.set_cross_repulsion_enabled(false);  // Pour contrôle A uniquement
```

Si ce paramètre n'existe pas dans le code GPU actuel, **ne pas implémenter 
ce contrôle** mais me le rapporter. Dans ce cas, ne fais que les 40 runs 
restants (25 critical + 15 saturation) et me préviens.

**Contrôle B — Mass_factor FIXE** : pour ces 5 runs, forcer 
`set_mass_factor(3.33)` au lieu de calculer `Ω_b × (1+μ) / 0.3`. Plus 
simple à implémenter — un boolean dans la struct de configuration.

### 4. Structure du loop principal

```rust
fn main() {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M");
    let outdir = format!("/app/output/scan_phase3_{}", timestamp);
    fs::create_dir_all(&format!("{}/snapshots", outdir)).unwrap();
    
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SCAN PHASE 3 — μ ∈ [10, 60] avec contrôles (50 runs)                    ║");
    println!("║  Output : {}                                                              ║", outdir);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    
    let cuda_device = GpuNBodySimulation::compile_kernels()
        .expect("Failed to compile CUDA kernels");
    println!("  ✓ CUDA kernels compiled (one-time)");
    
    let csv_path = format!("{}/scan_phase3_results.csv", outdir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "phase,run_id,label,mu,L_box,N_init,mass_factor_used,cross_repulsion,\
                   z_final_reached,\
                   v_rms_z3,v_rms_z2,v_rms_z15,v_rms_z1,v_rms_zfinal,\
                   rho_plus_max_z3,rho_plus_max_z2,rho_plus_max_z15,rho_plus_max_z1,rho_plus_max_zfinal,\
                   rho_max_z3,rho_max_z2,rho_max_z15,rho_max_z1,rho_max_zfinal,\
                   rho_plus_mean,n_overdense_zfinal,wall_time_sec,status").unwrap();
    csv.flush().unwrap();
    
    let global_start = Instant::now();
    let mut run_id = 0u32;
    
    // ─── 25 runs critical zone ───
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Zone critique [10, 35] — 25 runs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    for &mu in SCAN_CRITICAL.iter() {
        run_id += 1;
        let label = format!("crit_mu{:.2}", mu);
        let result = run_simulation(
            &cuda_device, run_id, "critical", label, mu, SCAN_L,
            -1.0,   // mass_factor calculé (pas forcé)
            true,   // cross_repulsion ON
            &outdir
        );
        write_csv_row(&mut csv, &result);
        log_progress(run_id, 50, &result, global_start);
    }
    
    // ─── 15 runs saturation zone ───
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Zone saturation [36, 60] — 15 runs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    for &mu in SCAN_SATURATION.iter() {
        run_id += 1;
        let label = format!("sat_mu{:.2}", mu);
        let result = run_simulation(
            &cuda_device, run_id, "saturation", label, mu, SCAN_L,
            -1.0, true, &outdir
        );
        write_csv_row(&mut csv, &result);
        log_progress(run_id, 50, &result, global_start);
    }
    
    // ─── 5 runs CONTROL A (cross repulsion OFF) ───
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Contrôle A — couplage répulsif OFF — 5 runs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    for &mu in CONTROL_A_MU.iter() {
        run_id += 1;
        let label = format!("ctrlA_mu{:.0}_norepulse", mu);
        let result = run_simulation(
            &cuda_device, run_id, "control_A", label, mu, SCAN_L,
            -1.0, false, &outdir   // cross_repulsion OFF
        );
        write_csv_row(&mut csv, &result);
        log_progress(run_id, 50, &result, global_start);
    }
    
    // ─── 5 runs CONTROL B (mass_factor FIXÉ) ───
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Contrôle B — mass_factor FIXÉ à 3.33 — 5 runs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    for &mu in CONTROL_B_MU.iter() {
        run_id += 1;
        let label = format!("ctrlB_mu{:.0}_mfFIX", mu);
        let result = run_simulation(
            &cuda_device, run_id, "control_B", label, mu, SCAN_L,
            3.33, true, &outdir   // mass_factor FORCÉ
        );
        write_csv_row(&mut csv, &result);
        log_progress(run_id, 50, &result, global_start);
    }
    
    let total_h = global_start.elapsed().as_secs_f64() / 3600.0;
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SCAN PHASE 3 COMPLETE                                                   ║");
    println!("║  Total time: {:.2}h ({} runs)                                             ║", total_h, run_id);
    println!("║  Results: {}/scan_phase3_results.csv                                      ║", outdir);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
```

### 5. Modification de `run_simulation`

Ajouter deux paramètres :

```rust
fn run_simulation(
    cuda_device: &Arc<CudaDevice>,
    run_id: u32,
    phase_name: &str,
    label: String,
    mu: f64,
    l_box: f64,
    mass_factor_override: f64,    // -1.0 = calculer, sinon forcer
    cross_repulsion: bool,         // true = standard, false = contrôle A
    outdir: &str,
) -> RunResult {
    // ... ICs Zel'dovich (identique) ...
    
    // ... Init GPU (identique) ...
    
    // === MASS FACTOR ===
    let janus_mass_factor = if mass_factor_override > 0.0 {
        mass_factor_override
    } else {
        OMEGA_B * (1.0 + mu) / 0.3
    };
    gpu_sim.set_mass_factor(janus_mass_factor);
    
    // === CROSS REPULSION ===
    if !cross_repulsion {
        gpu_sim.set_cross_repulsion_enabled(false);  // Si pas implémenté → erreur compilation
    }
    
    // ... reste du run identique ...
}
```

---

## Étapes pour CLI

### 1. Vérifier l'état du code

Avant de coder, vérifier dans `janus_adaptive_zoom.rs` ou les modules GPU 
si `set_cross_repulsion_enabled` ou équivalent existe :

```bash
cd /mnt/T2/janus-sim
grep -r "cross_repulsion\|cross_force\|mu_repulse" src/
```

**Si la fonction n'existe PAS** :
- Ne pas implémenter le contrôle A (couplage off)
- Faire seulement 45 runs (25 + 15 + 5 contrôle B)
- Me prévenir dans le rapport

**Si la fonction existe** :
- Implémenter les 50 runs complets

### 2. Créer le binaire

```bash
cp src/bin/janus_screening.rs src/bin/janus_screening_phase3.rs
```

Modifier selon les specs ci-dessus.

### 3. Ajouter dans Cargo.toml

```toml
[[bin]]
name = "janus_screening_phase3"
path = "src/bin/janus_screening_phase3.rs"
```

### 4. Compiler

```bash
cargo build --release --features cuda --bin janus_screening_phase3 2>&1 | tail -10
```

### 5. Test sanity (optionnel mais recommandé)

Lancer 1 run de test pour vérifier que tout fonctionne :
```bash
# Modifier temporairement les arrays pour 1 seul run
# OU laisser tourner et killer après 10 min
```

### 6. Lancer

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)

./target/release/janus_screening_phase3 2>&1 | tee /tmp/scan_phase3_${TIMESTAMP}.log &
SCREEN_PID=$!
echo "$SCREEN_PID" > /tmp/scan_phase3.pid
echo "Scan Phase 3 lancé : PID=$SCREEN_PID"
```

### 7. Surveillance

```bash
# Suivi en temps réel
tail -f /tmp/scan_phase3_*.log

# Voir avancement
RESULTS=$(ls -td /app/output/scan_phase3_*/scan_phase3_results.csv | head -1)
echo "Lignes CSV : $(wc -l < $RESULTS) (1 header + 50 max)"

# Quels phases done ?
awk -F, 'NR>1 {print $1}' "$RESULTS" | sort | uniq -c
```

---

## Estimation budget

- 50 runs × ~6 min = **~5h GPU**
- Compilation : ~2 min
- Total : ~5h

Tu as 6h disponibles → **marge confortable**.

---

## Limites strictes

1. **Pas de modification de `janus_adaptive_zoom.rs`** — nouveau binaire indépendant
2. **L = 100 Mpc fixe** (optimum Phase 1)
3. **Single seed = 42** (cohérent avec Phase 1)
4. **Hubble friction = 1.0** (vraie physique)
5. **CSV flushé après CHAQUE run** (suivi temps réel)
6. **Auto-stop sur instabilité** (passe au suivant, ne crash pas tout)
7. **Pas de rapport interprétatif par CLI** — juste le CSV à la fin

---

## Après le scan

À la fin (~5h), m'envoyer **uniquement** :
- `scan_phase3_results.csv` (le CSV avec les 50 lignes)
- Le log si quelque chose d'anormal (crash, NaN, etc.)

Je ferai :
- Plot ratio(μ) sur la zone fine
- Test Poisson rigoureux sur chaque point
- Identification de la transition de phase exacte
- Validation par les contrôles A et B
- Recommandation finale pour Phase 2D (run de production avec vidéo)

**Pas besoin de conclusions hâtives**. Le but est juste de cartographier 
finement, pas de conclure.
