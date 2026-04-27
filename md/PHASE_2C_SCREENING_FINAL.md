# Phase 2C — Screening Janus — VERSION CORRIGÉE

**Date** : 25 avril 2026  
**Status** : Plan révisé après critique de 3 IA externes (ChatGPT, Grok, Gemini)

## Stratégie

**Phase 0 (sentinelles)** + **Phase 1 (scan 2D)** intégrés dans UN seul binaire 
Rust qui boucle, avec **arrêt automatique sur critère** entre les deux.

- **Phase 0** : 4 sentinelles diagnostiques (1-3h GPU) pour valider la config
- **Phase 1** : si critère sentinelle OK → scan 8 μ × 3 L = 24 runs (~10h GPU)

Total : ~13h GPU si tout va bien, beaucoup moins en cas d'échec sentinelles.

---

## Étape 0 — Killer le run actuel (si pas déjà fait)

```bash
PID_FILE=$(ls /app/output/janus_invitro_L100_*/PID 2>/dev/null | head -1)
if [ -n "$PID_FILE" ]; then
    kill $(cat "$PID_FILE") 2>/dev/null
    sleep 5
    OLD_DIR=$(dirname "$PID_FILE")
    rm -f "$OLD_DIR/PRODUCTION_ACTIVE.lock"
    echo "Killed run in $OLD_DIR"
fi

# Vérifier que le code est propre (revert post-Test 1B)
cd /mnt/T2/janus-sim
git status src/bin/janus_adaptive_zoom.rs
# Doit afficher "nothing to commit"
```

---

## Étape 1 — Création du binaire `janus_screening.rs`

**Localisation** : `/mnt/T2/janus-sim/src/bin/janus_screening.rs`

Réutilise massivement le code de `janus_adaptive_zoom.rs` :
- ICs Zel'dovich (`generate_zeldovich_ics`)
- GpuNBodySimulation
- GpuCooling
- VSL dynamique
- Fonctions de capture metrics (compute_vrms, etc.)

### Configuration (hardcoded)

```rust
// ════════════════════════════════════════════════════════════════
// JANUS SCREENING — CONFIGURATION (hardcoded, pas de CLI args)
// ════════════════════════════════════════════════════════════════

// Paramètres communs à tous les runs
const N_GRID: usize = 58;              // 58³ = 195_112 particules
const Z_INIT: f64 = 4.0;
const Z_FINAL: f64 = 0.5;
const DT: f64 = 0.002;
const THETA: f64 = 0.7;
const H0: f64 = 69.9;
const OMEGA_B: f64 = 0.05;
const ETA_VSL: f64 = 1.045;
const HUBBLE_FRICTION: f64 = 1.0;      // Vraie physique
const T_INIT: f64 = 10000.0;
const SEED_IC: u64 = 42;               // SINGLE seed pour tous

// IMPORTANT : mass_factor VARIABLE = Ω_b × (1+μ) / 0.3
// Calculé par run, pas constante. Physique correcte.

// Auto-stop
const V_RMS_HARD_LIMIT: f64 = 50000.0; // km/s
const MAX_STEPS: usize = 5000;

// ════════════════════════════════════════════════════════════════
// PHASE 0 — SENTINELLES (4 runs diagnostiques)
// ════════════════════════════════════════════════════════════════
const SENTINEL_CONFIGS: [(f64, f64, &str); 4] = [
    (2.0,  100.0, "S1_mu2_L100"),    // doit collapser fortement
    (2.0,  50.0,  "S2_mu2_L50"),     // doit collapser
    (19.0, 50.0,  "S3_mu19_L50"),    // référence Janus canonique petit-L
    (19.0, 100.0, "S4_mu19_L100"),   // reproduction in vitro précédent
];

// Critère GO/NO-GO :
// Si S1 (μ=2, L=100) ne dépasse PAS ρ_plus_max > 5×ρ_plus_mean(μ=2) à z=0.5
// → ARRÊT, le code/setup a un problème
const SENTINEL_RHO_RATIO_THRESHOLD: f64 = 5.0;  // au moins 5× ρ_mean+
const SENTINEL_VRMS_THRESHOLD: f64 = 30.0;       // km/s à z_final

// ════════════════════════════════════════════════════════════════
// PHASE 1 — SCAN PRINCIPAL (8 μ × 3 L = 24 runs)
// ════════════════════════════════════════════════════════════════
const SCAN_MU: [f64; 8] = [1.5, 2.5, 4.0, 6.5, 10.0, 16.0, 25.0, 40.0];
const SCAN_L:  [f64; 3] = [50.0, 100.0, 200.0];
// Total : 24 runs
```

### Structure des données

```rust
#[derive(Default, Clone, Copy)]
struct ZMetrics {
    z: f64,
    v_rms: f64,
    rho_plus_max: f64,
    rho_max: f64,
}

struct RunResult {
    phase: u8,                // 0 = sentinelle, 1 = scan
    run_id: u32,
    label: String,
    mu: f64,
    l_box: f64,
    n_init: usize,
    z_final_reached: f64,
    
    // Métriques aux 4 z cibles + final
    metrics_z3: ZMetrics,
    metrics_z2: ZMetrics,
    metrics_z15: ZMetrics,
    metrics_z1: ZMetrics,
    metrics_zfinal: ZMetrics,
    
    n_overdense_zfinal: u32,    // nb cellules avec δ+ > 10 sur grille 32³
    rho_plus_mean: f64,         // calculé : Ω_b × ρ_crit
    
    wall_time_sec: f64,
    status: String,
}
```

### Logique principale du `main()`

```rust
fn main() {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M");
    let outdir = format!("/app/output/screening_{}", timestamp);
    fs::create_dir_all(&format!("{}/snapshots", outdir)).unwrap();
    
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS SCREENING — Phase 0 (sentinelles) + Phase 1 (scan)                ║");
    println!("║  Output : {}                                              ║", outdir);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    
    // Compile CUDA UNE FOIS
    let cuda_device = GpuNBodySimulation::compile_kernels()
        .expect("Failed to compile CUDA kernels");
    println!("  ✓ CUDA kernels compiled (one-time)");
    
    // CSV global
    let csv_path = format!("{}/screening_results.csv", outdir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "phase,run_id,label,mu,L_box,N_init,z_final_reached,\
                   v_rms_z3,v_rms_z2,v_rms_z15,v_rms_z1,v_rms_zfinal,\
                   rho_plus_max_z3,rho_plus_max_z2,rho_plus_max_z15,rho_plus_max_z1,rho_plus_max_zfinal,\
                   rho_max_z3,rho_max_z2,rho_max_z15,rho_max_z1,rho_max_zfinal,\
                   rho_plus_mean,n_overdense_zfinal,wall_time_sec,status").unwrap();
    csv.flush().unwrap();
    
    let global_start = Instant::now();
    let mut total_runs_done = 0u32;
    
    // ─────────────────────────────────────────────
    // PHASE 0 — SENTINELLES
    // ─────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 0 — Sentinelles diagnostiques (4 runs)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    let mut sentinel_results: Vec<RunResult> = Vec::new();
    
    for (i, (mu, l_box, label)) in SENTINEL_CONFIGS.iter().enumerate() {
        total_runs_done += 1;
        println!("[Sentinel {}/4] μ={}, L={} ({})", i+1, mu, l_box, label);
        
        let result = run_single_simulation(
            &cuda_device, 0, total_runs_done, label.to_string(),
            *mu, *l_box, &outdir
        );
        
        write_csv_row(&mut csv, &result);
        let collapse_ratio = result.metrics_zfinal.rho_plus_max / result.rho_plus_mean;
        println!("  → ρ+_max/ρ+_mean = {:.1}, v_rms = {:.0} km/s, status = {}",
            collapse_ratio, result.metrics_zfinal.v_rms, result.status);
        
        sentinel_results.push(result);
    }
    
    // ─────────────────────────────────────────────
    // CRITÈRE GO/NO-GO sur sentinelles
    // ─────────────────────────────────────────────
    let s1 = &sentinel_results[0];  // μ=2, L=100
    let s1_ratio = s1.metrics_zfinal.rho_plus_max / s1.rho_plus_mean;
    let s1_vrms = s1.metrics_zfinal.v_rms;
    
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CRITÈRE GO/NO-GO");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  S1 (μ=2, L=100) : ρ+_max/ρ+_mean = {:.2} (seuil {})", 
             s1_ratio, SENTINEL_RHO_RATIO_THRESHOLD);
    println!("  S1 (μ=2, L=100) : v_rms = {:.0} km/s (seuil {})", 
             s1_vrms, SENTINEL_VRMS_THRESHOLD);
    
    let go = s1_ratio >= SENTINEL_RHO_RATIO_THRESHOLD 
          && s1_vrms >= SENTINEL_VRMS_THRESHOLD
          && s1.status == "OK";
    
    if !go {
        println!("\n❌ NO-GO : S1 (μ=2, L=100) ne collapse pas suffisamment.");
        println!("   Le scan complet ne sera PAS lancé.");
        println!("   Vérifier configuration : Hubble friction, ICs, mass_factor variable.");
        
        let summary_path = format!("{}/SENTINEL_DIAGNOSIS.txt", outdir);
        let mut diag = File::create(&summary_path).unwrap();
        writeln!(diag, "SENTINEL DIAGNOSIS").unwrap();
        writeln!(diag, "==================").unwrap();
        writeln!(diag, "S1 (μ=2, L=100) : ρ+_max/ρ+_mean = {:.2}, v_rms = {:.0}",
                 s1_ratio, s1_vrms).unwrap();
        writeln!(diag, "Threshold : {} for ratio, {} for v_rms",
                 SENTINEL_RHO_RATIO_THRESHOLD, SENTINEL_VRMS_THRESHOLD).unwrap();
        writeln!(diag, "Result : NO-GO, scan not launched").unwrap();
        
        std::process::exit(1);
    }
    
    println!("\n✅ GO : sentinelles validées, scan principal lancé.");
    
    // ─────────────────────────────────────────────
    // PHASE 1 — SCAN 2D
    // ─────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 1 — Scan principal {} μ × {} L = {} runs", 
             SCAN_MU.len(), SCAN_L.len(), SCAN_MU.len() * SCAN_L.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    let total_scan_runs = SCAN_MU.len() * SCAN_L.len();
    let mut scan_run_idx = 0u32;
    
    for &mu in &SCAN_MU {
        for &l_box in &SCAN_L {
            scan_run_idx += 1;
            total_runs_done += 1;
            let label = format!("scan_mu{:.1}_L{:.0}", mu, l_box);
            
            println!("[Scan {}/{}] μ={:.1}, L={:.0} ({})",
                scan_run_idx, total_scan_runs, mu, l_box, label);
            
            let result = run_single_simulation(
                &cuda_device, 1, total_runs_done, label,
                mu, l_box, &outdir
            );
            
            write_csv_row(&mut csv, &result);
            
            let collapse_ratio = result.metrics_zfinal.rho_plus_max / result.rho_plus_mean;
            let elapsed_h = global_start.elapsed().as_secs_f64() / 3600.0;
            let avg = elapsed_h / total_runs_done as f64;
            let remaining = avg * (total_scan_runs - scan_run_idx as usize) as f64;
            
            println!("  → ratio={:.1}, v_rms={:.0} | status={} | elapsed={:.1}h | ETA +{:.1}h",
                collapse_ratio, result.metrics_zfinal.v_rms,
                result.status, elapsed_h, remaining);
        }
    }
    
    let total_h = global_start.elapsed().as_secs_f64() / 3600.0;
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SCREENING COMPLETE                                                      ║");
    println!("║  Total time: {:.2}h                                                       ║", total_h);
    println!("║  Total runs: {} (4 sentinelles + {} scan)                                 ║", 
             total_runs_done, total_scan_runs);
    println!("║  Results: {}/screening_results.csv                                       ║", outdir);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
```

### Fonction `run_single_simulation`

```rust
fn run_single_simulation(
    cuda_device: &Arc<CudaDevice>,
    phase: u8,
    run_id: u32,
    label: String,
    mu: f64,
    l_box: f64,
    outdir: &str,
) -> RunResult {
    let start = Instant::now();
    let mut result = RunResult {
        phase, run_id, label: label.clone(),
        mu, l_box, n_init: 0, z_final_reached: Z_INIT,
        metrics_z3: Default::default(),
        metrics_z2: Default::default(),
        metrics_z15: Default::default(),
        metrics_z1: Default::default(),
        metrics_zfinal: Default::default(),
        n_overdense_zfinal: 0,
        rho_plus_mean: 0.0,
        wall_time_sec: 0.0,
        status: "INIT".to_string(),
    };
    
    // Compute ρ+_mean cosmologique pour cette config
    let rho_crit = 2.775e11 * (H0/100.0).powi(2);
    result.rho_plus_mean = OMEGA_B * rho_crit;  // ρ_b à z=0, comobile
    
    // ICs Zel'dovich (réutilise generate_zeldovich_ics)
    // Le seed est SEED_IC, pas le run_id !
    let (positions, velocities, signs) = match generate_zeldovich_ics_screening(
        N_GRID, l_box, Z_INIT, mu, SEED_IC
    ) {
        Ok(v) => v,
        Err(e) => {
            result.status = format!("IC_FAIL:{}", e);
            result.wall_time_sec = start.elapsed().as_secs_f64();
            return result;
        }
    };
    
    let n_plus = signs.iter().filter(|&&s| s == 1).count();
    let n_minus = signs.len() - n_plus;
    result.n_init = n_plus + n_minus;
    
    if n_plus < 100 {
        result.status = format!("TOO_FEW_PLUS:{}", n_plus);
        result.wall_time_sec = start.elapsed().as_secs_f64();
        return result;
    }
    
    // Init GPU
    let mut gpu_sim = match GpuNBodySimulation::new_with_state(
        n_plus, n_minus, l_box, positions, velocities, signs.clone()
    ) {
        Ok(s) => s,
        Err(e) => {
            result.status = format!("GPU_INIT_FAIL:{}", e);
            result.wall_time_sec = start.elapsed().as_secs_f64();
            return result;
        }
    };
    
    let eps_plus = 0.05 * l_box / 100.0;
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(eps_plus);
    
    // ⚠️ MASS_FACTOR VARIABLE — physique correcte
    let janus_mass_factor = OMEGA_B * (1.0 + mu) / 0.3;
    gpu_sim.set_mass_factor(janus_mass_factor);
    
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA_VSL);
    gpu_sim.set_c_ratio(c_ratio_sq_init.sqrt());
    
    // Cosmological state
    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.5;
    
    // Z capture targets
    let z_targets = [3.0, 2.0, 1.5, 1.0];
    let mut next_target_idx = 0;
    
    let mut step = 0usize;
    loop {
        let z = 1.0 / a - 1.0;
        if z <= Z_FINAL {
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z);
            result.n_overdense_zfinal = count_overdense_cells(&mut gpu_sim, &signs, l_box, 32, 10.0);
            break;
        }
        if step >= MAX_STEPS {
            result.status = "MAX_STEPS".to_string();
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z);
            break;
        }
        
        while next_target_idx < z_targets.len() && z <= z_targets[next_target_idx] {
            let z_cap = z_targets[next_target_idx];
            let m = capture_metrics(&mut gpu_sim, &signs, l_box, z_cap);
            
            match next_target_idx {
                0 => result.metrics_z3 = m,
                1 => result.metrics_z2 = m,
                2 => result.metrics_z15 = m,
                3 => result.metrics_z1 = m,
                _ => {}
            }
            next_target_idx += 1;
        }
        
        let h = compute_friedmann_h(z, mu);
        
        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, HUBBLE_FRICTION) {
            result.status = format!("STEP_FAIL:{}", e);
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z);
            break;
        }
        
        if step % 100 == 0 {
            if let Ok(v) = gpu_sim.compute_vrms() {
                if v > V_RMS_HARD_LIMIT || v.is_nan() {
                    result.status = format!("INSTABILITY:v_rms={:.0}", v);
                    result.z_final_reached = z;
                    break;
                }
            }
        }
        
        if step % 50 == 0 {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA_VSL);
            gpu_sim.set_c_ratio(c_ratio_sq.sqrt());
        }
        
        a += a * h * DT;
        t_gyr += DT;
        step += 1;
    }
    
    if result.status == "INIT" {
        let snap_path = format!("{}/snapshots/snap_{}_{}.bin", 
            outdir, run_id, label);
        let _ = save_snapshot_minimal(&gpu_sim, &snap_path, l_box);
        result.status = "OK".to_string();
    }
    
    result.wall_time_sec = start.elapsed().as_secs_f64();
    result
}
```

### Fonctions helper à implémenter

**`generate_zeldovich_ics_screening`** : copie de `generate_zeldovich_ics` 
existante adaptée pour prendre `seed` et `mu` en arguments, retourner Result.

**`capture_metrics`** : à implémenter. Calcule sur grille 32³ :
- v_rms : sqrt(<v²>) sur toutes les particules
- ρ_max : max sur les cellules
- ρ+_max : max sur les cellules en ne comptant que les m+

**`count_overdense_cells`** : compte les cellules de la grille 32³ où la 
densité m+ dépasse `factor × ρ_plus_mean`

**`save_snapshot_minimal`** : sauvegarde compacte (positions + signs + masses)

**`compute_friedmann_h`** : récupérer depuis `vsl_dynamic.rs` ou 
`friedmann.rs`. La formule doit dépendre de μ pour Janus.

**`write_csv_row`** : écrit une ligne CSV depuis un RunResult.

### Modifications dans Cargo.toml

```toml
[[bin]]
name = "janus_screening"
path = "src/bin/janus_screening.rs"
```

Si nécessaire :
```toml
chrono = "0.4"
```

---

## Étape 2 — Compilation

```bash
cd /mnt/T2/janus-sim
cargo build --release --features cuda --bin janus_screening 2>&1 | tail -20
```

**Si compilation échoue** → me rapporter les erreurs avant de lancer.

---

## Étape 3 — Lancement

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)

./target/release/janus_screening 2>&1 | tee /tmp/screening_${TIMESTAMP}.log &
SCREEN_PID=$!
echo "$SCREEN_PID" > /tmp/screening.pid
echo "Screening lancé : PID=$SCREEN_PID"
```

---

## Étape 4 — Surveillance

```bash
# Suivi en temps réel
tail -f /tmp/screening_*.log

# Voir avancement runs
RESULTS=$(ls -td /app/output/screening_*/screening_results.csv | head -1)
echo "Lignes CSV : $(wc -l < $RESULTS) (header + 4 sentinelles + 24 scan = 29 max)"

# Status par run
awk -F, 'NR>1 {print $1, $3, $NF}' "$RESULTS"

# Configs où ρ+_max > 5× ρ_mean (potentiel collapse)
awk -F, 'NR>1 && $24 > 0 {ratio = $17/$23; if (ratio > 5) print $3, "ratio="ratio, "vrms="$12}' "$RESULTS"
```

---

## Étape 5 — Que faire après

Quand le binaire s'arrête (succès complet ou échec sentinelles) :

**Cas A — NO-GO sur sentinelles**
- Lire `SENTINEL_DIAGNOSIS.txt`
- Vérifier S1 (μ=2, L=100) : pourquoi ne collapse-t-il pas ?
- Possibles : Hubble friction mal implémentée, ICs corrompues, mass_factor variable mal géré
- M'envoyer le CSV + log → diagnostic conjoint

**Cas B — GO mais scan ne montre aucune transition**
- Toutes les configs ont collapse ratio ~1 → résultat publiable mais négatif
- Considérer Phase 2 zoom haute résolution sur la zone marginale

**Cas C — GO et transition observée**
- Identifier le (μ, L) à la transition
- Phase 2 production : run complet z=4→0 sur cette config
- Préparation preprint

Dans tous les cas : m'envoyer `screening_results.csv` pour analyse heatmap.

---

## Limites strictes

1. **Ne PAS modifier `janus_adaptive_zoom.rs`**
2. **Mass_factor VARIABLE** : `Ω_b × (1+μ) / 0.3` calculé par run
3. **Hubble friction = 1.0** partout (vraie physique)
4. **CSV flushé après CHAQUE run** (pour suivi temps réel)
5. **Auto-stop sur sentinelle NO-GO** (économise GPU)
6. **Auto-stop sur instabilité v_rms > 50000** (passe au run suivant, ne crash pas tout)

## Budget estimé

- Code (réutilisation forte de janus_adaptive_zoom) : 3-4h dev
- Compilation + test : 30 min
- Phase 0 sentinelles : ~3h GPU
- Phase 1 scan complet : ~10h GPU (si GO)

**Total : ~14h GPU si tout va bien, beaucoup moins en cas de NO-GO précoce.**

Marge dans le budget 30h pour Phase 2 zoom après.
