//! Test NO-DIPOLE 512k — GPU TreePM with k-space filter
//!
//! Parameters:
//!   - N = 512k (80³ grid)
//!   - L = 492 Mpc
//!   - ε = 0.4 Mpc (softening)
//!   - k_min = 3 (suppresses k=0,1,2 in PM solver)
//!   - NO COM recentering (pure Janus dynamics)
//!
//! Goal: See if suppressing dipole via k-space filter allows filaments to form
//!
//! Build:
//!   ./cuda/build_cufft.sh
//!   cargo build --release --features cuda,cufft --bin test_nodipole_gpu_512k
//!
//! Run:
//!   LD_LIBRARY_PATH=target/release docker compose run --rm dev \
//!     cargo run --release --features cuda,cufft --bin test_nodipole_gpu_512k

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::fs::{self, File};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::io::{Write, BufWriter};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const SOFTENING: f64 = 0.4;            // Mpc

// k-space filter in PM solver
const K_MIN: usize = 3;                // Filter k=0,1,2 (suppresses dipole)

// Simulation parameters
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 2000;
const SNAPSHOT_INTERVAL: usize = 20;
const LOG_INTERVAL: usize = 10;
const THETA: f64 = 0.7;

// TreePM split
const R_CUT_FACTOR: f64 = 16.0;        // r_cut = L_BOX / R_CUT_FACTOR

// Janus parameter
const ETA: f64 = 1.045;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TEST NO-DIPOLE 512k — GPU TreePM with k-space filter");
    println!("═══════════════════════════════════════════════════════════════");

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_plus = (n3 as f64 / (1.0 + ETA)) as usize;
    let n_minus = n3 - n_plus;
    let r_cut = L_BOX / R_CUT_FACTOR;

    println!("  Particles: {}³ = {} (N+ = {}, N- = {})", N_GRID, n3, n_plus, n_minus);
    println!("  Box: {} Mpc", L_BOX);
    println!("  Softening: {} Mpc", SOFTENING);
    println!("  θ (BH): {}", THETA);
    println!("  r_cut (TreePM): {:.2} Mpc", r_cut);
    println!("  k_min (PM filter): {} (suppresses dipole)", K_MIN);
    println!("  Steps: {}", TOTAL_STEPS);
    println!("  dt: {}", DT);
    println!();
    println!("  *** NO COM RECENTERING — Pure Janus dynamics ***");
    println!();

    // Output directory
    let output_dir = format!("/app/output/nodipole_gpu_512k_{}",
        chrono::Local::now().format("%Y-%m-%d_%H%M%S"));
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("  Output: {}", output_dir);
    println!();

    // Initialize GPU simulation with Zel'dovich ICs (100 modes)
    println!("  Initializing GPU TreePM simulation...");
    let t0 = Instant::now();

    let mut sim = match GpuNBodyTwoPass::new(n_plus, n_minus, L_BOX) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            return;
        }
    };

    // Configure simulation
    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_pm_k_min(K_MIN);  // Enable k-space filter

    println!("  Init time: {:.2?}", t0.elapsed());
    println!();

    // Open CSV log
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,ke,segregation,time_ms").unwrap();

    // Get initial state
    let ke_0 = sim.kinetic_energy().unwrap_or(1.0);
    let seg_0 = sim.segregation().unwrap_or(0.0);
    println!("  Initial KE: {:.4e}", ke_0);
    println!("  Initial Seg: {:.2} Mpc", seg_0);
    println!();

    // Save initial snapshot
    save_snapshot(&sim, 0, &output_dir);

    // Main simulation loop
    println!("  Starting simulation ({} steps)...\n", TOTAL_STEPS);
    let start = Instant::now();
    let mut seg_max = seg_0;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        // TreePM GPU step (no Hubble friction for this test)
        if let Err(e) = sim.step_treepm_gpu(DT, r_cut, 0.0, 0.0) {
            eprintln!("ERROR at step {}: {}", step, e);
            return;
        }

        let step_time = step_start.elapsed().as_millis();

        // Diagnostics
        if step % LOG_INTERVAL == 0 || step == 1 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let seg = sim.segregation().unwrap_or(0.0);
            seg_max = seg_max.max(seg);

            writeln!(csv, "{},{:.6e},{:.6},{}", step, ke, seg, step_time).unwrap();

            if step % 100 == 0 || step == 1 {
                println!("  Step {:5} | KE={:.3e} | Seg={:.2} Mpc | {:.0}ms/step",
                    step, ke, seg, step_time);
            }
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &output_dir);
        }
    }

    csv.flush().unwrap();
    let total_time = start.elapsed().as_secs_f64();
    let avg_ms = total_time * 1000.0 / TOTAL_STEPS as f64;

    // Final state
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let seg_final = sim.segregation().unwrap_or(0.0);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total time: {:.1} s ({:.1} ms/step)", total_time, avg_ms);
    println!();
    println!("  KE: {:.4e} → {:.4e} (ratio = {:.4})", ke_0, ke_final, ke_final / ke_0);
    println!("  Seg: {:.2} → {:.2} Mpc (max = {:.2})", seg_0, seg_final, seg_max);
    println!();

    // Check if dipole was suppressed (seg should stay small)
    let dipole_suppressed = seg_final < 100.0;  // Should not reach box/2 = 246 Mpc
    println!("  Dipole suppressed (Seg < 100 Mpc): {} (Seg = {:.2})",
        if dipole_suppressed { "YES" } else { "NO" }, seg_final);
    println!();
    println!("  Output: {}", output_dir);
    println!();

    // Write summary
    let summary_path = format!("{}/summary.txt", output_dir);
    let mut f = File::create(&summary_path).unwrap();
    writeln!(f, "Test NO-DIPOLE 512k — GPU TreePM with k-space filter").unwrap();
    writeln!(f, "=================================================").unwrap();
    writeln!(f, "N particles: {} (80³)", n3).unwrap();
    writeln!(f, "N+: {}, N-: {}", n_plus, n_minus).unwrap();
    writeln!(f, "Box: {} Mpc", L_BOX).unwrap();
    writeln!(f, "Softening: {} Mpc", SOFTENING).unwrap();
    writeln!(f, "r_cut: {:.2} Mpc", r_cut).unwrap();
    writeln!(f, "k_min: {} (filter k=0,1,2)", K_MIN).unwrap();
    writeln!(f, "Steps: {}", TOTAL_STEPS).unwrap();
    writeln!(f, "dt: {}", DT).unwrap();
    writeln!(f, "theta: {}", THETA).unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "Results:").unwrap();
    writeln!(f, "  Total time: {:.1} s ({:.1} ms/step)", total_time, avg_ms).unwrap();
    writeln!(f, "  KE: {:.4e} → {:.4e}", ke_0, ke_final).unwrap();
    writeln!(f, "  Seg: {:.2} → {:.2} (max: {:.2})", seg_0, seg_final, seg_max).unwrap();
    writeln!(f, "  Dipole suppressed: {}", dipole_suppressed).unwrap();
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, step: usize, output_dir: &str) {
    let pos = match sim.get_positions() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR getting positions: {}", e);
            return;
        }
    };
    let signs = match sim.get_signs() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR getting signs: {}", e);
            return;
        }
    };

    let n = signs.len();
    let path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
    let mut f = BufWriter::new(File::create(&path).unwrap());

    // Header: n_particles, step, reserved
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&(step as u64).to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();

    // Data: x, y, z, sign (all f32)
    for i in 0..n {
        f.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        f.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        f.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        f.write_all(&(signs[i] as f32).to_le_bytes()).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Error: This binary requires both 'cuda' and 'cufft' features.");
    eprintln!("");
    eprintln!("Build:");
    eprintln!("  ./cuda/build_cufft.sh");
    eprintln!("  cargo build --release --features cuda,cufft --bin test_nodipole_gpu_512k");
    eprintln!("");
    eprintln!("Run:");
    eprintln!("  LD_LIBRARY_PATH=target/release cargo run --release --features cuda,cufft --bin test_nodipole_gpu_512k");
    std::process::exit(1);
}
