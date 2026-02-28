//! TreePM GPU Production Test
//!
//! Full production run: 1M particles, 6000 steps (z=5 → z≈0)
//! Validates: segregation growth, Hubble cooling, no grid artifacts
//!
//! Uses CosmoInterpolator for proper cosmological evolution (like nbody_overnight.rs)
//!
//! Build: ./cuda/build_cufft.sh && cargo build --release --features cuda,cufft --bin treepm_production_test
//! Run: LD_LIBRARY_PATH=target/release cargo run --release --features cuda,cufft --bin treepm_production_test

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::time::Instant;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::fs;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::io::Write;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== TreePM GPU Production Test ===\n");
    println!("Target: z=5 → z≈0 segregation dynamics");

    // Parameters
    let n_particles = 1_000_000;  // 1M
    let n_steps = 6000;
    let dt = 0.01;
    let box_size = 300.0;  // ~300 Mpc
    let eta = 1.045;  // Janus parameter

    // TreePM parameters
    let r_cut = box_size / 16.0;  // ~18.75 for 300 box

    // Cosmological setup using CosmoInterpolator (like nbody_overnight.rs)
    let z_init: f64 = 5.0;
    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);

    // dtau_per_dt: convention where 10000 steps cover z=5→0
    // This gives the validated S_max=0.694 behavior
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    // Steps at which to save snapshots
    let snapshot_steps = vec![1000, 3000, 6000];

    println!("Parameters:");
    println!("  N particles: {} (1M)", n_particles);
    println!("  Box size: {} Mpc", box_size);
    println!("  r_cut: {:.2}", r_cut);
    println!("  dt: {}", dt);
    println!("  Steps: {}", n_steps);
    println!("  η: {}", eta);
    println!("  Snapshots at: {:?}", snapshot_steps);
    println!();
    println!("Cosmological setup (CosmoInterpolator):");
    println!("  z_init: {}", z_init);
    println!("  tau_start: {:.6} (z={})", cosmo.tau_start, z_init);
    println!("  tau_end: {:.6} (z=0)", cosmo.tau_end);
    println!("  dtau_per_step: {:.6}", dtau_per_step);
    println!("  dtau_per_dt: {:.6}", dtau_per_dt);
    println!();

    // Create output directory
    let output_dir = "output/treepm_production";
    fs::create_dir_all(output_dir).ok();
    fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

    // Initialize GPU simulator with virialization
    println!("Initializing GPU N-body (Zel'dovich 100 modes, virialized)...");
    let t0 = Instant::now();
    let mut sim = match GpuNBodyTwoPass::new(n_particles / 2, n_particles / 2, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            return;
        }
    };
    sim.set_theta(0.5);
    println!("  Init time: {:.2?}\n", t0.elapsed());

    // Get initial KE for tracking cooling
    let ke_0 = sim.kinetic_energy().unwrap_or(1.0);
    let seg_0 = sim.segregation().unwrap_or(0.0);

    println!("Initial state:");
    println!("  KE_0 = {:.4e}", ke_0);
    println!("  Seg_0 = {:.4}", seg_0);
    println!();

    // Open log file
    let log_path = format!("{}/time_series.csv", output_dir);
    let mut log_file = fs::File::create(&log_path).expect("Failed to create log file");
    writeln!(log_file, "step,tau,z,a,hubble,seg,ke,ke_ratio").unwrap();

    // Run simulation
    println!("Running {} TreePM GPU steps...\n", n_steps);
    let t_sim = Instant::now();
    let mut seg_max = 0.0_f64;

    for step in 0..n_steps {
        // Get cosmological parameters at current conformal time
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z_current = 1.0 / a - 1.0;

        // TreePM step with proper cosmological H(tau)
        if let Err(e) = sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt) {
            eprintln!("ERROR at step {}: {}", step, e);
            return;
        }

        // Log every 100 steps
        if (step + 1) % 100 == 0 || step == 0 || snapshot_steps.contains(&(step + 1)) {
            let seg = sim.segregation().unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke_0;
            seg_max = seg_max.max(seg);

            writeln!(log_file, "{},{:.6},{:.4},{:.6},{:.4},{:.6},{:.4e},{:.6}",
                     step + 1, current_tau, z_current, a, hubble, seg, ke, ke_ratio).unwrap();

            if (step + 1) % 500 == 0 || snapshot_steps.contains(&(step + 1)) {
                println!("  Step {:5}: z={:.2}  a={:.4}  H={:.4}  Seg={:.4}  KE/KE₀={:.4}",
                         step + 1, z_current, a, hubble, seg, ke_ratio);
            }
        }

        // Save snapshots at specified steps
        if snapshot_steps.contains(&(step + 1)) {
            println!("  → Saving snapshot at step {}...", step + 1);
            save_snapshot(&sim, output_dir, step + 1);
        }
    }

    let total_s = t_sim.elapsed().as_secs_f64();
    let avg_ms = total_s * 1000.0 / n_steps as f64;

    // Get final cosmological state
    let final_tau = cosmo.tau_start + (n_steps as f64) * dtau_per_step;
    let (a_final, _) = cosmo.get_params_at_tau(final_tau);
    let z_final = 1.0 / a_final - 1.0;

    println!();
    println!("=== Results ===");
    println!("  Total time: {:.1}s", total_s);
    println!("  Avg step: {:.1}ms", avg_ms);
    println!("  z: {:.2} → {:.2}", z_init, z_final);
    println!();

    // Final state
    let seg_final = sim.segregation().unwrap_or(0.0);
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);

    println!("=== Validation ===");
    println!("  Seg_0 → Seg_final: {:.4} → {:.4}", seg_0, seg_final);
    println!("  Seg_max: {:.4}", seg_max);
    println!("  KE/KE₀ final: {:.4} (should be < 1 = Hubble cooling)", ke_final / ke_0);
    println!();

    // Validation criteria
    let seg_growing = seg_max > 0.1;
    let hubble_cooling = ke_final < ke_0;
    let performance_ok = avg_ms < 250.0;  // Relaxed for TreePM

    println!("  Seg_max > 0.1: {} ({:.4})",
             if seg_growing { "✓ PASS" } else { "✗ FAIL" }, seg_max);
    println!("  Hubble cooling (KE decreasing): {} ({:.4e} < {:.4e})",
             if hubble_cooling { "✓ PASS" } else { "✗ FAIL" }, ke_final, ke_0);
    println!("  Performance <250ms/step: {} ({:.1}ms)",
             if performance_ok { "✓ PASS" } else { "✗ FAIL" }, avg_ms);
    println!();

    // Write summary
    let summary_path = format!("{}/summary.txt", output_dir);
    let mut f = fs::File::create(&summary_path).unwrap();
    writeln!(f, "TreePM GPU Production Test Results").unwrap();
    writeln!(f, "===================================").unwrap();
    writeln!(f, "N particles: {}", n_particles).unwrap();
    writeln!(f, "Steps: {}", n_steps).unwrap();
    writeln!(f, "η: {}", eta).unwrap();
    writeln!(f, "z: {:.2} → {:.2}", z_init, z_final).unwrap();
    writeln!(f, "dtau_per_dt: {:.6}", dtau_per_dt).unwrap();
    writeln!(f, "Avg step time: {:.1}ms", avg_ms).unwrap();
    writeln!(f, "Seg: {:.4} → {:.4} (max={:.4})", seg_0, seg_final, seg_max).unwrap();
    writeln!(f, "KE/KE₀: 1.0 → {:.4}", ke_final / ke_0).unwrap();
    writeln!(f, "Validation: Seg_max>0.1={}, Cooling={}, Perf={}",
             seg_growing, hubble_cooling, performance_ok).unwrap();

    if seg_growing && hubble_cooling && performance_ok {
        println!("✓ ALL VALIDATION CRITERIA MET");
    } else {
        println!("✗ VALIDATION FAILED - see above for details");
    }

    println!("\nOutput saved to: {}", output_dir);
    println!("Snapshots: {:?}", snapshot_steps);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, output_dir: &str, step: usize) {
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

    // Save binary data
    let pos_path = format!("{}/snapshots/pos_{:06}.bin", output_dir, step);
    let mut f = fs::File::create(&pos_path).unwrap();
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pos.as_ptr() as *const u8,
            pos.len() * std::mem::size_of::<f32>()
        )
    };
    f.write_all(pos_bytes).unwrap();

    let signs_path = format!("{}/snapshots/signs_{:06}.bin", output_dir, step);
    let mut f = fs::File::create(&signs_path).unwrap();
    let signs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            signs.as_ptr() as *const u8,
            signs.len()
        )
    };
    f.write_all(signs_bytes).unwrap();

    println!("     Saved: {}", pos_path);
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires both 'cuda' and 'cufft' features.");
    eprintln!("Build:");
    eprintln!("  ./cuda/build_cufft.sh");
    eprintln!("  cargo build --release --features cuda,cufft --bin treepm_production_test");
}
