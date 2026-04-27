//! VSL Option B Validation Test — 10 steps only
//!
//! Checks 4 criteria:
//! 1. ρ+_max < 100 at step 0
//! 2. ρ-_max < 100 at step 0
//! 3. Seg < 0.005 at step 0
//! 4. ratio v_rms between 0.99-1.01 at step 10

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::{CoupledFriedmann, JanusVSLParams};

const N_PLUS: usize = 250_000;
const N_MINUS: usize = 250_000;
const BOX_SIZE: f64 = 100.0;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;
const STEPS: usize = 10;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: --features cuda required");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_validation();
}

#[cfg(feature = "cuda")]
fn run_validation() {
    let output_dir = "/app/output/vsl_optionb_validation";
    create_dir_all(output_dir).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  VSL OPTION B VALIDATION TEST — 10 steps                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Criteria:                                                       ║");
    println!("║  1. ρ+_max < 100 at step 0                                       ║");
    println!("║  2. ρ-_max < 100 at step 0                                       ║");
    println!("║  3. Seg < 0.005 at step 0                                        ║");
    println!("║  4. ratio v_rms ∈ [0.99, 1.01] at step 10                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Initialize simulation
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    let c_ratio_init = c_ratio_sq_init.sqrt();
    println!("\nc_ratio(z={}) = {:.6}", Z_INIT, c_ratio_init);

    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(0.5);
    gpu_sim.set_c_ratio(c_ratio_init);

    // Initial state
    let pos = gpu_sim.get_positions().expect("get_positions");
    let vel = gpu_sim.get_velocities().expect("get_velocities");
    let signs = gpu_sim.get_signs().expect("get_signs");

    let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs);
    let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);
    let seg = compute_segregation(&pos, &signs, BOX_SIZE);
    let ratio = v_rms_minus / v_rms_plus;

    println!("\n=== STEP 0 (Initial Conditions) ===");
    println!("  ρ+_max = {:.1}", rho_plus_max);
    println!("  ρ-_max = {:.1}", rho_minus_max);
    println!("  Seg    = {:.6}", seg);
    println!("  v_rms+ = {:.2} km/s", v_rms_plus);
    println!("  v_rms- = {:.2} km/s", v_rms_minus);
    println!("  ratio  = {:.6}", ratio);

    // Check criteria 1-3
    let mut pass = true;

    if rho_plus_max >= 100.0 {
        println!("\n❌ FAIL: ρ+_max = {:.1} >= 100", rho_plus_max);
        pass = false;
    } else {
        println!("\n✓ PASS: ρ+_max = {:.1} < 100", rho_plus_max);
    }

    if rho_minus_max >= 100.0 {
        println!("❌ FAIL: ρ-_max = {:.1} >= 100", rho_minus_max);
        pass = false;
    } else {
        println!("✓ PASS: ρ-_max = {:.1} < 100", rho_minus_max);
    }

    if seg >= 0.005 {
        println!("❌ FAIL: Seg = {:.6} >= 0.005", seg);
        pass = false;
    } else {
        println!("✓ PASS: Seg = {:.6} < 0.005", seg);
    }

    // Run 10 steps
    println!("\n=== Running {} steps ===", STEPS);

    let mut current_z = Z_INIT;
    for step in 1..=STEPS {
        // Update c_ratio
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(current_z, ETA);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        // Step simulation
        gpu_sim.step(DT);

        // Update z (simple approximation: dz/dt = -H(z)*(1+z))
        // For validation test, just decrement z slightly
        let h_z = 70.0 * (0.3 * (1.0 + current_z).powi(3) + 0.7).sqrt();  // km/s/Mpc
        let h_gyr = h_z / 977.8;  // Gyr^-1
        let da = h_gyr * DT / (1.0 + current_z);
        let a = 1.0 / (1.0 + current_z) + da;
        current_z = 1.0 / a - 1.0;

        if step % 2 == 0 {
            print!("  step {} (z={:.3})... ", step, current_z);
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    // Final state
    let pos = gpu_sim.get_positions().expect("get_positions");
    let vel = gpu_sim.get_velocities().expect("get_velocities");
    let signs = gpu_sim.get_signs().expect("get_signs");

    let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs);
    let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);
    let seg = compute_segregation(&pos, &signs, BOX_SIZE);
    let ratio = v_rms_minus / v_rms_plus;

    println!("\n=== STEP {} (Final) ===", STEPS);
    println!("  z      = {:.4}", current_z);
    println!("  ρ+_max = {:.1}", rho_plus_max);
    println!("  ρ-_max = {:.1}", rho_minus_max);
    println!("  Seg    = {:.6}", seg);
    println!("  v_rms+ = {:.2} km/s", v_rms_plus);
    println!("  v_rms- = {:.2} km/s", v_rms_minus);
    println!("  ratio  = {:.6}", ratio);

    // Check criterion 4
    if ratio < 0.99 || ratio > 1.01 {
        println!("\n❌ FAIL: ratio = {:.6} not in [0.99, 1.01]", ratio);
        pass = false;
    } else {
        println!("\n✓ PASS: ratio = {:.6} ∈ [0.99, 1.01]", ratio);
    }

    // Final verdict
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    if pass {
        println!("║  ✅ ALL 4 CRITERIA PASSED — Ready for 5000 steps               ║");
    } else {
        println!("║  ❌ VALIDATION FAILED — Fix before full run                    ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Write result
    let result_path = format!("{}/validation_result.txt", output_dir);
    let mut f = BufWriter::new(File::create(&result_path).unwrap());
    writeln!(f, "VALIDATION_PASSED={}", pass).unwrap();
    writeln!(f, "rho_plus_max_0={:.1}", rho_plus_max).unwrap();
    writeln!(f, "rho_minus_max_0={:.1}", rho_minus_max).unwrap();
    writeln!(f, "seg_0={:.6}", seg).unwrap();
    writeln!(f, "ratio_10={:.6}", ratio).unwrap();

    if pass {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn compute_vrms(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let n = signs.len();
    let mut sum_v2_plus = 0.0;
    let mut sum_v2_minus = 0.0;
    let mut n_plus = 0;
    let mut n_minus = 0;

    for i in 0..n {
        let v2 = vel[i*3].powi(2) + vel[i*3+1].powi(2) + vel[i*3+2].powi(2);
        if signs[i] > 0 {
            sum_v2_plus += v2;
            n_plus += 1;
        } else {
            sum_v2_minus += v2;
            n_minus += 1;
        }
    }

    let v_rms_plus = if n_plus > 0 { (sum_v2_plus / n_plus as f64).sqrt() * 977.8 } else { 0.0 };
    let v_rms_minus = if n_minus > 0 { (sum_v2_minus / n_minus as f64).sqrt() * 977.8 } else { 0.0 };

    (v_rms_plus, v_rms_minus)
}

#[cfg(feature = "cuda")]
fn compute_max_densities(pos: &[f64], signs: &[i32], box_size: f64, n_grid: usize) -> (f64, f64) {
    let cell_size = box_size / n_grid as f64;
    let n = signs.len();

    let mut grid_plus = vec![0u32; n_grid * n_grid * n_grid];
    let mut grid_minus = vec![0u32; n_grid * n_grid * n_grid];

    for i in 0..n {
        let ix = ((pos[i*3] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let iy = ((pos[i*3+1] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let iz = ((pos[i*3+2] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let idx = ix * n_grid * n_grid + iy * n_grid + iz;

        if signs[i] > 0 {
            grid_plus[idx] += 1;
        } else {
            grid_minus[idx] += 1;
        }
    }

    let max_plus = *grid_plus.iter().max().unwrap_or(&0) as f64;
    let max_minus = *grid_minus.iter().max().unwrap_or(&0) as f64;

    (max_plus, max_minus)
}

#[cfg(feature = "cuda")]
fn compute_segregation(pos: &[f64], signs: &[i32], box_size: f64) -> f64 {
    let n = signs.len();
    let mut sum_plus = [0.0f64; 3];
    let mut sum_minus = [0.0f64; 3];
    let mut n_plus = 0;
    let mut n_minus = 0;
    let mut ref_plus = [0.0f64; 3];
    let mut ref_minus = [0.0f64; 3];

    // Find first particle of each type as reference
    for i in 0..n {
        if signs[i] > 0 && n_plus == 0 {
            ref_plus = [pos[i*3], pos[i*3+1], pos[i*3+2]];
        }
        if signs[i] < 0 && n_minus == 0 {
            ref_minus = [pos[i*3], pos[i*3+1], pos[i*3+2]];
        }

        if signs[i] > 0 {
            for k in 0..3 {
                let mut d = pos[i*3+k] - ref_plus[k];
                if d > box_size / 2.0 { d -= box_size; }
                if d < -box_size / 2.0 { d += box_size; }
                sum_plus[k] += d;
            }
            n_plus += 1;
        } else {
            for k in 0..3 {
                let mut d = pos[i*3+k] - ref_minus[k];
                if d > box_size / 2.0 { d -= box_size; }
                if d < -box_size / 2.0 { d += box_size; }
                sum_minus[k] += d;
            }
            n_minus += 1;
        }
    }

    let com_plus = [
        ref_plus[0] + sum_plus[0] / n_plus as f64,
        ref_plus[1] + sum_plus[1] / n_plus as f64,
        ref_plus[2] + sum_plus[2] / n_plus as f64,
    ];
    let com_minus = [
        ref_minus[0] + sum_minus[0] / n_minus as f64,
        ref_minus[1] + sum_minus[1] / n_minus as f64,
        ref_minus[2] + sum_minus[2] / n_minus as f64,
    ];

    let mut d2 = 0.0;
    for k in 0..3 {
        let mut d = com_plus[k] - com_minus[k];
        if d > box_size / 2.0 { d -= box_size; }
        if d < -box_size / 2.0 { d += box_size; }
        d2 += d * d;
    }

    d2.sqrt() / box_size
}
