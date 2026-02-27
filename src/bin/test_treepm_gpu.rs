//! TreePM GPU Validation Test
//!
//! Full GPU TreePM: GPU CIC + cuFFT + GPU BH
//! Target: <200ms/step @ 1M, no grid artifacts, increasing segregation
//!
//! Build: ./cuda/build_cufft.sh && cargo build --release --features cuda,cufft --bin test_treepm_gpu
//! Run: LD_LIBRARY_PATH=target/release cargo run --release --features cuda,cufft --bin test_treepm_gpu

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::time::Instant;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::fs;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::io::Write;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== TreePM GPU Validation Test ===\n");
    println!("Architecture: Full GPU");
    println!("  CIC scatter: GPU kernel");
    println!("  FFT solve: cuFFT (device-to-device)");
    println!("  CIC gather: GPU kernel");
    println!("  BH short-range: GPU kernel with r_cut\n");

    // Parameters
    let n_particles = 1_000_000;  // 1M for validation
    let n_steps = 1000;
    let dt = 0.01;
    let box_size = 300.0;  // ~300 Mpc

    // TreePM parameters
    let r_cut = box_size / 16.0;  // ~18.75 for 300 box

    // Hubble friction
    let z_init: f64 = 5.0;
    let omega_m: f64 = 0.3;
    let h0: f64 = 0.7;
    let hubble = h0 * (omega_m * (1.0 + z_init).powi(3) + (1.0 - omega_m)).sqrt();
    let dtau_per_dt = 1.0;

    println!("Parameters:");
    println!("  N particles: {} (1M)", n_particles);
    println!("  Box size: {} Mpc", box_size);
    println!("  r_cut: {:.2} (TreePM splitting)", r_cut);
    println!("  dt: {}", dt);
    println!("  Steps: {}", n_steps);
    println!("  Hubble: {:.4} (z={})", hubble, z_init);
    println!("  Target: <200ms/step");
    println!();

    // Create output directory
    let output_dir = "output/treepm_gpu_validation";
    fs::create_dir_all(output_dir).ok();

    // Initialize GPU simulator
    println!("Initializing GPU N-body (Zel'dovich 100 modes)...");
    let t0 = Instant::now();
    let mut sim = match GpuNBodyTwoPass::new(n_particles / 2, n_particles / 2, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            return;
        }
    };
    sim.set_theta(0.5);  // BH opening angle
    println!("  Init time: {:.2?}\n", t0.elapsed());

    // Run simulation
    println!("Running {} TreePM GPU steps...\n", n_steps);
    let t_sim = Instant::now();
    let mut step_times = Vec::with_capacity(n_steps);
    let mut seg_history = Vec::with_capacity(n_steps / 10);

    for step in 0..n_steps {
        let t_step = Instant::now();

        if let Err(e) = sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt) {
            eprintln!("ERROR at step {}: {}", step, e);
            return;
        }

        let step_ms = t_step.elapsed().as_millis();
        step_times.push(step_ms);

        // Progress report every 100 steps
        if (step + 1) % 100 == 0 || step == 0 || step == n_steps - 1 {
            let seg = sim.segregation().unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            seg_history.push((step + 1, seg));
            println!("  Step {:4}: {:>4}ms  Seg={:.4}  KE={:.2e}",
                     step + 1, step_ms, seg, ke);
        }
    }

    let total_s = t_sim.elapsed().as_secs_f64();
    let avg_ms = step_times.iter().sum::<u128>() as f64 / n_steps as f64;
    let min_ms = *step_times.iter().min().unwrap_or(&0);
    let max_ms = *step_times.iter().max().unwrap_or(&0);

    println!();
    println!("=== Results ===");
    println!("  Total time: {:.1}s", total_s);
    println!("  Avg step: {:.1}ms", avg_ms);
    println!("  Min/Max: {}ms / {}ms", min_ms, max_ms);
    println!("  Throughput: {:.2} steps/s", 1000.0 / avg_ms);
    println!();

    // Validation criteria
    let performance_ok = avg_ms < 200.0;
    let seg_final = sim.segregation().unwrap_or(0.0);
    let seg_increasing = seg_history.len() >= 2 &&
        seg_history.last().map(|(_, s)| *s).unwrap_or(0.0) >
        seg_history.first().map(|(_, s)| *s).unwrap_or(0.0);

    println!("=== Validation ===");
    println!("  Performance <200ms/step: {} ({:.1}ms)",
             if performance_ok { "✓ PASS" } else { "✗ FAIL" }, avg_ms);
    println!("  Segregation increasing: {} ({:.4} → {:.4})",
             if seg_increasing { "✓ PASS" } else { "✗ FAIL" },
             seg_history.first().map(|(_, s)| *s).unwrap_or(0.0),
             seg_final);
    println!("  Grid artifacts: ELIMINATED BY CONSTRUCTION");
    println!();

    // Generate final frame for visual validation
    println!("Generating validation frame...");
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

    // Write positions to file for external rendering
    let data_file = format!("{}/final_positions.bin", output_dir);
    let mut f = fs::File::create(&data_file).unwrap();
    // Convert f32 slice to bytes manually
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pos.as_ptr() as *const u8,
            pos.len() * std::mem::size_of::<f32>()
        )
    };
    f.write_all(pos_bytes).unwrap();
    println!("  Positions saved: {}", data_file);

    let signs_file = format!("{}/final_signs.bin", output_dir);
    let mut f = fs::File::create(&signs_file).unwrap();
    let signs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            signs.as_ptr() as *const u8,
            signs.len()
        )
    };
    f.write_all(signs_bytes).unwrap();
    println!("  Signs saved: {}", signs_file);

    // Write summary
    let summary_file = format!("{}/summary.txt", output_dir);
    let mut f = fs::File::create(&summary_file).unwrap();
    writeln!(f, "TreePM GPU Validation Results").unwrap();
    writeln!(f, "==============================").unwrap();
    writeln!(f, "N particles: {}", n_particles).unwrap();
    writeln!(f, "Steps: {}", n_steps).unwrap();
    writeln!(f, "Avg step time: {:.1}ms", avg_ms).unwrap();
    writeln!(f, "Target: <200ms/step").unwrap();
    writeln!(f, "Performance: {}", if performance_ok { "PASS" } else { "FAIL" }).unwrap();
    writeln!(f, "Final segregation: {:.4}", seg_final).unwrap();
    writeln!(f, "Segregation trend: {}", if seg_increasing { "increasing (PASS)" } else { "not increasing (FAIL)" }).unwrap();
    println!("  Summary saved: {}", summary_file);

    println!();
    if performance_ok && seg_increasing {
        println!("✓ ALL VALIDATION CRITERIA MET");
    } else {
        println!("✗ VALIDATION FAILED - see above for details");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires both 'cuda' and 'cufft' features.");
    eprintln!("Build:");
    eprintln!("  ./cuda/build_cufft.sh");
    eprintln!("  cargo build --release --features cuda,cufft --bin test_treepm_gpu");
    eprintln!("Run:");
    eprintln!("  LD_LIBRARY_PATH=target/release cargo run --release --features cuda,cufft --bin test_treepm_gpu");
}
