//! TreePM 2M Production Run
//!
//! Parameters:
//! - N = 2M, theta = 0.7, r_cut = box/16
//! - 12000 steps (z=5 → z=0)
//! - Frames every 500 steps
//! - Objective: S_max > 0.5 @ z ≈ 1.8

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::io::Write;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== TreePM 2M Production Run ===\n");

    let n = 2_000_000;
    let n_steps = 12000;
    let dt = 0.01;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);  // ~272 Mpc
    let eta = 1.045;
    let z_init = 5.0;
    let theta = 0.7;
    let r_cut = box_size / 16.0;
    let frame_interval = 500;

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  box = {:.1} Mpc", box_size);
    println!("  theta = {}", theta);
    println!("  r_cut = {:.2} Mpc (box/16)", r_cut);
    println!("  steps = {}", n_steps);
    println!("  frame_interval = {}", frame_interval);
    println!();

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    // Create output directory
    let output_dir = "/app/output/treepm_2m_production";
    std::fs::create_dir_all(output_dir).ok();
    std::fs::create_dir_all(format!("{}/frames", output_dir)).ok();

    // Open CSV for time series
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = std::fs::File::create(&csv_path).expect("csv create failed");
    writeln!(csv, "step,z,ke_ratio,segregation,step_ms").unwrap();

    println!("Initializing {} particles...", n);
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(theta);
    println!("  Init time: {:.1}s\n", t0.elapsed().as_secs_f64());

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}\n", ke_0, seg_0);

    let mut seg_max = seg_0;
    let mut seg_max_step = 0;
    let mut seg_max_z = z_init;

    println!("{:>6} {:>8} {:>10} {:>10} {:>10} {:>10}",
             "Step", "z", "KE/KE₀", "Seg", "S_max", "ms/step");
    println!("{}", "-".repeat(70));

    let run_start = std::time::Instant::now();

    for step in 1..=n_steps {
        let step_start = std::time::Instant::now();

        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;
        let dtau_per_dt = dtau_per_step / dt;

        sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt).expect("TreePM step failed");

        let step_ms = step_start.elapsed().as_millis();

        // Save frame every frame_interval steps
        if step % frame_interval == 0 {
            let frame_path = format!("{}/frames/frame_{:05}.bin", output_dir, step);
            save_frame(&sim, &frame_path, box_size as f32);
        }

        // Compute diagnostics every 100 steps or at key points
        let should_print = step % 500 == 0 || step == n_steps;

        if should_print || step % 100 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();

            if seg > seg_max {
                seg_max = seg;
                seg_max_step = step;
                seg_max_z = z;
            }

            // Write to CSV
            writeln!(csv, "{},{:.4},{:.6},{:.6},{}",
                     step, z, ke / ke_0, seg, step_ms).unwrap();

            if should_print {
                println!("{:>6} {:>8.3} {:>10.4} {:>10.4} {:>10.4} {:>10}",
                         step, z, ke / ke_0, seg, seg_max, step_ms);
            }
        }
    }

    let total_time = run_start.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("  S_max = {:.4} at step {} (z = {:.2})", seg_max, seg_max_step, seg_max_z);
    println!("  Total runtime: {:.1} hours", total_time / 3600.0);
    println!("  Avg step time: {:.0} ms", total_time * 1000.0 / n_steps as f64);
    println!();
    println!("Output: {}", output_dir);
    println!("  frames/frame_*.bin ({} frames)", n_steps / frame_interval);
    println!("  time_series.csv");
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_frame(sim: &GpuNBodyTwoPass, path: &str, box_size: f32) {
    let (pos, _vel, signs) = sim.get_particles().expect("get_particles failed");
    let n = signs.len();

    let mut file = std::fs::File::create(path).expect("create file failed");

    // Write header
    file.write_all(&(n as u32).to_le_bytes()).expect("write n failed");
    file.write_all(&box_size.to_le_bytes()).expect("write box failed");

    // Write positions (f32 x 3)
    for i in 0..n {
        file.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
    }

    // Write signs (i8)
    for i in 0..n {
        file.write_all(&[signs[i] as u8]).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
