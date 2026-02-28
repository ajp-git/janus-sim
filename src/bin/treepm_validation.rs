//! TreePM Validation Test
//!
//! Criteria:
//! - No grid artifacts (visual check at step 1000, 3000)
//! - S_max > 0.4 (comparable to BH pure theta=0.7)
//! - z @ S_max ≈ 1.8
//!
//! Parameters:
//! - N = 100K, theta = 0.7, r_cut = box/16
//! - 5000 steps with Hubble friction
//! - Uniform random ICs (Zel'dovich suppresses segregation)

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::io::Write;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== TreePM Validation Test ===\n");

    let n = 100_000;
    let n_steps = 5000;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;
    let theta = 0.7;  // Required for correct physics
    let r_cut = box_size / 16.0;  // TreePM split: BH for r<r_cut, PM for r>r_cut

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  theta = {} (validated)", theta);
    println!("  r_cut = {:.2} Mpc (box/16)", r_cut);
    println!("  steps = {}", n_steps);
    println!();

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    // Create output directory
    let output_dir = "/app/output/treepm_validation";
    std::fs::create_dir_all(output_dir).ok();

    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(theta);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}\n", ke_0, seg_0);

    let mut seg_max = seg_0;
    let mut seg_max_step = 0;
    let mut seg_max_z = z_init;

    println!("{:>6} {:>8} {:>10} {:>10} {:>10}",
             "Step", "z", "KE/KE₀", "Seg", "Trend");
    println!("{}", "-".repeat(55));

    let mut last_seg = seg_0;

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;
        let dtau_per_dt = dtau_per_step / dt;

        // Use TreePM step
        sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt).expect("TreePM step failed");

        // Output frames at key steps for visual inspection
        if step == 1000 || step == 3000 {
            let frame_path = format!("{}/frame_{:05}.bin", output_dir, step);
            save_frame(&sim, &frame_path);
            println!("  → Saved frame: {}", frame_path);
        }

        // Sample key steps
        let should_print = step % 500 == 0
            || (z < 2.0 && z > 1.5 && step % 100 == 0)
            || step <= 100 && step % 20 == 0;

        if should_print || step == n_steps {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();

            let trend = if seg > last_seg + 0.001 {
                "↑"
            } else if seg < last_seg - 0.001 {
                "↓"
            } else {
                "→"
            };

            if seg > seg_max {
                seg_max = seg;
                seg_max_step = step;
                seg_max_z = z;
            }

            println!("{:>6} {:>8.3} {:>10.4} {:>10.4} {:>10}",
                     step, z, ke / ke_0, seg, trend);

            last_seg = seg;
        }
    }

    println!("\n=== Results ===");
    println!("  S_max = {:.4} at step {} (z = {:.2})", seg_max, seg_max_step, seg_max_z);
    println!();

    // Validation criteria
    let pass_seg = seg_max > 0.4;
    let pass_z = (seg_max_z - 1.8).abs() < 0.5;

    println!("Validation:");
    println!("  [{}] S_max > 0.4: {:.4}", if pass_seg { "✓" } else { "✗" }, seg_max);
    println!("  [{}] z @ S_max ≈ 1.8: {:.2}", if pass_z { "✓" } else { "✗" }, seg_max_z);
    println!("  [ ] No grid artifacts: check frames visually");
    println!();
    println!("Frames saved:");
    println!("  {}/frame_01000.bin (step 1000)", output_dir);
    println!("  {}/frame_03000.bin (step 3000)", output_dir);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_frame(sim: &GpuNBodyTwoPass, path: &str) {
    let (pos, _vel, signs) = sim.get_particles().expect("get_particles failed");
    let n = signs.len();

    let mut file = std::fs::File::create(path).expect("create file failed");

    // Write header
    file.write_all(&(n as u32).to_le_bytes()).expect("write n failed");
    file.write_all(&(100.0f32).to_le_bytes()).expect("write box failed");

    // Write positions (f32 x 3)
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        file.write_all(&x.to_le_bytes()).unwrap();
        file.write_all(&y.to_le_bytes()).unwrap();
        file.write_all(&z.to_le_bytes()).unwrap();
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
