/// Comparison test: GPU tree build vs KDK original
/// Both methods start from identical initial conditions
/// Validates S(t) accuracy for physics validation

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(500_000);  // Default 500K for faster test

        let n_steps: usize = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let dt = 0.01;

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   S(t) Comparison: GPU Tree vs KDK Original                    ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Particles: {} ({:.1}K)", n_particles, n_particles as f64 / 1e3);
        println!("Box size: {:.1}", box_size);
        println!("Steps: {}", n_steps);
        println!();

        // Cosmological params (constant for test)
        let a = 1.0;
        let h = 0.0;
        let dtau_per_dt = 0.0;

        // ═══════════════════════════════════════════════════════════════════
        // Part 1: Morton+DKD (Reference - same integrator as GPU tree)
        // ═══════════════════════════════════════════════════════════════════
        println!("═══ Part 1: Morton+DKD (Reference) ═══");
        println!("Creating simulation for Morton+DKD...");

        let mut kdk_sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");

        // Get initial state BEFORE virialization for copying
        let positions_init = kdk_sim.positions().to_vec();
        let velocities_init = kdk_sim.velocities().to_vec();
        let signs_init = kdk_sim.signs().to_vec();

        kdk_sim.virialize().expect("Virialization failed");

        // Get state AFTER virialization
        let positions_virial = kdk_sim.positions().to_vec();
        let velocities_virial = kdk_sim.velocities().to_vec();

        let seg_0_kdk = kdk_sim.segregation_distance().expect("Failed to compute segregation");
        println!("Initial segregation S(0) = {:.6}", seg_0_kdk);

        let mut kdk_seg_values = Vec::with_capacity(n_steps);
        let kdk_start = Instant::now();

        for step in 1..=n_steps {
            // Use step_with_expansion_dkd_morton (Morton+DKD - same integrator)
            kdk_sim.step_with_expansion_dkd_morton(dt, a, h, dtau_per_dt)
                .expect("Morton+DKD step failed");

            let seg = kdk_sim.segregation_distance().expect("Failed to compute segregation");
            kdk_seg_values.push(seg);

            if step % 20 == 0 || step == n_steps {
                let seg_change = (seg - seg_0_kdk) / seg_0_kdk * 100.0;
                println!("  Step {:3}: S = {:.6} ({:+.2}%)", step, seg, seg_change);
            }
        }

        let kdk_total_time = kdk_start.elapsed().as_secs_f64();
        let kdk_final_seg = *kdk_seg_values.last().unwrap();
        println!("KDK total time: {:.1}s ({:.0} ms/step)",
                 kdk_total_time, kdk_total_time * 1000.0 / n_steps as f64);

        // ═══════════════════════════════════════════════════════════════════
        // Part 2: GPU Tree Build
        // ═══════════════════════════════════════════════════════════════════
        println!();
        println!("═══ Part 2: GPU Tree Build (Karras 2012) ═══");
        println!("Creating simulation with IDENTICAL initial conditions...");

        let mut gpu_sim = GpuNBodySimulation::new_with_state(
            n_positive, n_negative, box_size,
            positions_virial.clone(),
            velocities_virial.clone(),
            signs_init.clone(),
        ).expect("Failed to create GPU simulation with state");

        let seg_0_gpu = gpu_sim.segregation_distance().expect("Failed to compute segregation");
        println!("Initial segregation S(0) = {:.6}", seg_0_gpu);

        // Verify same initial state
        let diff_seg_0 = (seg_0_gpu - seg_0_kdk).abs() / seg_0_kdk * 100.0;
        if diff_seg_0 > 0.01 {
            println!("WARNING: Initial S(0) differs by {:.4}%!", diff_seg_0);
        } else {
            println!("✓ Initial S(0) matches within 0.01%");
        }

        let mut gpu_seg_values = Vec::with_capacity(n_steps);
        let gpu_start = Instant::now();

        for step in 1..=n_steps {
            gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_per_dt)
                .expect("GPU tree step failed");

            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            gpu_seg_values.push(seg);

            if step % 20 == 0 || step == n_steps {
                let seg_change = (seg - seg_0_gpu) / seg_0_gpu * 100.0;
                println!("  Step {:3}: S = {:.6} ({:+.2}%)", step, seg, seg_change);
            }
        }

        let gpu_total_time = gpu_start.elapsed().as_secs_f64();
        let gpu_final_seg = *gpu_seg_values.last().unwrap();
        println!("GPU tree total time: {:.1}s ({:.0} ms/step)",
                 gpu_total_time, gpu_total_time * 1000.0 / n_steps as f64);

        // ═══════════════════════════════════════════════════════════════════
        // Comparison Results
        // ═══════════════════════════════════════════════════════════════════
        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                     COMPARISON RESULTS                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        let seg_diff_percent = (gpu_final_seg - kdk_final_seg).abs() / kdk_final_seg * 100.0;

        println!("  S(0)   Morton+DKD:  {:.6}", seg_0_kdk);
        println!("  S(0)   GPU tree:    {:.6}", seg_0_gpu);
        println!();
        println!("  S({:3}) Morton+DKD:  {:.6}", n_steps, kdk_final_seg);
        println!("  S({:3}) GPU tree:    {:.6}", n_steps, gpu_final_seg);
        println!("  Difference:  {:.2}%", seg_diff_percent);
        println!();

        // Step-by-step comparison
        println!("  Step-by-step S(t) comparison:");
        let checkpoints = [20, 50, 100, 150, 200];
        for &step in checkpoints.iter() {
            if step <= n_steps {
                let ref_s = kdk_seg_values[step - 1];
                let gpu_s = gpu_seg_values[step - 1];
                let diff = (gpu_s - ref_s).abs() / ref_s * 100.0;
                println!("    Step {:3}: Morton={:.6}  GPU={:.6}  Δ={:.2}%",
                         step, ref_s, gpu_s, diff);
            }
        }
        println!();

        // Speedup
        let speedup = kdk_total_time / gpu_total_time;
        println!("  Morton+DKD time/step:  {:.0} ms", kdk_total_time * 1000.0 / n_steps as f64);
        println!("  GPU tree  time/step:  {:.0} ms", gpu_total_time * 1000.0 / n_steps as f64);
        println!("  Speedup:        {:.1}×", speedup);
        println!();

        // Validation
        let tolerance = 10.0;  // ±10%
        if seg_diff_percent <= tolerance {
            println!("  ╔═══════════════════════════════════════════════════════════╗");
            println!("  ║  ✓ VALIDATED: S({}) within ±{:.0}% ({:.2}% diff)      ║",
                     n_steps, tolerance, seg_diff_percent);
            println!("  ║    GPU tree build is physics-accurate                    ║");
            println!("  ║    Ready to proceed to opt5 (incremental updates)        ║");
            println!("  ╚═══════════════════════════════════════════════════════════╝");
        } else {
            println!("  ╔═══════════════════════════════════════════════════════════╗");
            println!("  ║  ✗ NOT VALIDATED: S({}) differs by {:.2}%             ║",
                     n_steps, seg_diff_percent);
            println!("  ║    Exceeds ±{:.0}% tolerance                              ║", tolerance);
            println!("  ║    Investigation required                                 ║");
            println!("  ╚═══════════════════════════════════════════════════════════╝");
        }
        println!();
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support. Build with --features cuda");
    }
}
