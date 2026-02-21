/// GPU Barnes-Hut validation test
/// Compares GPU results against CPU reference implementation
///
/// Validation criteria:
/// - 100K particles, 50 steps
/// - Segregation difference: < 5% vs CPU
/// - KE/KE0 < 50 at step 50

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::nbody::NBodySimulation;
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(70));
    println!("GPU Barnes-Hut Validation Test");
    println!("{}", "=".repeat(70));

    let n_particles = 100_000;
    let eta = 1.045;
    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;
    let box_size = 100.0;
    let steps = 50;
    let dt = 0.001;

    println!("\nParameters:");
    println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
    println!("  eta = {:.3}", eta);
    println!("  box = {}", box_size);
    println!("  steps = {}", steps);
    println!("  dt = {}", dt);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    println!("\n--- CPU Reference (Barnes-Hut) ---");

    let mut cpu_sim = NBodySimulation::new(n_positive, n_negative, box_size);

    let ke0_cpu = cpu_sim.kinetic_energy();
    let seg0_cpu = cpu_sim.segregation_distance();

    println!("Initial: KE = {:.4e}, Seg = {:.4}", ke0_cpu, seg0_cpu);

    let start_cpu = Instant::now();

    for step in 1..=steps {
        cpu_sim.step(dt);

        if step % 10 == 0 || step == steps {
            let ke = cpu_sim.kinetic_energy();
            let seg = cpu_sim.segregation_distance();
            let ke_ratio = ke / ke0_cpu;
            println!("  Step {:3}: KE/KE0 = {:7.2}, Seg = {:.4}", step, ke_ratio, seg);
        }
    }

    let cpu_time = start_cpu.elapsed();
    let ke_final_cpu = cpu_sim.kinetic_energy();
    let seg_final_cpu = cpu_sim.segregation_distance();
    let ke_ratio_cpu = ke_final_cpu / ke0_cpu;
    let seg_change_cpu = (seg_final_cpu - seg0_cpu) / seg0_cpu * 100.0;

    println!("\nCPU Results:");
    println!("  Time: {:.2?}", cpu_time);
    println!("  KE/KE0: {:.2}", ke_ratio_cpu);
    println!("  Segregation: {:.4} -> {:.4} ({:+.1}%)", seg0_cpu, seg_final_cpu, seg_change_cpu);

    // =========================================================================
    // GPU Test
    // =========================================================================
    #[cfg(feature = "cuda")]
    {
        println!("\n--- GPU (Barnes-Hut CUDA) ---");

        match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
            Ok(mut gpu_sim) => {
                let ke0_gpu = gpu_sim.kinetic_energy().unwrap();
                let seg0_gpu = gpu_sim.segregation_distance().unwrap();

                println!("Initial: KE = {:.4e}, Seg = {:.4}", ke0_gpu, seg0_gpu);

                let start_gpu = Instant::now();

                for step in 1..=steps {
                    if let Err(e) = gpu_sim.step(dt) {
                        eprintln!("GPU error at step {}: {}", step, e);
                        break;
                    }

                    if step % 10 == 0 || step == steps {
                        let ke = gpu_sim.kinetic_energy().unwrap();
                        let seg = gpu_sim.segregation_distance().unwrap();
                        let ke_ratio = ke / ke0_gpu;
                        println!("  Step {:3}: KE/KE0 = {:7.2}, Seg = {:.4}", step, ke_ratio, seg);
                    }
                }

                let gpu_time = start_gpu.elapsed();
                let ke_final_gpu = gpu_sim.kinetic_energy().unwrap();
                let seg_final_gpu = gpu_sim.segregation_distance().unwrap();
                let ke_ratio_gpu = ke_final_gpu / ke0_gpu;
                let seg_change_gpu = (seg_final_gpu - seg0_gpu) / seg0_gpu * 100.0;

                println!("\nGPU Results:");
                println!("  Time: {:.2?}", gpu_time);
                println!("  KE/KE0: {:.2}", ke_ratio_gpu);
                println!("  Segregation: {:.4} -> {:.4} ({:+.1}%)", seg0_gpu, seg_final_gpu, seg_change_gpu);

                // =========================================================================
                // Comparison
                // =========================================================================
                println!("\n{}", "=".repeat(70));
                println!("VALIDATION RESULTS");
                println!("{}", "=".repeat(70));

                let seg_diff_pct = (seg_final_gpu - seg_final_cpu).abs() / seg_final_cpu * 100.0;
                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

                println!("\n1. Segregation Comparison (f64 precision):");
                println!("   CPU: {:.4}", seg_final_cpu);
                println!("   GPU: {:.4}", seg_final_gpu);
                println!("   Difference: {:.4}%", seg_diff_pct);
                if seg_diff_pct < 1.0 {
                    println!("   Status: PASS (< 1%)");
                } else if seg_diff_pct < 5.0 {
                    println!("   Status: MARGINAL (1-5%)");
                } else {
                    println!("   Status: FAIL (>= 5%)");
                }

                println!("\n2. Energy Stability:");
                println!("   CPU KE/KE0: {:.2}", ke_ratio_cpu);
                println!("   GPU KE/KE0: {:.2}", ke_ratio_gpu);
                if ke_ratio_gpu < 50.0 {
                    println!("   Status: PASS (< 50)");
                } else {
                    println!("   Status: FAIL (>= 50)");
                }

                println!("\n3. Performance:");
                println!("   CPU time: {:.2?}", cpu_time);
                println!("   GPU time: {:.2?}", gpu_time);
                println!("   Speedup: {:.1}x", speedup);

                println!("\n{}", "=".repeat(70));
                if seg_diff_pct < 1.0 && ke_ratio_gpu < 50.0 && speedup > 4.0 {
                    println!("OVERALL: PASS - GPU f64 implementation validated");
                    println!("Ready for 10M particle simulation");
                } else if seg_diff_pct < 5.0 && ke_ratio_gpu < 50.0 {
                    println!("OVERALL: MARGINAL - Consider f32 if speedup insufficient");
                    if speedup < 2.0 {
                        println!("Recommendation: Accept f32 with documented 5.74% error");
                    }
                } else {
                    println!("OVERALL: FAIL - GPU implementation needs adjustment");
                }
                println!("{}", "=".repeat(70));
            }
            Err(e) => {
                eprintln!("Failed to initialize GPU simulation: {}", e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n--- GPU Test Skipped (CUDA feature not enabled) ---");
        println!("To enable GPU: cargo run --release --features cuda --bin nbody_gpu_test");
    }
}
