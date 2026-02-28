//! Quick GPU N-body run with snapshot output for visual validation
//! No cosmological expansion - pure Janus physics

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    let n_particles: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let n_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let output_dir = std::env::args()
        .nth(3)
        .unwrap_or_else(|| format!("/app/output/gpu_visual_{}", n_particles));

    let eta = 1.045;
    let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
    let dt = 0.01;
    let snapshot_interval = 10;

    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();
    fs::create_dir_all(format!("{}/render_data", output_dir)).ok();

    println!("=== GPU Visual Validation Run ===\n");
    println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
    println!("  Box = {:.2}", box_size);
    println!("  Steps = {}", n_steps);
    println!("  Snapshot every {} steps", snapshot_interval);
    println!("  Output: {}\n", output_dir);

    // No cosmological expansion
    let a = 1.0;
    let h = 0.0;
    let dtau_per_dt = 0.0;

    // Initialize
    let mut gpu_sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
        .expect("Failed to create GPU simulation");
    gpu_sim.virialize().expect("Virialization failed");

    let seg_0 = gpu_sim.segregation_distance().expect("Failed to compute segregation");
    let ke_0 = gpu_sim.kinetic_energy().expect("Failed to compute KE");

    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.6}\n", seg_0);

    // Save initial snapshot
    save_snapshot(&gpu_sim, &output_dir, 0, box_size, seg_0, 1.0);

    // Open CSV for time series
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Cannot create CSV");
    writeln!(csv, "step,ke,ke_ratio,seg,seg_ratio").ok();
    writeln!(csv, "0,{:.6e},1.0,{:.6},{:.6}", ke_0, seg_0, seg_0 / box_size).ok();

    let start = Instant::now();
    let mut last_log = Instant::now();

    for step in 1..=n_steps {
        // Step without expansion
        gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_per_dt)
            .expect("GPU step failed");

        // Compute metrics every 10 steps
        if step % 10 == 0 || step == n_steps {
            let ke = gpu_sim.kinetic_energy().expect("Failed to compute KE");
            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            let ke_ratio = ke / ke_0;
            let seg_ratio = seg / box_size;

            writeln!(csv, "{},{:.6e},{:.6},{:.6},{:.6}", step, ke, ke_ratio, seg, seg_ratio).ok();
        }

        // Save snapshot
        if step % snapshot_interval == 0 {
            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            let ke = gpu_sim.kinetic_energy().expect("Failed to compute KE");
            save_snapshot(&gpu_sim, &output_dir, step, box_size, seg, ke / ke_0);
        }

        // Progress log
        if step % 50 == 0 || last_log.elapsed().as_secs() >= 5 {
            let elapsed = start.elapsed().as_secs_f64();
            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            let ke = gpu_sim.kinetic_energy().expect("Failed to compute KE");
            let seg_change = (seg - seg_0) / seg_0 * 100.0;

            println!("Step {:>5}/{}: KE/KE₀={:.4}, Seg={:.6} ({:+.1}%), {:.1}s",
                     step, n_steps, ke / ke_0, seg, seg_change, elapsed);
            last_log = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let seg_final = gpu_sim.segregation_distance().expect("Failed to compute segregation");
    let ke_final = gpu_sim.kinetic_energy().expect("Failed to compute KE");

    println!("\n=== Final Results ===");
    println!("  Steps: {}", n_steps);
    println!("  KE/KE₀ = {:.4}", ke_final / ke_0);
    println!("  Seg: {:.6} → {:.6} ({:+.2}%)", seg_0, seg_final, (seg_final - seg_0) / seg_0 * 100.0);
    println!("  Time: {:.1}s ({:.2}ms/step)", elapsed, elapsed * 1000.0 / n_steps as f64);
    println!("  Output: {}", output_dir);
}

#[cfg(feature = "cuda")]
fn save_snapshot(gpu_sim: &GpuNBodySimulation, output_dir: &str, step: usize, box_size: f64, seg: f64, ke_ratio: f64) {
    // Get positions (f64) and signs (i32) from GPU
    let pos_f64 = gpu_sim.get_positions().expect("Failed to get positions");
    let signs_i32 = gpu_sim.get_signs().expect("Failed to get signs");

    // Convert to f32 for compact storage
    let pos_f32: Vec<f32> = pos_f64.iter().map(|&x| x as f32).collect();
    let signs_i8: Vec<i8> = signs_i32.iter().map(|&s| s as i8).collect();

    let path = format!("{}/render_data/step_{:06}.bin", output_dir, step);
    let mut file = File::create(&path).expect("Cannot create snapshot file");

    // Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + n(u32)
    let n = (pos_f32.len() / 3) as u32;
    file.write_all(&(step as u32).to_le_bytes()).ok();
    file.write_all(&box_size.to_le_bytes()).ok();
    file.write_all(&seg.to_le_bytes()).ok();
    file.write_all(&ke_ratio.to_le_bytes()).ok();
    file.write_all(&n.to_le_bytes()).ok();

    // Positions: N×3×f32
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pos_f32.as_ptr() as *const u8, pos_f32.len() * 4)
    };
    file.write_all(pos_bytes).ok();

    // Signs: N×i8
    let signs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(signs_i8.as_ptr() as *const u8, signs_i8.len())
    };
    file.write_all(signs_bytes).ok();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires CUDA support. Build with --features cuda");
}
