/// Janus N-body GPU — Anisotropic Mode Test (Jour 1)
/// Compare Run A (attraction only) vs Run B (Janus alpha=1)
/// Single-mode perturbation: pos.x += A * sin(kx * pos.x)
/// Measures anisotropy = sigma_x / mean(sigma_y, sigma_z)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::f64::consts::PI;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Parameters from janus_roadmap.md Jour 1
        let n_particles: usize = 4_000_000;
        let eta: f64 = 1.045;
        let steps: usize = 500;
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let box_size: f64 = 400.0;  // 400 Mpc

        // Grid dimensions: cube root of N rounded
        let grid_side = (n_particles as f64).powf(1.0/3.0).round() as usize;
        let actual_n = grid_side * grid_side * grid_side;

        // Perturbation parameters
        let kx = 2.0 * PI / box_size;
        let amplitude = 0.02 * box_size;  // A = 0.02 * box_size

        // Output directory
        let timestamp = chrono_lite();
        let output_dir = format!("/app/output/aniso_test_{}", timestamp);
        fs::create_dir_all(&output_dir).ok();

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus Anisotropic Mode Test — Jour 1                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nParameters:");
        println!("  Grid: {}³ = {} particles", grid_side, actual_n);
        println!("  box = {} Mpc", box_size);
        println!("  eta = {}", eta);
        println!("  theta = {}", theta);
        println!("  dt = {}", dt);
        println!("  steps = {}", steps);
        println!("\nPerturbation:");
        println!("  pos.x += A * sin(kx * pos.x)");
        println!("  A = {:.2} Mpc", amplitude);
        println!("  kx = 2*PI / box = {:.6}", kx);
        println!("\nOutput: {}\n", output_dir);

        // Generate initial grid positions
        let spacing = box_size / (grid_side as f64);
        let mut positions: Vec<[f64; 3]> = Vec::with_capacity(actual_n);
        let mut velocities: Vec<[f64; 3]> = Vec::with_capacity(actual_n);
        let mut signs: Vec<i32> = Vec::with_capacity(actual_n);

        let n_positive = (actual_n as f64 / (1.0 + eta)) as usize;

        println!("Generating {}³ grid with single-mode perturbation...", grid_side);

        for iz in 0..grid_side {
            for iy in 0..grid_side {
                for ix in 0..grid_side {
                    let idx = iz * grid_side * grid_side + iy * grid_side + ix;

                    // Base grid position (centered around box/2)
                    let x0 = (ix as f64 + 0.5) * spacing;
                    let y0 = (iy as f64 + 0.5) * spacing;
                    let z0 = (iz as f64 + 0.5) * spacing;

                    // Apply single-mode perturbation ONLY to x
                    // pos.x += A * sin(kx * pos.x)
                    let x = x0 + amplitude * (kx * x0).sin();

                    positions.push([x, y0, z0]);
                    velocities.push([0.0, 0.0, 0.0]);  // Zero initial velocities

                    // Assign signs: first n_positive are +1, rest -1
                    signs.push(if idx < n_positive { 1 } else { -1 });
                }
            }
        }

        println!("  N+ = {} ({:.1}%)", n_positive, n_positive as f64 / actual_n as f64 * 100.0);
        println!("  N- = {} ({:.1}%)", actual_n - n_positive, (actual_n - n_positive) as f64 / actual_n as f64 * 100.0);

        // Compute initial sigma values (convert to flat for compute_sigma)
        let pos_flat_init: Vec<f64> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        let (sx0, sy0, sz0) = compute_sigma(&pos_flat_init);
        println!("\nInitial sigma values:");
        println!("  sigma_x = {:.6}", sx0);
        println!("  sigma_y = {:.6}", sy0);
        println!("  sigma_z = {:.6}", sz0);
        println!("  anisotropy = {:.6}", sx0 / ((sy0 + sz0) * 0.5));

        // ========== RUN A: Attraction only (cross_factor = 0.0) ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN A: Attraction Only (alpha=0, cross_factor=0.0)         ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_a = run_simulation(
            &positions, &velocities, &signs,
            box_size, theta, dt, steps,
            0.0,  // cross_factor = 0.0 (no cross interaction)
            &format!("{}/run_a.csv", output_dir),
        );

        // ========== RUN B: Janus (cross_factor = -1.0) ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN B: Janus alpha=1 (cross_factor=-1.0)                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_b = run_simulation(
            &positions, &velocities, &signs,
            box_size, theta, dt, steps,
            -1.0,  // cross_factor = -1.0 (Janus repulsion)
            &format!("{}/run_b.csv", output_dir),
        );

        // ========== Combined results ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   Combined Results                                           ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Write combined CSV
        let mut combined = BufWriter::new(
            File::create(format!("{}/combined.csv", output_dir)).unwrap()
        );
        writeln!(combined, "step,aniso_A,aniso_B,sigma_x_A,sigma_x_B,sigma_y_A,sigma_y_B,sigma_z_A,sigma_z_B").unwrap();

        for i in 0..results_a.len().min(results_b.len()) {
            let (step_a, aniso_a, sx_a, sy_a, sz_a) = results_a[i];
            let (_step_b, aniso_b, sx_b, sy_b, sz_b) = results_b[i];
            writeln!(combined, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                step_a, aniso_a, aniso_b, sx_a, sx_b, sy_a, sy_b, sz_a, sz_b).unwrap();
        }

        // Summary
        println!("Summary (step 0 → step {}):", steps);
        println!("\n  Run A (attraction only):");
        if let (Some(first), Some(last)) = (results_a.first(), results_a.last()) {
            println!("    anisotropy: {:.4} → {:.4} ({:+.1}%)",
                first.1, last.1, (last.1 / first.1 - 1.0) * 100.0);
        }
        println!("\n  Run B (Janus alpha=1):");
        if let (Some(first), Some(last)) = (results_b.first(), results_b.last()) {
            println!("    anisotropy: {:.4} → {:.4} ({:+.1}%)",
                first.1, last.1, (last.1 / first.1 - 1.0) * 100.0);
        }

        println!("\n  Interpretation:");
        if let (Some(last_a), Some(last_b)) = (results_a.last(), results_b.last()) {
            if last_a.1 > 1.5 && (last_b.1 - 1.0).abs() < 0.2 {
                println!("    ✓ Run A anisotropy >> 1 (growing) → collapse OK");
                println!("    ✓ Run B anisotropy ≈ 1 (constant) → alpha=1 blocks anisotropic growth");
                println!("    → LINEAR THEORY CONFIRMED");
            } else if last_a.1 > 1.2 {
                println!("    ~ Run A anisotropy growing but slow");
                println!("    ~ Run B anisotropy = {:.3}", last_b.1);
                println!("    → Need more steps or stronger perturbation");
            } else {
                println!("    ? Unexpected behavior - investigate");
                println!("      A final anisotropy: {:.4}", last_a.1);
                println!("      B final anisotropy: {:.4}", last_b.1);
            }
        }

        println!("\nOutput files:");
        println!("  {}/run_a.csv", output_dir);
        println!("  {}/run_b.csv", output_dir);
        println!("  {}/combined.csv", output_dir);
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This binary requires CUDA. Compile with --features cuda");
    }
}

#[cfg(feature = "cuda")]
fn run_simulation(
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    signs: &[i32],
    box_size: f64,
    theta: f64,
    dt: f64,
    steps: usize,
    cross_factor: f64,
    csv_path: &str,
) -> Vec<(usize, f64, f64, f64, f64)> {  // (step, anisotropy, sigma_x, sigma_y, sigma_z)
    let n = positions.len();
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n - n_positive;

    // Convert to flat arrays for new_with_state
    let pos_flat: Vec<f64> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    let vel_flat: Vec<f64> = velocities.iter().flat_map(|v| v.iter().copied()).collect();
    let signs_vec: Vec<i32> = signs.to_vec();

    // Create simulation from provided data
    match GpuNBodySimulation::new_with_state(n_positive, n_negative, box_size, pos_flat, vel_flat, signs_vec) {
        Ok(mut sim) => {
            sim.set_theta(theta);

            let mut results = Vec::with_capacity(steps / 10 + 1);
            let mut csv = BufWriter::new(File::create(csv_path).unwrap());
            writeln!(csv, "step,anisotropy,sigma_x,sigma_y,sigma_z,ke,time_ms").unwrap();

            // Initial state
            let pos0 = sim.get_positions().unwrap();
            let (sx, sy, sz) = compute_sigma(&pos0);
            let aniso = sx / ((sy + sz) * 0.5);
            let ke = sim.kinetic_energy().unwrap();
            results.push((0, aniso, sx, sy, sz));
            writeln!(csv, "0,{:.6},{:.6},{:.6},{:.6},{:.4e},0.0", aniso, sx, sy, sz, ke).unwrap();

            println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
                "Step", "Anisotropy", "sigma_x", "sigma_y", "sigma_z", "Time");
            println!("{}", "─".repeat(70));
            println!("{:>6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10}",
                0, aniso, sx, sy, sz, "-");

            let start = Instant::now();

            for step in 1..=steps {
                let step_start = Instant::now();

                // Step with configurable cross-sign interaction
                if let Err(e) = sim.step_with_cross_factor(dt, cross_factor) {
                    eprintln!("ERROR at step {}: {}", step, e);
                    break;
                }

                let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                // Report every 50 steps
                if step % 50 == 0 || step == steps {
                    let pos = sim.get_positions().unwrap();
                    let (sx, sy, sz) = compute_sigma(&pos);
                    let aniso = sx / ((sy + sz) * 0.5);
                    let ke = sim.kinetic_energy().unwrap();

                    results.push((step, aniso, sx, sy, sz));
                    writeln!(csv, "{},{:.6},{:.6},{:.6},{:.6},{:.4e},{:.1}",
                        step, aniso, sx, sy, sz, ke, step_time_ms).unwrap();

                    println!("{:>6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>8.1} ms",
                        step, aniso, sx, sy, sz, step_time_ms);
                }
            }

            let total_time = start.elapsed();
            println!("\nTotal time: {:.1}s ({:.0} ms/step avg)",
                total_time.as_secs_f64(),
                total_time.as_millis() as f64 / steps as f64);

            results
        }
        Err(e) => {
            eprintln!("Failed to create simulation: {}", e);
            Vec::new()
        }
    }
}

fn compute_sigma(positions: &[f64]) -> (f64, f64, f64) {
    let n = (positions.len() / 3) as f64;
    if n == 0.0 { return (0.0, 0.0, 0.0); }

    // Compute mean position first
    let (mut sum_x, mut sum_y, mut sum_z) = (0.0, 0.0, 0.0);
    for i in 0..(n as usize) {
        sum_x += positions[i * 3];
        sum_y += positions[i * 3 + 1];
        sum_z += positions[i * 3 + 2];
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mean_z = sum_z / n;

    // Compute variance (centered)
    let (mut var_x, mut var_y, mut var_z) = (0.0, 0.0, 0.0);
    for i in 0..(n as usize) {
        let dx = positions[i * 3] - mean_x;
        let dy = positions[i * 3 + 1] - mean_y;
        let dz = positions[i * 3 + 2] - mean_z;
        var_x += dx * dx;
        var_y += dy * dy;
        var_z += dz * dz;
    }

    (
        (var_x / n).sqrt(),
        (var_y / n).sqrt(),
        (var_z / n).sqrt(),
    )
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = d.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}{:02}{:02}", hours, mins, s)
}
