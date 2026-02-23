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
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Parameters - 500K for actual test
        let n_particles: usize = 500_000;
        let eta: f64 = 1.045;
        let steps: usize = 2000;  // More steps to see collapse
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let box_size: f64 = 400.0;

        let grid_side = (n_particles as f64).powf(1.0/3.0).round() as usize;
        let actual_n = grid_side * grid_side * grid_side;
        let n_positive = (actual_n as f64 / (1.0 + eta)) as usize;
        let n_negative = actual_n - n_positive;

        let kx = 2.0 * PI / box_size;
        let amplitude = 0.10 * box_size;  // 10% perturbation

        let timestamp = chrono_lite();
        let output_dir = format!("/app/output/aniso_test_{}", timestamp);
        fs::create_dir_all(&output_dir).ok();

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus Anisotropic Mode Test — Jour 1                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nParameters:");
        println!("  Grid: {}³ = {} particles", grid_side, actual_n);
        println!("  N+ = {} ({:.1}%)", n_positive, n_positive as f64 / actual_n as f64 * 100.0);
        println!("  N- = {} ({:.1}%)", n_negative, n_negative as f64 / actual_n as f64 * 100.0);
        println!("  box = {} Mpc, theta = {}, dt = {}, steps = {}", box_size, theta, dt, steps);
        println!("  Perturbation: A = {:.2} Mpc, kx = {:.6}", amplitude, kx);
        println!("\nOutput: {}\n", output_dir);

        // Generate grid with small noise to avoid degenerate octree
        println!("Generating grid with single-mode perturbation + noise...");
        let spacing = box_size / (grid_side as f64);
        let noise_amp = spacing * 0.01;  // 1% of spacing to break degeneracy

        let mut rng = StdRng::seed_from_u64(42);
        let mut positions = Vec::with_capacity(actual_n * 3);
        let mut velocities = vec![0.0f64; actual_n * 3];
        let mut signs = Vec::with_capacity(actual_n);

        let half_box = box_size / 2.0;
        for iz in 0..grid_side {
            for iy in 0..grid_side {
                for ix in 0..grid_side {
                    let idx = iz * grid_side * grid_side + iy * grid_side + ix;
                    // CENTERED positions: [-box/2, box/2] as expected by LinearOctree
                    let x0 = (ix as f64 + 0.5) * spacing - half_box;
                    let y0 = (iy as f64 + 0.5) * spacing - half_box;
                    let z0 = (iz as f64 + 0.5) * spacing - half_box;

                    // Add small noise to break grid degeneracy
                    let nx: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                    let ny: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                    let nz: f64 = (rng.random::<f64>() - 0.5) * noise_amp;

                    // Single-mode perturbation + noise
                    let x = x0 + amplitude * (kx * x0).sin() + nx;
                    let y = y0 + ny;
                    let z = z0 + nz;

                    positions.push(x);
                    positions.push(y);
                    positions.push(z);
                    signs.push(if idx < n_positive { 1 } else { -1 });
                }
            }
        }

        let (sx0, sy0, sz0) = compute_sigma(&positions);
        let aniso0 = sx0 / ((sy0 + sz0) * 0.5);
        println!("Initial: sigma_x={:.4}, sigma_y={:.4}, sigma_z={:.4}, aniso={:.4}\n",
            sx0, sy0, sz0, aniso0);

        // ========== RUN A: Attraction only ==========
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN A: Attraction Only (cross_factor=0.0)                  ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_a = run_test(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            0.0, &format!("{}/run_a.csv", output_dir)
        );

        // ========== RUN B: Janus ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN B: Janus alpha=1 (cross_factor=-1.0)                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_b = run_test(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions, velocities, signs,
            -1.0, &format!("{}/run_b.csv", output_dir)
        );

        // Summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   SUMMARY                                                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Write combined CSV
        let mut combined = BufWriter::new(File::create(format!("{}/combined.csv", output_dir)).unwrap());
        writeln!(combined, "step,aniso_A,aniso_B").unwrap();
        for i in 0..results_a.len().min(results_b.len()) {
            writeln!(combined, "{},{:.6},{:.6}", results_a[i].0, results_a[i].1, results_b[i].1).unwrap();
        }

        if let (Some(a0), Some(af), Some(b0), Some(bf)) =
            (results_a.first(), results_a.last(), results_b.first(), results_b.last()) {
            println!("Run A (attraction only): aniso {:.4} → {:.4} ({:+.1}%)",
                a0.1, af.1, (af.1 / a0.1 - 1.0) * 100.0);
            println!("Run B (Janus α=1):       aniso {:.4} → {:.4} ({:+.1}%)",
                b0.1, bf.1, (bf.1 / b0.1 - 1.0) * 100.0);

            println!("\nInterpretation:");
            if af.1 > 1.2 && (bf.1 - b0.1).abs() < 0.3 {
                println!("  ✓ A growing, B stable → LINEAR THEORY CONFIRMED");
                println!("  → α=1 blocks anisotropic growth as predicted");
            } else {
                println!("  ? A={:.3}, B={:.3} - needs investigation", af.1, bf.1);
            }
        }

        println!("\nOutput: {}/combined.csv", output_dir);
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("ERROR: Requires CUDA. Compile with --features cuda");
}

#[cfg(feature = "cuda")]
fn run_test(
    n_positive: usize, n_negative: usize, box_size: f64,
    theta: f64, dt: f64, steps: usize,
    positions: Vec<f64>, velocities: Vec<f64>, signs: Vec<i32>,
    cross_factor: f64, csv_path: &str
) -> Vec<(usize, f64)> {
    match GpuNBodySimulation::new_with_state(n_positive, n_negative, box_size, positions, velocities, signs) {
        Ok(mut sim) => {
            sim.set_theta(theta);
            let mut results = Vec::new();
            let mut csv = BufWriter::new(File::create(csv_path).unwrap());
            writeln!(csv, "step,anisotropy,sigma_x,sigma_y,sigma_z").unwrap();

            // Initial
            let pos = sim.get_positions().unwrap();
            let (sx, sy, sz) = compute_sigma(&pos);
            let aniso = sx / ((sy + sz) * 0.5);
            results.push((0, aniso));
            writeln!(csv, "0,{:.6},{:.6},{:.6},{:.6}", aniso, sx, sy, sz).unwrap();

            println!("{:>6}  {:>10}  {:>10}", "Step", "Anisotropy", "Time");
            println!("{}", "─".repeat(35));
            println!("{:>6}  {:>10.4}  {:>10}", 0, aniso, "-");

            let start = Instant::now();
            for step in 1..=steps {
                let t0 = Instant::now();
                if sim.step_with_cross_factor(dt, cross_factor).is_err() { break; }
                let ms = t0.elapsed().as_millis();

                if step % 50 == 0 || step == steps {
                    let pos = sim.get_positions().unwrap();
                    let (sx, sy, sz) = compute_sigma(&pos);
                    let aniso = sx / ((sy + sz) * 0.5);
                    results.push((step, aniso));
                    writeln!(csv, "{},{:.6},{:.6},{:.6},{:.6}", step, aniso, sx, sy, sz).unwrap();
                    println!("{:>6}  {:>10.4}  {:>8} ms", step, aniso, ms);
                }
            }
            println!("Total: {:.1}s\n", start.elapsed().as_secs_f64());
            results
        }
        Err(e) => { eprintln!("Init failed: {}", e); Vec::new() }
    }
}

fn compute_sigma(pos: &[f64]) -> (f64, f64, f64) {
    let n = (pos.len() / 3) as f64;
    if n == 0.0 { return (0.0, 0.0, 0.0); }

    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    for i in 0..(n as usize) {
        sx += pos[i*3]; sy += pos[i*3+1]; sz += pos[i*3+2];
    }
    let (mx, my, mz) = (sx/n, sy/n, sz/n);

    let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
    for i in 0..(n as usize) {
        let dx = pos[i*3] - mx; let dy = pos[i*3+1] - my; let dz = pos[i*3+2] - mz;
        vx += dx*dx; vy += dy*dy; vz += dz*dz;
    }
    ((vx/n).sqrt(), (vy/n).sqrt(), (vz/n).sqrt())
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let s = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("{:02}{:02}{:02}", (s/3600)%24, (s/60)%60, s%60)
}
