/// Janus N-body GPU — Anisotropic Mode Test (Jour 1-2)
/// Compare Run A (attraction only) vs Run B (Janus α=1) vs Run C (Yukawa α(r))
/// Single-mode perturbation: pos.x += A * sin(kx * x0)
/// Measures mode amplitude δ(t) via Fourier analysis

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
        // Parameters - LINEAR REGIME
        let n_particles: usize = 500_000;
        let eta: f64 = 1.045;
        let steps: usize = 500;
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let box_size: f64 = 400.0;

        let grid_side = (n_particles as f64).powf(1.0/3.0).round() as usize;
        let actual_n = grid_side * grid_side * grid_side;
        let n_positive = (actual_n as f64 / (1.0 + eta)) as usize;
        let n_negative = actual_n - n_positive;

        let kx = 2.0 * PI / box_size;
        let amplitude = 0.002 * box_size;  // 0.2% - LINEAR REGIME

        // Yukawa screening parameters (Jour 2)
        let epsilon = 0.3;   // Breaking strength
        let r_c = 40.0;      // Characteristic scale (Mpc)

        let timestamp = chrono_lite();
        let output_dir = format!("/app/output/aniso_3runs_{}", timestamp);
        fs::create_dir_all(&output_dir).ok();

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus Anisotropic Mode Test — 3 Runs (A/B/C)                 ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nParameters:");
        println!("  Grid: {}³ = {} particles", grid_side, actual_n);
        println!("  N+ = {} ({:.1}%), N- = {} ({:.1}%)",
            n_positive, n_positive as f64 / actual_n as f64 * 100.0,
            n_negative, n_negative as f64 / actual_n as f64 * 100.0);
        println!("  box = {} Mpc, theta = {}, dt = {}, steps = {}", box_size, theta, dt, steps);
        println!("  Perturbation: A = {:.4} Mpc ({:.2}% of box)", amplitude, amplitude / box_size * 100.0);
        println!("  kx = 2π/box = {:.6}", kx);
        println!("  Yukawa: ε = {}, r_c = {} Mpc", epsilon, r_c);
        println!("\nOutput: {}\n", output_dir);

        // Generate grid with single-mode perturbation
        println!("Generating grid with single-mode perturbation...");
        let spacing = box_size / (grid_side as f64);
        let half_box = box_size / 2.0;
        let noise_amp = spacing * 0.001;  // Tiny noise to break degeneracy

        let mut rng = StdRng::seed_from_u64(42);
        let mut positions = Vec::with_capacity(actual_n * 3);
        let velocities = vec![0.0f64; actual_n * 3];
        let mut signs = Vec::with_capacity(actual_n);

        for iz in 0..grid_side {
            for iy in 0..grid_side {
                for ix in 0..grid_side {
                    let idx = iz * grid_side * grid_side + iy * grid_side + ix;

                    // Grid position (centered in [-box/2, box/2])
                    let x0 = (ix as f64 + 0.5) * spacing - half_box;
                    let y0 = (iy as f64 + 0.5) * spacing - half_box;
                    let z0 = (iz as f64 + 0.5) * spacing - half_box;

                    // Tiny noise to break degeneracy
                    let nx: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                    let ny: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                    let nz: f64 = (rng.random::<f64>() - 0.5) * noise_amp;

                    // Single-mode perturbation: x = x0 + A × sin(kx × x0)
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

        // Initial measurements using Fourier method
        let (sx0, sy0, sz0) = compute_sigma(&positions);
        let delta_k0 = compute_mode_amplitude_fourier(&positions, kx, box_size);
        let (delta_rms0, _) = compute_density_contrast(&positions, box_size, 128);

        println!("Initial state:");
        println!("  σx = {:.6}, σy = {:.6}, σz = {:.6}", sx0, sy0, sz0);
        println!("  δ_k(0) = {:.6e} (Fourier mode amplitude)", delta_k0);
        println!("  δ_rms(0) = {:.6} (density contrast RMS)", delta_rms0);
        println!();

        // ========== RUN A: Attraction only ==========
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN A: Attraction Only (cross_factor=0.0)                  ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_a = run_test(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            kx,
            0.0, &format!("{}/run_a.csv", output_dir)
        );

        // ========== RUN B: Janus ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN B: Janus alpha=1 (cross_factor=-1.0)                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_b = run_test(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            kx,
            -1.0, &format!("{}/run_b.csv", output_dir)
        );

        // ========== RUN C: Yukawa ε=0.3, rc=40 ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN C: Yukawa ε=0.3, rc=40  → α(5)=0.74                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_c = run_test_yukawa(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            kx, 0.3, 40.0,
            &format!("{}/run_c.csv", output_dir)
        );

        // ========== RUN D: Yukawa ε=0.3, rc=10 ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN D: Yukawa ε=0.3, rc=10  → α(5)=0.82                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_d = run_test_yukawa(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            kx, 0.3, 10.0,
            &format!("{}/run_d.csv", output_dir)
        );

        // ========== RUN E: Yukawa ε=0.7, rc=40 ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN E: Yukawa ε=0.7, rc=40  → α(5)=0.38                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_e = run_test_yukawa(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions.clone(), velocities.clone(), signs.clone(),
            kx, 0.7, 40.0,
            &format!("{}/run_e.csv", output_dir)
        );

        // ========== RUN F: Yukawa ε=0.7, rc=10 ==========
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   RUN F: Yukawa ε=0.7, rc=10  → α(5)=0.57                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let results_f = run_test_yukawa(
            n_positive, n_negative, box_size, theta, dt, steps,
            positions, velocities, signs,
            kx, 0.7, 10.0,
            &format!("{}/run_f.csv", output_dir)
        );

        // Summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║   SUMMARY                                                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Write combined CSV with all 6 runs
        let mut combined = BufWriter::new(File::create(format!("{}/combined.csv", output_dir)).unwrap());
        writeln!(combined, "step,delta_k_A,delta_k_B,delta_k_C,delta_k_D,delta_k_E,delta_k_F,sigma_x_A,sigma_x_B,sigma_x_C,sigma_x_D,sigma_x_E,sigma_x_F").unwrap();
        let n_results = results_a.len().min(results_b.len()).min(results_c.len())
            .min(results_d.len()).min(results_e.len()).min(results_f.len());
        for i in 0..n_results {
            writeln!(combined, "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                results_a[i].0,
                results_a[i].1, results_b[i].1, results_c[i].1, results_d[i].1, results_e[i].1, results_f[i].1,
                results_a[i].3, results_b[i].3, results_c[i].3, results_d[i].3, results_e[i].3, results_f[i].3).unwrap();
        }

        // Helper to compute growth percentage
        let growth = |r: &Vec<(usize, f64, f64, f64)>| -> f64 {
            if let (Some(r0), Some(rf)) = (r.first(), r.last()) {
                (rf.1 / r0.1 - 1.0) * 100.0
            } else { 0.0 }
        };
        let sigma_change = |r: &Vec<(usize, f64, f64, f64)>| -> f64 {
            if let (Some(r0), Some(rf)) = (r.first(), r.last()) {
                (rf.3 / r0.3 - 1.0) * 100.0
            } else { 0.0 }
        };

        let g_a = growth(&results_a);
        let g_b = growth(&results_b);
        let g_c = growth(&results_c);
        let g_d = growth(&results_d);
        let g_e = growth(&results_e);
        let g_f = growth(&results_f);

        println!("δ_k growth (Fourier mode amplitude):");
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Run │ Parameters          │ α(5Mpc) │ δ_k growth │ vs Run A │");
        println!("├─────┼─────────────────────┼─────────┼────────────┼──────────┤");
        println!("│  A  │ Attraction only     │   1.0   │ {:+7.1}%  │   100%   │", g_a);
        println!("│  B  │ Janus α=1           │  -1.0   │ {:+7.1}%  │   {:4.0}%   │", g_b, g_b/g_a*100.0);
        println!("│  C  │ ε=0.3, rc=40        │   0.74  │ {:+7.1}%  │   {:4.0}%   │", g_c, g_c/g_a*100.0);
        println!("│  D  │ ε=0.3, rc=10        │   0.82  │ {:+7.1}%  │   {:4.0}%   │", g_d, g_d/g_a*100.0);
        println!("│  E  │ ε=0.7, rc=40        │   0.38  │ {:+7.1}%  │   {:4.0}%   │", g_e, g_e/g_a*100.0);
        println!("│  F  │ ε=0.7, rc=10        │   0.57  │ {:+7.1}%  │   {:4.0}%   │", g_f, g_f/g_a*100.0);
        println!("└─────────────────────────────────────────────────────────────┘");

        println!("\nσx evolution:");
        println!("  A: {:+.2}%  B: {:+.2}%  C: {:+.2}%  D: {:+.2}%  E: {:+.2}%  F: {:+.2}%",
            sigma_change(&results_a), sigma_change(&results_b), sigma_change(&results_c),
            sigma_change(&results_d), sigma_change(&results_e), sigma_change(&results_f));

        println!("\nInterpretation:");
        let best_yukawa = if g_e > g_d && g_e > g_c && g_e > g_f { ("E", g_e, "ε=0.7, rc=40") }
            else if g_f > g_d && g_f > g_c { ("F", g_f, "ε=0.7, rc=10") }
            else if g_d > g_c { ("D", g_d, "ε=0.3, rc=10") }
            else { ("C", g_c, "ε=0.3, rc=40") };

        if best_yukawa.1 > g_b * 1.5 {
            println!("  ✓ Run {} ({}) shows SIGNIFICANTLY faster growth!", best_yukawa.0, best_yukawa.2);
            println!("    Growth: {:+.1}% vs {:+.1}% for Janus α=1 ({:.0}× faster)",
                best_yukawa.1, g_b, best_yukawa.1 / g_b);
            println!("  → Yukawa screening RESTORES perturbation growth");
            println!("  → Filament formation possible with α(r) < 1 at small scales");
        } else if best_yukawa.1 > g_b * 1.1 {
            println!("  ~ Run {} ({}) shows moderately faster growth", best_yukawa.0, best_yukawa.2);
            println!("    Growth: {:+.1}% vs {:+.1}% for Janus α=1", best_yukawa.1, g_b);
        } else {
            println!("  ? All Yukawa runs similar to Janus α=1");
            println!("  → May need even larger ε or non-linear regime");
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
    kx: f64,
    cross_factor: f64, csv_path: &str
) -> Vec<(usize, f64, f64, f64)> {  // (step, delta_k, delta_rms, sigma_x)
    match GpuNBodySimulation::new_with_state(n_positive, n_negative, box_size, positions, velocities, signs) {
        Ok(mut sim) => {
            sim.set_theta(theta);
            let mut results = Vec::new();
            let mut csv = BufWriter::new(File::create(csv_path).unwrap());
            writeln!(csv, "step,delta_k,delta_rms,sigma_x,sigma_y,sigma_z,time_ms").unwrap();

            // Initial measurements
            let pos = sim.get_positions().unwrap();
            let (sx, sy, sz) = compute_sigma(&pos);
            let delta_k = compute_mode_amplitude_fourier(&pos, kx, box_size);
            let (delta_rms, _) = compute_density_contrast(&pos, box_size, 128);
            results.push((0, delta_k, delta_rms, sx));
            writeln!(csv, "0,{:.6e},{:.6e},{:.6},{:.6},{:.6},0", delta_k, delta_rms, sx, sy, sz).unwrap();

            println!("{:>6}  {:>12}  {:>10}  {:>10}  {:>10}", "Step", "δ_k", "δ_rms", "σx", "Time");
            println!("{}", "─".repeat(60));
            println!("{:>6}  {:>12.6e}  {:>10.6}  {:>10.4}  {:>10}", 0, delta_k, delta_rms, sx, "-");

            let start = Instant::now();
            for step in 1..=steps {
                let t0 = Instant::now();
                if sim.step_with_cross_factor(dt, cross_factor).is_err() { break; }
                let ms = t0.elapsed().as_millis();

                if step % 50 == 0 || step == steps {
                    let pos = sim.get_positions().unwrap();
                    let (sx, sy, sz) = compute_sigma(&pos);
                    let delta_k = compute_mode_amplitude_fourier(&pos, kx, box_size);
                    let (delta_rms, _) = compute_density_contrast(&pos, box_size, 128);
                    results.push((step, delta_k, delta_rms, sx));
                    writeln!(csv, "{},{:.6e},{:.6e},{:.6},{:.6},{:.6},{}", step, delta_k, delta_rms, sx, sy, sz, ms).unwrap();
                    println!("{:>6}  {:>12.6e}  {:>10.6}  {:>10.4}  {:>8} ms", step, delta_k, delta_rms, sx, ms);
                }
            }
            println!("Total: {:.1}s\n", start.elapsed().as_secs_f64());
            results
        }
        Err(e) => { eprintln!("Init failed: {}", e); Vec::new() }
    }
}

#[cfg(feature = "cuda")]
fn run_test_yukawa(
    n_positive: usize, n_negative: usize, box_size: f64,
    theta: f64, dt: f64, steps: usize,
    positions: Vec<f64>, velocities: Vec<f64>, signs: Vec<i32>,
    kx: f64, epsilon: f64, r_c: f64,
    csv_path: &str
) -> Vec<(usize, f64, f64, f64)> {  // (step, delta_k, delta_rms, sigma_x)
    match GpuNBodySimulation::new_with_state(n_positive, n_negative, box_size, positions, velocities, signs) {
        Ok(mut sim) => {
            sim.set_theta(theta);
            let mut results = Vec::new();
            let mut csv = BufWriter::new(File::create(csv_path).unwrap());
            writeln!(csv, "step,delta_k,delta_rms,sigma_x,sigma_y,sigma_z,time_ms").unwrap();

            // Initial measurements
            let pos = sim.get_positions().unwrap();
            let (sx, sy, sz) = compute_sigma(&pos);
            let delta_k = compute_mode_amplitude_fourier(&pos, kx, box_size);
            let (delta_rms, _) = compute_density_contrast(&pos, box_size, 128);
            results.push((0, delta_k, delta_rms, sx));
            writeln!(csv, "0,{:.6e},{:.6e},{:.6},{:.6},{:.6},0", delta_k, delta_rms, sx, sy, sz).unwrap();

            println!("{:>6}  {:>12}  {:>10}  {:>10}  {:>10}", "Step", "δ_k", "δ_rms", "σx", "Time");
            println!("{}", "─".repeat(60));
            println!("{:>6}  {:>12.6e}  {:>10.6}  {:>10.4}  {:>10}", 0, delta_k, delta_rms, sx, "-");

            let start = Instant::now();
            for step in 1..=steps {
                let t0 = Instant::now();
                if sim.step_with_yukawa(dt, epsilon, r_c).is_err() { break; }
                let ms = t0.elapsed().as_millis();

                if step % 50 == 0 || step == steps {
                    let pos = sim.get_positions().unwrap();
                    let (sx, sy, sz) = compute_sigma(&pos);
                    let delta_k = compute_mode_amplitude_fourier(&pos, kx, box_size);
                    let (delta_rms, _) = compute_density_contrast(&pos, box_size, 128);
                    results.push((step, delta_k, delta_rms, sx));
                    writeln!(csv, "{},{:.6e},{:.6e},{:.6},{:.6},{:.6},{}", step, delta_k, delta_rms, sx, sy, sz, ms).unwrap();
                    println!("{:>6}  {:>12.6e}  {:>10.6}  {:>10.4}  {:>8} ms", step, delta_k, delta_rms, sx, ms);
                }
            }
            println!("Total: {:.1}s\n", start.elapsed().as_secs_f64());
            results
        }
        Err(e) => { eprintln!("Init failed: {}", e); Vec::new() }
    }
}

/// Compute mode amplitude using density field Fourier analysis
/// This method works regardless of particle reordering (Morton sort)
/// δ_k = (1/N) × Σ exp(i k x) = (1/N) × (Σ cos(kx) + i Σ sin(kx))
/// For our perturbation pos.x = x0 + A sin(kx x0), at t=0:
///   δ_k ≈ A × k / 2 (for small A)
/// We measure |δ_k| / |δ_k(t=0)| to track growth
fn compute_mode_amplitude_fourier(pos: &[f64], kx: f64, box_size: f64) -> f64 {
    let n = pos.len() / 3;
    if n == 0 { return 0.0; }

    // Compute Fourier mode at k = kx (only x-component matters)
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;

    for i in 0..n {
        let x = pos[i * 3];
        let phase = kx * x;
        sum_cos += phase.cos();
        sum_sin += phase.sin();
    }

    // |δ_k| = sqrt(cos² + sin²) / N
    let delta_k = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n as f64;

    // For a uniform distribution, δ_k ≈ 0
    // For our sinusoidal perturbation, δ_k is proportional to amplitude
    delta_k
}

/// Compute density contrast in 1D bins along x-axis
/// Returns (delta_rms, delta_max) where delta = (rho - rho_mean) / rho_mean
fn compute_density_contrast(pos: &[f64], box_size: f64, n_bins: usize) -> (f64, f64) {
    let n = pos.len() / 3;
    if n == 0 { return (0.0, 0.0); }

    let half_box = box_size / 2.0;
    let bin_width = box_size / n_bins as f64;

    // Count particles in each bin
    let mut counts = vec![0usize; n_bins];
    for i in 0..n {
        let x = pos[i * 3];
        // Map from [-box/2, box/2] to [0, n_bins)
        let bin = ((x + half_box) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }

    // Mean count per bin
    let mean = n as f64 / n_bins as f64;

    // Compute density contrast
    let mut delta_sq_sum = 0.0;
    let mut delta_max = 0.0f64;
    for &c in &counts {
        let delta = (c as f64 - mean) / mean;
        delta_sq_sum += delta * delta;
        delta_max = delta_max.max(delta.abs());
    }

    let delta_rms = (delta_sq_sum / n_bins as f64).sqrt();
    (delta_rms, delta_max)
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
