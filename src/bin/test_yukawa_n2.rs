/// Diagnostic: Compare N² brute force vs Barnes-Hut for Yukawa
/// 10K particles, 1 step, Run B (α=1) vs Run E (ε=0.7, rc=40)

use std::f64::consts::PI;
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles: usize = 10_000;
        let box_size: f64 = 400.0;
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let kx = 2.0 * PI / box_size;
        let amplitude = 0.002 * box_size;

        // Yukawa parameters for Run E
        let epsilon = 0.7;
        let r_c = 40.0;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Diagnostic: N² vs Barnes-Hut for Yukawa                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Parameters: N={}, box={} Mpc, dt={}", n_particles, box_size, dt);
        println!("Yukawa: ε={}, r_c={} Mpc\n", epsilon, r_c);

        // Generate grid
        let grid_side = (n_particles as f64).powf(1.0/3.0).round() as usize;
        let actual_n = grid_side * grid_side * grid_side;
        let n_positive = actual_n / 2;
        let n_negative = actual_n - n_positive;
        let spacing = box_size / (grid_side as f64);
        let half_box = box_size / 2.0;

        let mut positions = Vec::with_capacity(actual_n * 3);
        let velocities = vec![0.0f64; actual_n * 3];
        let mut signs = Vec::with_capacity(actual_n);

        for iz in 0..grid_side {
            for iy in 0..grid_side {
                for ix in 0..grid_side {
                    let idx = iz * grid_side * grid_side + iy * grid_side + ix;
                    let x0 = (ix as f64 + 0.5) * spacing - half_box;
                    let y0 = (iy as f64 + 0.5) * spacing - half_box;
                    let z0 = (iz as f64 + 0.5) * spacing - half_box;

                    let x = x0 + amplitude * (kx * x0).sin();
                    positions.push(x);
                    positions.push(y0);
                    positions.push(z0);
                    signs.push(if idx < n_positive { 1 } else { -1 });
                }
            }
        }

        println!("Grid: {}³ = {} particles", grid_side, actual_n);
        println!("Spacing: {:.1} Mpc", spacing);
        println!("At r=spacing: α(r) = 1 - {}×exp(-{:.1}/{}) = {:.3}",
            epsilon, spacing, r_c, 1.0 - epsilon * (-spacing / r_c).exp());
        let delta_k0 = compute_delta_k(&positions, kx);
        println!("Initial δ_k = {:.6e}\n", delta_k0);

        let n_steps = 50;  // More steps to see evolution

        // ========== N² BRUTE FORCE ==========
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("N² BRUTE FORCE (CPU) - {} steps", n_steps);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Run B: Janus α=1 (N²)
        let mut pos_b_n2 = positions.clone();
        let mut vel_b_n2 = velocities.clone();
        let t0 = Instant::now();
        for _ in 0..n_steps {
            step_n2_janus(&mut pos_b_n2, &mut vel_b_n2, &signs, box_size, dt);
        }
        let time_b_n2 = t0.elapsed().as_millis();
        let delta_k_b_n2 = compute_delta_k(&pos_b_n2, kx);
        println!("Run B (Janus α=1) N²:  δ_k = {:.6e}  ({} ms)", delta_k_b_n2, time_b_n2);

        // Run E: Yukawa ε=0.7, rc=40 (N²)
        let mut pos_e_n2 = positions.clone();
        let mut vel_e_n2 = velocities.clone();
        let t0 = Instant::now();
        for _ in 0..n_steps {
            step_n2_yukawa(&mut pos_e_n2, &mut vel_e_n2, &signs, box_size, dt, epsilon, r_c);
        }
        let time_e_n2 = t0.elapsed().as_millis();
        let delta_k_e_n2 = compute_delta_k(&pos_e_n2, kx);
        println!("Run E (Yukawa) N²:     δ_k = {:.6e}  ({} ms)", delta_k_e_n2, time_e_n2);

        let growth_b_n2 = (delta_k_b_n2 / delta_k0 - 1.0) * 100.0;
        let growth_e_n2 = (delta_k_e_n2 / delta_k0 - 1.0) * 100.0;
        let diff_n2 = (delta_k_e_n2 - delta_k_b_n2) / delta_k_b_n2 * 100.0;
        println!("\nN² growth B: {:+.2}%,  E: {:+.2}%,  diff: {:+.4}%", growth_b_n2, growth_e_n2, diff_n2);

        // ========== BARNES-HUT ==========
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("BARNES-HUT GPU (θ={}) - {} steps", theta, n_steps);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Run B: Janus α=1 (BH)
        let mut sim_b = GpuNBodySimulation::new_with_state(
            n_positive, n_negative, box_size,
            positions.clone(), velocities.clone(), signs.clone()
        ).unwrap();
        sim_b.set_theta(theta);
        let t0 = Instant::now();
        for _ in 0..n_steps {
            sim_b.step_with_cross_factor(dt, -1.0).unwrap();
        }
        let time_b_bh = t0.elapsed().as_millis();
        let pos_b_bh = sim_b.get_positions().unwrap();
        let delta_k_b_bh = compute_delta_k(&pos_b_bh, kx);
        println!("Run B (Janus α=1) BH:  δ_k = {:.6e}  ({} ms)", delta_k_b_bh, time_b_bh);

        // Run E: Yukawa ε=0.7, rc=40 (BH)
        let mut sim_e = GpuNBodySimulation::new_with_state(
            n_positive, n_negative, box_size,
            positions.clone(), velocities.clone(), signs.clone()
        ).unwrap();
        sim_e.set_theta(theta);
        let t0 = Instant::now();
        for _ in 0..n_steps {
            sim_e.step_with_yukawa(dt, epsilon, r_c).unwrap();
        }
        let time_e_bh = t0.elapsed().as_millis();
        let pos_e_bh = sim_e.get_positions().unwrap();
        let delta_k_e_bh = compute_delta_k(&pos_e_bh, kx);
        println!("Run E (Yukawa) BH:     δ_k = {:.6e}  ({} ms)", delta_k_e_bh, time_e_bh);

        let growth_b_bh = (delta_k_b_bh / delta_k0 - 1.0) * 100.0;
        let growth_e_bh = (delta_k_e_bh / delta_k0 - 1.0) * 100.0;
        let diff_bh = (delta_k_e_bh - delta_k_b_bh) / delta_k_b_bh * 100.0;
        println!("\nBH growth B: {:+.2}%,  E: {:+.2}%,  diff: {:+.4}%", growth_b_bh, growth_e_bh, diff_bh);

        // ========== COMPARISON ==========
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("COMPARISON N² vs BH");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let err_b = (delta_k_b_bh - delta_k_b_n2) / delta_k_b_n2 * 100.0;
        let err_e = (delta_k_e_bh - delta_k_e_n2) / delta_k_e_n2 * 100.0;

        println!("Run B: N²={:.6e}, BH={:.6e}, error={:+.2}%", delta_k_b_n2, delta_k_b_bh, err_b);
        println!("Run E: N²={:.6e}, BH={:.6e}, error={:+.2}%", delta_k_e_n2, delta_k_e_bh, err_e);

        println!("\n┌─────────────────────────────────────────────────────────────┐");
        if diff_n2.abs() > 0.1 {
            println!("│ ✓ N² shows Yukawa effect: E differs from B by {:+.2}%       │", diff_n2);
            if diff_bh.abs() < diff_n2.abs() * 0.5 {
                println!("│ ✗ BH does NOT show this effect → BUG in tree code!         │");
            } else {
                println!("│ ✓ BH also shows effect → Tree approximation OK             │");
            }
        } else {
            println!("│ ✗ N² shows NO Yukawa effect (E ≈ B)                         │");
            println!("│ → Physics doesn't respond in this regime                    │");
            println!("│ → Yukawa screening affects all pairs equally                │");
            println!("│ → On uniform grid, no distance variation → no effect        │");
        }
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("ERROR: Requires CUDA. Compile with --features cuda");
}

/// N² brute force with Janus α=1 (CPU)
fn step_n2_janus(pos: &mut [f64], vel: &mut [f64], signs: &[i32], box_size: f64, dt: f64) {
    let n = signs.len();
    let half_dt = dt * 0.5;
    let box_half = box_size / 2.0;
    let softening = box_size / (n as f64).powf(1.0/3.0) * 0.5;
    let eps2 = softening * softening;

    // Drift dt/2
    for i in 0..n {
        for d in 0..3 {
            pos[i*3+d] += vel[i*3+d] * half_dt;
            // Periodic BC
            if pos[i*3+d] > box_half { pos[i*3+d] -= box_size; }
            if pos[i*3+d] < -box_half { pos[i*3+d] += box_size; }
        }
    }

    // Compute accelerations (N²)
    let mut acc = vec![0.0f64; n * 3];
    for i in 0..n {
        let si = signs[i];
        for j in 0..n {
            if i == j { continue; }
            let sj = signs[j];

            let dx = pos[j*3] - pos[i*3];
            let dy = pos[j*3+1] - pos[i*3+1];
            let dz = pos[j*3+2] - pos[i*3+2];

            let r2 = dx*dx + dy*dy + dz*dz + eps2;
            let inv_r3 = 1.0 / (r2 * r2.sqrt());

            // Janus α=1: same sign → attraction (+1), opposite sign → repulsion (-1)
            let interaction = if si == sj { 1.0 } else { -1.0 };
            let f = interaction * inv_r3;

            acc[i*3] += f * dx;
            acc[i*3+1] += f * dy;
            acc[i*3+2] += f * dz;
        }
    }

    // Kick dt
    for i in 0..n {
        for d in 0..3 {
            vel[i*3+d] += acc[i*3+d] * dt;
        }
    }

    // Drift dt/2
    for i in 0..n {
        for d in 0..3 {
            pos[i*3+d] += vel[i*3+d] * half_dt;
            if pos[i*3+d] > box_half { pos[i*3+d] -= box_size; }
            if pos[i*3+d] < -box_half { pos[i*3+d] += box_size; }
        }
    }
}

/// N² brute force with Yukawa screening (CPU)
fn step_n2_yukawa(pos: &mut [f64], vel: &mut [f64], signs: &[i32], box_size: f64, dt: f64, epsilon: f64, r_c: f64) {
    let n = signs.len();
    let half_dt = dt * 0.5;
    let box_half = box_size / 2.0;
    let softening = box_size / (n as f64).powf(1.0/3.0) * 0.5;
    let eps2 = softening * softening;

    // Drift dt/2
    for i in 0..n {
        for d in 0..3 {
            pos[i*3+d] += vel[i*3+d] * half_dt;
            if pos[i*3+d] > box_half { pos[i*3+d] -= box_size; }
            if pos[i*3+d] < -box_half { pos[i*3+d] += box_size; }
        }
    }

    // Compute accelerations (N²) with Yukawa
    let mut acc = vec![0.0f64; n * 3];
    for i in 0..n {
        let si = signs[i];
        for j in 0..n {
            if i == j { continue; }
            let sj = signs[j];

            let dx = pos[j*3] - pos[i*3];
            let dy = pos[j*3+1] - pos[i*3+1];
            let dz = pos[j*3+2] - pos[i*3+2];

            let r2 = dx*dx + dy*dy + dz*dz + eps2;
            let r = r2.sqrt();
            let inv_r3 = 1.0 / (r2 * r);

            // Yukawa: α(r) = 1 - ε×exp(-r/r_c)
            let interaction = if si == sj {
                1.0  // Same sign: attraction
            } else {
                // Opposite sign: -α(r)
                let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
                -alpha_r
            };
            let f = interaction * inv_r3;

            acc[i*3] += f * dx;
            acc[i*3+1] += f * dy;
            acc[i*3+2] += f * dz;
        }
    }

    // Kick dt
    for i in 0..n {
        for d in 0..3 {
            vel[i*3+d] += acc[i*3+d] * dt;
        }
    }

    // Drift dt/2
    for i in 0..n {
        for d in 0..3 {
            pos[i*3+d] += vel[i*3+d] * half_dt;
            if pos[i*3+d] > box_half { pos[i*3+d] -= box_size; }
            if pos[i*3+d] < -box_half { pos[i*3+d] += box_size; }
        }
    }
}

fn compute_delta_k(pos: &[f64], kx: f64) -> f64 {
    let n = pos.len() / 3;
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    for i in 0..n {
        let x = pos[i * 3];
        let phase = kx * x;
        sum_cos += phase.cos();
        sum_sin += phase.sin();
    }
    (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n as f64
}
