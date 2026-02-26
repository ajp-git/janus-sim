/// Janus Segregation Simulation from z=1000
///
/// Answers: At what redshift z does segregation S(z) exceed 4.3%?
/// If z > 2.59 → the H² < 0 "orphan zone" is never physically realized.
///
/// Uses full N-body Barnes-Hut with Janus force laws (same sign attract, opposite repel)
/// and cosmological Hubble damping.

use janus::nbody::{Vec3, Particle, BoundingBox, Octree, NBodySimulation};
use janus::MassSign;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;

// ============================================================
// PHYSICAL PARAMETERS
// ============================================================

const ETA: f64 = 1.045;
const OMEGA_PLUS: f64 = 1.0 / (1.0 + ETA);  // ≈ 0.489
const H0: f64 = 76.0;  // km/s/Mpc (normalized to 1 in code units)

const N_TOTAL: usize = 50_000;
const N_PLUS: usize = N_TOTAL / 2;
const N_MINUS: usize = N_TOTAL / 2;
const BOX_SIZE: f64 = 200.0;  // Mpc

const Z_INIT: f64 = 1000.0;
const Z_END: f64 = 0.0;
const S_CRITICAL: f64 = 0.043;  // 4.3% threshold

const THETA: f64 = 0.7;  // Barnes-Hut opening angle
const N_CELLS: usize = 8;  // 8³ = 512 cells for segregation measure

// ============================================================
// HUBBLE FUNCTION
// ============================================================

/// H(z)/H₀ for Janus cosmology
/// z > 1062: Milne regime H = H₀(1+z)
/// z ≤ 1062: Matter-dominated H = H₀√(Ω₊(1+z)³ + (1-Ω₊))
fn h_over_h0(z: f64) -> f64 {
    if z > 1062.0 {
        1.0 + z  // Milne VSL regime
    } else {
        (OMEGA_PLUS * (1.0 + z).powi(3) + (1.0 - OMEGA_PLUS)).sqrt()
    }
}

// ============================================================
// SEGREGATION MEASURE
// ============================================================

/// COM-based segregation: S = |COM+ - COM-| / (L/2)
/// S = 0 for perfect mixing (COMs coincide)
/// S → 1 for complete segregation (COMs at maximum distance)
fn measure_segregation(particles: &[Particle], box_size: f64) -> f64 {
    let half_box = box_size / 2.0;

    // Compute COM for each sector using minimum image convention
    let mut sum_plus = Vec3::zero();
    let mut n_plus = 0usize;
    let mut sum_minus = Vec3::zero();
    let mut n_minus = 0usize;

    // Use first particle as reference for periodic unwrapping
    let ref_pos = if !particles.is_empty() {
        particles[0].pos
    } else {
        return 0.0;
    };

    for p in particles {
        // Minimum image relative to reference
        let dx = p.pos.x - ref_pos.x;
        let dy = p.pos.y - ref_pos.y;
        let dz = p.pos.z - ref_pos.z;

        let unwrapped = Vec3::new(
            ref_pos.x + minimum_image(dx, box_size),
            ref_pos.y + minimum_image(dy, box_size),
            ref_pos.z + minimum_image(dz, box_size),
        );

        match p.sign {
            MassSign::Positive => {
                sum_plus.x += unwrapped.x;
                sum_plus.y += unwrapped.y;
                sum_plus.z += unwrapped.z;
                n_plus += 1;
            }
            MassSign::Negative => {
                sum_minus.x += unwrapped.x;
                sum_minus.y += unwrapped.y;
                sum_minus.z += unwrapped.z;
                n_minus += 1;
            }
        }
    }

    if n_plus == 0 || n_minus == 0 {
        return 0.0;
    }

    let com_plus = Vec3::new(
        sum_plus.x / n_plus as f64,
        sum_plus.y / n_plus as f64,
        sum_plus.z / n_plus as f64,
    );
    let com_minus = Vec3::new(
        sum_minus.x / n_minus as f64,
        sum_minus.y / n_minus as f64,
        sum_minus.z / n_minus as f64,
    );

    // Distance between COMs with minimum image
    let dx = minimum_image(com_plus.x - com_minus.x, box_size);
    let dy = minimum_image(com_plus.y - com_minus.y, box_size);
    let dz = minimum_image(com_plus.z - com_minus.z, box_size);

    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    // Normalize by L/2 (max meaningful separation)
    (dist / half_box).min(1.0)
}

/// Minimum image convention for periodic boundaries
fn minimum_image(dx: f64, box_size: f64) -> f64 {
    let half = box_size / 2.0;
    if dx > half {
        dx - box_size
    } else if dx < -half {
        dx + box_size
    } else {
        dx
    }
}

// ============================================================
// INITIAL CONDITIONS
// ============================================================

fn create_particles(z: f64) -> (Vec<Particle>, f64) {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let m_total = OMEGA_PLUS * BOX_SIZE.powi(3);
    let m_particle = m_total / N_TOTAL as f64;
    let v_thermal = 1e-3 * BOX_SIZE / (1.0 + z);

    // Grille de perturbations 8x8x8
    const NC: usize = 8;
    let cell_size = BOX_SIZE / NC as f64;

    // Champ de densité avec spectre HZ : P(k) ∝ k
    // On génère des modes de grande longueur d'onde
    // δ(x) = Σ_k A(k) cos(k·x + φ_k)  avec A(k) ∝ sqrt(k)
    let mut delta = vec![0.0f64; NC * NC * NC];

    // Modes fondamentaux (grandes échelles)
    let n_modes = 4;  // premiers harmoniques seulement
    for nx in 0..=n_modes {
        for ny in 0..=n_modes {
            for nz in 0..=n_modes {
                if nx == 0 && ny == 0 && nz == 0 { continue; }
                let k2 = (nx*nx + ny*ny + nz*nz) as f64;
                let k = k2.sqrt();
                // Amplitude HZ : P(k) ∝ k → A ∝ sqrt(k)
                let amplitude = 0.3 * k.sqrt() / (n_modes as f64 * 2.0);
                let phase: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;

                for ix in 0..NC {
                    for iy in 0..NC {
                        for iz in 0..NC {
                            let kx = 2.0 * std::f64::consts::PI * nx as f64 * ix as f64 / NC as f64;
                            let ky = 2.0 * std::f64::consts::PI * ny as f64 * iy as f64 / NC as f64;
                            let kz = 2.0 * std::f64::consts::PI * nz as f64 * iz as f64 / NC as f64;
                            delta[ix * NC * NC + iy * NC + iz] +=
                                amplitude * (kx + ky + kz + phase).cos();
                        }
                    }
                }
            }
        }
    }

    // Normaliser δ à [-1, 1]
    let d_max = delta.iter().cloned().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-10);
    for d in delta.iter_mut() { *d /= d_max; }

    // Brisure de symétrie initiale : ε = fraction de déséquilibre
    // Dans les cellules à δ > 0 : fraction_plus = 0.5 + ε*δ/2
    // Dans les cellules à δ < 0 : fraction_minus = 0.5 - ε*δ/2
    let epsilon = 0.1;  // 10% de déséquilibre initial

    let mut particles = Vec::with_capacity(N_TOTAL);
    let half_box = BOX_SIZE / 2.0;

    // Nombre de particules par cellule
    let n_per_cell = N_TOTAL / (NC * NC * NC);
    let n_per_cell = n_per_cell.max(1);

    for ix in 0..NC {
        for iy in 0..NC {
            for iz in 0..NC {
                let d = delta[ix * NC * NC + iy * NC + iz];
                // fraction_plus dans cette cellule
                let frac_plus = (0.5 + epsilon * d / 2.0).clamp(0.1, 0.9);
                let n_plus_cell = (n_per_cell as f64 * frac_plus) as usize;
                let n_minus_cell = n_per_cell - n_plus_cell;

                let cx = -half_box + (ix as f64 + 0.5) * cell_size;
                let cy = -half_box + (iy as f64 + 0.5) * cell_size;
                let cz = -half_box + (iz as f64 + 0.5) * cell_size;

                for &(n, sign) in &[(n_plus_cell, MassSign::Positive),
                                     (n_minus_cell, MassSign::Negative)] {
                    for _ in 0..n {
                        let pos = Vec3::new(
                            cx + (rng.random::<f64>() - 0.5) * cell_size,
                            cy + (rng.random::<f64>() - 0.5) * cell_size,
                            cz + (rng.random::<f64>() - 0.5) * cell_size,
                        );
                        let vel = Vec3::new(
                            (rng.random::<f64>() - 0.5) * v_thermal,
                            (rng.random::<f64>() - 0.5) * v_thermal,
                            (rng.random::<f64>() - 0.5) * v_thermal,
                        );
                        particles.push(Particle::new(pos, vel, m_particle, sign));
                    }
                }
            }
        }
    }

    // Compléter si N pas exactement atteint (arrondi)
    while particles.len() < N_TOTAL {
        let sign = if rng.random::<bool>() { MassSign::Positive } else { MassSign::Negative };
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * BOX_SIZE,
            (rng.random::<f64>() - 0.5) * BOX_SIZE,
            (rng.random::<f64>() - 0.5) * BOX_SIZE,
        );
        particles.push(Particle::new(pos, Vec3::zero(), m_particle, sign));
    }
    particles.truncate(N_TOTAL);

    println!("  Particules initialisées avec spectre HZ (ε={:.2})", epsilon);
    println!("  δ_max = {:.3}, brisure symétrie initiale ≈ {:.1}%",
             1.0, epsilon * 100.0);

    (particles, m_particle)
}

// ============================================================
// MAIN SIMULATION
// ============================================================

fn main() {
    println!("════════════════════════════════════════════════════════════");
    println!("JANUS SEGREGATION — z=1000 → z=0");
    println!("════════════════════════════════════════════════════════════");
    println!("N = {} particles ({} + / {} -)", N_TOTAL, N_PLUS, N_MINUS);
    println!("Box = {} Mpc, η = {}", BOX_SIZE, ETA);
    println!("Ω₊ = {:.4}, Critical threshold S = {:.1}%", OMEGA_PLUS, S_CRITICAL * 100.0);
    println!("════════════════════════════════════════════════════════════\n");

    // Initialize
    let (mut particles, m_particle) = create_particles(Z_INIT);
    let mut z = Z_INIT;
    let mut a = 1.0 / (1.0 + z);

    println!("Particle mass = {:.4e} (M_total = {:.4e})", m_particle, m_particle * N_TOTAL as f64);

    // Simulation parameters
    let half_box = BOX_SIZE / 2.0;
    let bounds = BoundingBox::new(
        Vec3::new(-half_box, -half_box, -half_box),
        Vec3::new(half_box, half_box, half_box),
    );

    // Softening: ε ~ 0.5 * L / N^(1/3)
    let mean_sep = BOX_SIZE / (N_TOTAL as f64).powf(1.0 / 3.0);
    let softening = 0.5 * mean_sep;

    // Results storage
    let mut results: Vec<(usize, f64, f64, bool)> = Vec::new();  // (step, z, S, H²_ok)
    let mut z_threshold: Option<f64> = None;
    let mut step = 0;

    // CSV file
    let mut csv_file = File::create("segregation_sz.csv").expect("Cannot create CSV");
    writeln!(csv_file, "step,z,S,H2_positive").expect("Cannot write header");

    println!("  {:>6}  {:>8}  {:>8}  {:>7}", "step", "z", "S", "H²>0?");
    println!("  {}", "-".repeat(35));

    let start_time = std::time::Instant::now();

    // Main loop: integrate from z=1000 to z=0
    while z > Z_END && a < 1.0 {
        // Adaptive time step: dt = 0.005 / H(z), capped at 0.001
        let h_z = h_over_h0(z);
        let dt = (0.005 / h_z).min(0.001);

        // Build tree
        let tree = Octree::build(&particles, bounds, THETA);

        // Compute accelerations (parallel)
        let accelerations: Vec<Vec3> = particles.par_iter()
            .map(|p| tree.compute_acceleration(p.pos, p.sign, &particles, softening))
            .collect();

        // Leapfrog with Hubble damping
        // dv/dτ = a_comobile - 2H·v (Hubble friction)
        // dx/dτ = v
        let hubble_damp = 2.0 * h_z * dt;

        for (p, acc) in particles.iter_mut().zip(accelerations.iter()) {
            // Kick: v += (a - 2H·v)·dt
            p.vel.x = p.vel.x * (1.0 - hubble_damp) + acc.x * dt;
            p.vel.y = p.vel.y * (1.0 - hubble_damp) + acc.y * dt;
            p.vel.z = p.vel.z * (1.0 - hubble_damp) + acc.z * dt;

            // Drift: x += v·dt / a (comoving)
            p.pos.x += p.vel.x * dt / a;
            p.pos.y += p.vel.y * dt / a;
            p.pos.z += p.vel.z * dt / a;

            // Periodic boundary conditions
            if p.pos.x > half_box { p.pos.x -= BOX_SIZE; }
            if p.pos.x < -half_box { p.pos.x += BOX_SIZE; }
            if p.pos.y > half_box { p.pos.y -= BOX_SIZE; }
            if p.pos.y < -half_box { p.pos.y += BOX_SIZE; }
            if p.pos.z > half_box { p.pos.z -= BOX_SIZE; }
            if p.pos.z < -half_box { p.pos.z += BOX_SIZE; }
        }

        // Update scale factor: da/dτ = a·H → a_new = a·(1 + H·dt)
        a *= 1.0 + h_z * dt;
        z = 1.0 / a - 1.0;
        step += 1;

        // Measure every 20 steps
        if step % 20 == 0 {
            let s = measure_segregation(&particles, BOX_SIZE);

            // Check if H²_eff > 0
            // H²_eff = Ω₊(1+z)³ - (1-S)·Ω₋(1+z)³ + ...
            // Simplified: H² > 0 iff S > (Ω₋ - Ω₊) / Ω₋ = (η-1)/(η+1)
            // For η=1.045: threshold ≈ 0.022 → 2.2%
            // But the stated threshold is 4.3% from theory
            let h2_ok = s > S_CRITICAL || z < 2.59;
            let h2_mark = if h2_ok { "✓" } else { "✗" };

            let elapsed = start_time.elapsed().as_secs_f64();
            println!("  {:>6}  {:>8.2}  {:>8.4}  {:>7}  [{:.1}s]", step, z, s, h2_mark, elapsed);

            // Save to CSV
            writeln!(csv_file, "{},{:.6},{:.6},{}", step, z, s, h2_ok).ok();
            results.push((step, z, s, h2_ok));

            // Check threshold
            if z_threshold.is_none() && s > S_CRITICAL {
                z_threshold = Some(z);
                println!("\n  *** SEUIL S={:.1}% ATTEINT À z = {:.2} ***\n", S_CRITICAL * 100.0, z);
            }
        }

        // Safety: stop if z goes negative or NaN
        if z.is_nan() || z < -0.1 {
            println!("  WARNING: z became invalid ({}), stopping", z);
            break;
        }

        // Progress indicator every 1000 steps
        if step % 1000 == 0 && step % 20 != 0 {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();

    // ============================================================
    // RESULTS
    // ============================================================

    println!("\n════════════════════════════════════════════════════════════");
    println!("RÉSULTAT");
    println!("════════════════════════════════════════════════════════════");
    println!("  Total steps: {}", step);
    println!("  Final z: {:.4}", z);
    println!("  Runtime: {:.1}s", total_time);

    if let Some(z_thresh) = z_threshold {
        println!("\n  S > {:.1}% atteint à z = {:.2}", S_CRITICAL * 100.0, z_thresh);
        if z_thresh > 2.59 {
            println!("  → H²_eff > 0 AVANT z_cross = 2.59  ✓");
            println!("  → Zone orpheline résolue ✓");
        } else {
            println!("  → Seuil atteint trop tard (z < 2.59)");
            println!("  → Zone orpheline NON résolue ✗");
        }
    } else {
        println!("\n  Seuil S={:.1}% jamais atteint", S_CRITICAL * 100.0);

        // Find max S
        let max_s = results.iter().map(|(_, _, s, _)| *s).fold(0.0_f64, f64::max);
        println!("  S max = {:.4} ({:.2}%)", max_s, max_s * 100.0);
        println!("  → Ségrégation insuffisante dans ce modèle");
    }

    println!("\n  Résultats sauvegardés dans segregation_sz.csv");
    println!("════════════════════════════════════════════════════════════");
}
