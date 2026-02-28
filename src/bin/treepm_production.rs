//! TreePM Production Run with Virialized ICs
//!
//! Generates frames for visual validation of TreePM physics.
//! Uses PE_binding virialization (same-sign pairs only).

use janus::treepm::treepm_force::TreePMForce;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Compute PE_binding for TreePM: only pairs with r < r_cut
/// This is coherent with what TreePM actually computes
fn compute_pe_binding_treepm(particles: &[Particle], r_cut: f64, g_constant: f64, softening: f64) -> f64 {
    let r_cut_sq = r_cut * r_cut;
    let soft_sq = softening * softening;
    let mut pe_binding = 0.0;

    // O(N²) but only for initialization - acceptable
    for i in 0..particles.len() {
        for j in (i+1)..particles.len() {
            let pi = &particles[i];
            let pj = &particles[j];

            // Only same-sign pairs contribute to PE_binding
            if pi.sign != pj.sign {
                continue;
            }

            let dx = pj.pos.x - pi.pos.x;
            let dy = pj.pos.y - pi.pos.y;
            let dz = pj.pos.z - pi.pos.z;
            let r_sq = dx*dx + dy*dy + dz*dz;

            // Only pairs within r_cut (TreePM short-range)
            if r_sq >= r_cut_sq {
                continue;
            }

            // Plummer-softened potential
            let r_soft = (r_sq + soft_sq).sqrt();
            // Same-sign attraction: PE < 0
            pe_binding -= g_constant * pi.mass * pj.mass / r_soft;
        }
    }

    pe_binding
}

/// Generate particles with TreePM-coherent virialization
/// PE_binding only includes pairs within r_cut
fn generate_virialized_particles_treepm(
    n: usize, box_size: f64, eta: f64, seed: u64,
    r_cut: f64, g_constant: f64, softening: f64
) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Use random sign assignment to avoid artificial segregation
    let prob_pos = 1.0 / (1.0 + eta);

    println!("Generating {} particles (prob_pos={:.3})", n, prob_pos);

    let mut particles = Vec::with_capacity(n);
    let mut n_pos: usize = 0;
    let mut n_neg: usize = 0;

    // Initial velocity scale
    let v_init = 0.1;

    for _ in 0..n {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        let vel = Vec3::new(
            (rng.random::<f64>() - 0.5) * v_init,
            (rng.random::<f64>() - 0.5) * v_init,
            (rng.random::<f64>() - 0.5) * v_init,
        );

        let sign = if rng.random::<f64>() < prob_pos {
            n_pos += 1;
            MassSign::Positive
        } else {
            n_neg += 1;
            MassSign::Negative
        };

        particles.push(Particle::new(pos, vel, 1.0, sign));
    }

    println!("  Generated {}+ / {}- particles", n_pos, n_neg);

    // Compute PE_binding with TreePM-coherent r_cut
    println!("  Computing PE_binding (r < r_cut = {:.2})...", r_cut);
    let pe_binding = compute_pe_binding_treepm(&particles, r_cut, g_constant, softening);

    // Current KE
    let ke_current: f64 = particles.iter()
        .map(|p| 0.5 * (p.vel.x*p.vel.x + p.vel.y*p.vel.y + p.vel.z*p.vel.z))
        .sum();

    // Virial: 2*KE + PE_binding = 0 → KE_target = |PE_binding| / 2
    let ke_target = pe_binding.abs() / 2.0;
    let alpha = if ke_current > 1e-20 { (ke_target / ke_current).sqrt() } else { 1.0 };

    println!("TreePM-coherent virialization:");
    println!("  PE_binding (r<r_cut) = {:.4e}", pe_binding);
    println!("  KE_initial = {:.4e}", ke_current);
    println!("  KE_target  = {:.4e}", ke_target);
    println!("  alpha      = {:.6}", alpha);

    // Safety check: α should be reasonable (< 100)
    if alpha > 100.0 {
        println!("  ⚠ WARNING: alpha > 100, system may be unstable");
    }

    // Scale velocities
    for p in particles.iter_mut() {
        p.vel.x *= alpha;
        p.vel.y *= alpha;
        p.vel.z *= alpha;
    }

    // Verify
    let ke_after: f64 = particles.iter()
        .map(|p| 0.5 * (p.vel.x*p.vel.x + p.vel.y*p.vel.y + p.vel.z*p.vel.z))
        .sum();
    let virial_error = (2.0 * ke_after + pe_binding).abs() / pe_binding.abs();
    println!("  Virial error: {:.4}%", virial_error * 100.0);

    particles
}

fn compute_kinetic_energy(particles: &[Particle]) -> f64 {
    particles.iter()
        .map(|p| 0.5 * (p.vel.x*p.vel.x + p.vel.y*p.vel.y + p.vel.z*p.vel.z))
        .sum()
}

fn compute_segregation(particles: &[Particle]) -> f64 {
    let (mut com_pos, mut n_pos) = (Vec3::zero(), 0.0);
    let (mut com_neg, mut n_neg) = (Vec3::zero(), 0.0);

    for p in particles {
        match p.sign {
            MassSign::Positive => {
                com_pos.x += p.pos.x;
                com_pos.y += p.pos.y;
                com_pos.z += p.pos.z;
                n_pos += 1.0;
            }
            MassSign::Negative => {
                com_neg.x += p.pos.x;
                com_neg.y += p.pos.y;
                com_neg.z += p.pos.z;
                n_neg += 1.0;
            }
        }
    }

    if n_pos > 0.0 {
        com_pos.x /= n_pos;
        com_pos.y /= n_pos;
        com_pos.z /= n_pos;
    }
    if n_neg > 0.0 {
        com_neg.x /= n_neg;
        com_neg.y /= n_neg;
        com_neg.z /= n_neg;
    }

    let dx = com_pos.x - com_neg.x;
    let dy = com_pos.y - com_neg.y;
    let dz = com_pos.z - com_neg.z;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

/// Save snapshot in binary format for rendering
fn save_snapshot(particles: &[Particle], path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header: N (u32)
    let n = particles.len() as u32;
    file.write_all(&n.to_le_bytes())?;

    // For each particle: x, y, z (f32), sign (i8)
    for p in particles {
        file.write_all(&(p.pos.x as f32).to_le_bytes())?;
        file.write_all(&(p.pos.y as f32).to_le_bytes())?;
        file.write_all(&(p.pos.z as f32).to_le_bytes())?;
        let sign: i8 = match p.sign { MassSign::Positive => 1, MassSign::Negative => -1 };
        file.write_all(&sign.to_le_bytes())?;
    }

    Ok(())
}

fn main() {
    println!("=== TreePM Production Run ===\n");

    // Parameters matching previous Janus runs
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);  // Default 100K for quick test

    let box_size = 100.0;
    let grid_size = 128;  // 128³ for better PM resolution
    let r_cut = box_size / 16.0;
    let dt = 0.01;
    let n_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let softening = 0.5;
    let eta = 1.045;
    let theta = 0.5;

    let output_dir = format!("/app/output/treepm_{}", n);
    fs::create_dir_all(&output_dir).ok();
    fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

    println!("Configuration:");
    println!("  N particles: {}", n);
    println!("  Box size: {}", box_size);
    println!("  Grid: {}³", grid_size);
    println!("  r_cut: {:.2}", r_cut);
    println!("  dt: {}", dt);
    println!("  Steps: {}", n_steps);
    println!("  η: {}", eta);
    println!("  θ: {}", theta);
    println!("  Output: {}", output_dir);
    println!();

    // G constant
    let g_constant = 1.0;
    println!("  G_constant: {}", g_constant);

    // Generate virialized particles with TreePM-coherent PE_binding
    // Only pairs within r_cut contribute to PE_binding (same as TreePM forces)
    let mut particles = generate_virialized_particles_treepm(
        n, box_size, eta, 42, r_cut, g_constant, softening
    );

    // Initialize TreePM
    // For fast visual validation, use PM-only (skip Tree short-range)
    let pm_only = std::env::var("PM_ONLY").is_ok();
    let mut treepm = if pm_only {
        println!("  MODE: PM-only (Tree short-range disabled)");
        let mut t = TreePMForce::new_pm_only(grid_size, box_size);
        t.g_constant = g_constant;
        t
    } else {
        println!("  MODE: Full TreePM (PM + Tree)");
        let mut t = TreePMForce::new(r_cut, grid_size, box_size, theta, softening);
        t.g_constant = g_constant;
        t
    };

    // Initial metrics
    let ke_0 = compute_kinetic_energy(&particles);
    let seg_0 = compute_segregation(&particles);

    println!("\nInitial state:");
    println!("  KE₀ = {:.6e}", ke_0);
    println!("  Seg₀ = {:.6}", seg_0);
    println!();

    // Save initial snapshot
    let snap_path = Path::new(&output_dir).join("snapshots/snap_00000.bin");
    save_snapshot(&particles, &snap_path).ok();

    // Open time series CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Cannot create CSV");
    writeln!(csv, "step,ke,ke_ratio,seg,seg_ratio").ok();
    writeln!(csv, "0,{:.6e},1.0,{:.6},{:.6}", ke_0, seg_0, seg_0 / box_size).ok();

    // Run simulation
    let start = Instant::now();
    let mut last_log = Instant::now();

    for step in 1..=n_steps {
        // Update TreePM
        treepm.update(&particles);

        // Compute forces
        let forces = treepm.compute_all_forces(&particles);

        // Leapfrog kick (half)
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += 0.5 * dt * f.x;
            p.vel.y += 0.5 * dt * f.y;
            p.vel.z += 0.5 * dt * f.z;
        }

        // Drift
        for p in &mut particles {
            p.pos.x += dt * p.vel.x;
            p.pos.y += dt * p.vel.y;
            p.pos.z += dt * p.vel.z;
        }

        // Update TreePM again
        treepm.update(&particles);
        let forces = treepm.compute_all_forces(&particles);

        // Leapfrog kick (half)
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += 0.5 * dt * f.x;
            p.vel.y += 0.5 * dt * f.y;
            p.vel.z += 0.5 * dt * f.z;
        }

        // Compute metrics
        let ke = compute_kinetic_energy(&particles);
        let seg = compute_segregation(&particles);
        let ke_ratio = ke / ke_0;
        let seg_ratio = seg / box_size;

        // Log to CSV
        writeln!(csv, "{},{:.6e},{:.6},{:.6},{:.6}", step, ke, ke_ratio, seg, seg_ratio).ok();

        // Save snapshot every 100 steps
        if step % 100 == 0 {
            let snap_path = Path::new(&output_dir).join(format!("snapshots/snap_{:05}.bin", step));
            save_snapshot(&particles, &snap_path).ok();
        }

        // Progress log every 10 steps or 5 seconds
        if step % 10 == 0 || last_log.elapsed().as_secs() >= 5 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let eta_sec = (n_steps - step) as f64 / rate;

            // For cold start, show absolute KE instead of ratio
            if ke_0 < 1e-10 {
                println!("Step {:>5}/{}: KE={:.2e}, Seg={:.4}, ΔSeg={:+.2}%, {:.1}s elapsed",
                         step, n_steps, ke, seg, (seg - seg_0) / seg_0 * 100.0, elapsed);
            } else {
                println!("Step {:>5}/{}: KE/KE₀={:.3}, Seg={:.4}, {:.1}s elapsed, ETA {:.0}s",
                         step, n_steps, ke_ratio, seg, elapsed, eta_sec);
            }
            last_log = Instant::now();
        }

        // Auto-stop if KE explodes (skip for cold start where KE₀ = 0)
        if ke_0 > 1e-10 && ke_ratio > 50.0 {
            println!("\n⚠ STOP: KE/KE₀ = {:.1} > 50 — energy instability", ke_ratio);
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Final metrics
    let ke_final = compute_kinetic_energy(&particles);
    let seg_final = compute_segregation(&particles);
    let ke_ratio = ke_final / ke_0;

    println!("\n=== Final Results ===");
    println!("  Steps completed: {}", n_steps);
    println!("  KE/KE₀ = {:.3}", ke_ratio);
    println!("  Seg_final = {:.4}", seg_final);
    println!("  Elapsed: {:.1}s ({:.1}ms/step)", elapsed, elapsed * 1000.0 / n_steps as f64);
    println!("  Output: {}", output_dir);

    // Save final snapshot
    let snap_path = Path::new(&output_dir).join(format!("snapshots/snap_{:05}.bin", n_steps));
    save_snapshot(&particles, &snap_path).ok();

    // Validation summary
    println!("\n=== Validation ===");
    if ke_ratio < 10.0 {
        println!("✓ KE stable (< 10)");
    } else {
        println!("✗ KE unstable (> 10)");
    }
    if seg_final > seg_0 {
        println!("✓ Segregation increased ({:.4} → {:.4})", seg_0, seg_final);
    } else {
        println!("⚠ Segregation decreased ({:.4} → {:.4})", seg_0, seg_final);
    }
}
