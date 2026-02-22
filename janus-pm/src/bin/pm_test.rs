/// PM-3: Poisson Solver + Leapfrog Integration Test
/// Validates:
///   - Energy conservation < 2% over 50 steps
///   - KE/KE₀ < 100
///
/// Standard gravity only (no Janus, no Hubble)

use janus_pm::integrator::{PMSimulation, Particle, generate_uniform_ic};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-3: Poisson Solver + Leapfrog Validation                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Parameters
    let n_particles = 100_000;
    let grid_size = 64;  // 64³ grid for 100K particles
    let box_size = 100.0;
    let dt = 0.01_f32;  // Smaller timestep for stability
    let n_steps = 50;
    let velocity_dispersion = 0.1_f32;  // Lower velocity for stability
    let seed = 42;

    println!("\nParameters:");
    println!("  Particles: {}", n_particles);
    println!("  Grid: {}³", grid_size);
    println!("  Box size: {:.1}", box_size);
    println!("  dt: {:.2}", dt);
    println!("  Steps: {}", n_steps);

    // Generate initial conditions
    println!("\nGenerating initial conditions...");
    let t0 = Instant::now();
    let particles = generate_uniform_ic(n_particles, box_size, velocity_dispersion, seed);

    // Create simulation
    println!("Creating PM simulation...");
    let mut sim = match PMSimulation::new(
        particles,
        grid_size, grid_size, grid_size,
        box_size,
        dt,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("ERROR: Failed to create simulation: {}", e);
            std::process::exit(1);
        }
    };

    println!("Setup time: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Initial energy
    if let Err(e) = sim.compute_forces() {
        println!("ERROR: compute_forces failed: {}", e);
        std::process::exit(1);
    }

    let ke_0 = sim.kinetic_energy();
    let pe_0 = sim.potential_energy();
    let e_0 = ke_0 + pe_0;

    println!("\nInitial state:");
    println!("  KE₀ = {:.6e}", ke_0);
    println!("  PE₀ = {:.6e}", pe_0);
    println!("  E₀  = {:.6e}", e_0);

    // Time evolution
    println!("\n  Step      KE/KE₀      PE/PE₀      E/E₀        ΔE%         Time");
    println!("  ─────────────────────────────────────────────────────────────────");

    let mut max_ke_ratio = 1.0_f64;
    let mut max_energy_error = 0.0_f64;

    let t_loop = Instant::now();
    for step in 1..=n_steps {
        let t_step = Instant::now();

        if let Err(e) = sim.step() {
            println!("ERROR at step {}: {}", step, e);
            std::process::exit(1);
        }

        let ke = sim.kinetic_energy();
        let pe = sim.potential_energy();
        let e = ke + pe;

        let ke_ratio = ke / ke_0;
        let pe_ratio = if pe_0.abs() > 1e-10 { pe / pe_0 } else { 1.0 };
        let e_ratio = if e_0.abs() > 1e-10 { e / e_0 } else { 1.0 };
        let energy_error = ((e - e_0) / e_0.abs()) * 100.0;

        max_ke_ratio = max_ke_ratio.max(ke_ratio);
        max_energy_error = max_energy_error.max(energy_error.abs());

        let step_time = t_step.elapsed().as_secs_f64() * 1000.0;

        if step % 10 == 0 || step == 1 || step == n_steps {
            println!("  {:4}      {:.4}      {:.4}      {:.4}      {:+.2}%      {:.1}ms",
                     step, ke_ratio, pe_ratio, e_ratio, energy_error, step_time);
        }
    }

    let total_time = t_loop.elapsed().as_secs_f64();

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      RESULTS                                     ");
    println!("══════════════════════════════════════════════════════════════════");

    println!("\n  Total simulation time: {:.2} s", total_time);
    println!("  Time per step: {:.1} ms", total_time * 1000.0 / n_steps as f64);
    println!("  Max KE/KE₀: {:.4}", max_ke_ratio);
    println!("  Max |ΔE/E₀|: {:.2}%", max_energy_error);

    // Validation
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      VALIDATION SUMMARY                          ");
    println!("══════════════════════════════════════════════════════════════════");

    let energy_pass = max_energy_error < 2.0;
    let ke_pass = max_ke_ratio < 100.0;

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Test                    │ Result    │ Threshold │ Status       │");
    println!("├─────────────────────────┼───────────┼───────────┼──────────────┤");
    println!("│ Energy conservation     │ {:.2}%     │ < 2%      │ {}           │",
             max_energy_error, if energy_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("│ KE/KE₀ (max)            │ {:.2}      │ < 100     │ {}           │",
             max_ke_ratio, if ke_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n══════════════════════════════════════════════════════════════════");
    if energy_pass && ke_pass {
        println!("PM-3 VALIDATION: ✓ PASSED");
        println!("  Energy conservation: {:.2}% < 2%", max_energy_error);
        println!("  KE ratio: {:.2} < 100", max_ke_ratio);
    } else {
        println!("PM-3 VALIDATION: ✗ FAILED");
        if !energy_pass {
            println!("  ✗ Energy drift {:.2}% >= 2%", max_energy_error);
        }
        if !ke_pass {
            println!("  ✗ KE ratio {:.2} >= 100", max_ke_ratio);
        }
    }
    println!("══════════════════════════════════════════════════════════════════");
}
