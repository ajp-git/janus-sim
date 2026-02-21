/// Janus N-body Simulation
///
/// Demonstrates spatial segregation of positive and negative masses
/// following the Janus interaction rules.
///
/// Usage: cargo run --release --bin nbody -- --n 100000 --steps 100

use janus::nbody::NBodySimulation;
use janus::MassSign;
use std::fs;
use std::io::Write;

fn main() {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let n_particles = parse_arg(&args, "--n", 100_000);
    let n_steps = parse_arg(&args, "--steps", 100);
    let dt = parse_arg(&args, "--dt", 0.01);
    let output_freq = parse_arg(&args, "--output-freq", 10);
    let eta = parse_arg(&args, "--eta", 1.74);  // From Friedmann fit

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus Cosmological Model — N-body Simulation                 ║");
    println!("║   Barnes-Hut O(N log N) + Leapfrog integrator                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Use η to determine ratio of negative to positive masses
    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;

    println!("Parameters:");
    println!("  N particles  = {} ({} positive, {} negative)", n_particles, n_positive, n_negative);
    println!("  η = N₋/N₊    = {:.2}", eta);
    println!("  Steps        = {}", n_steps);
    println!("  dt           = {}", dt);
    println!("  Output freq  = every {} steps", output_freq);

    // Initialize simulation
    println!("\nInitializing simulation...");
    let box_size = 100.0;
    let mut sim = NBodySimulation::new(n_positive, n_negative, box_size);

    println!("  Box size     = {:.1}", box_size);
    println!("  Softening    = {:.4}", sim.softening);
    println!("  θ (opening)  = {:.2}", sim.theta);

    // Create output directory
    fs::create_dir_all("output/nbody").ok();

    // Output initial state
    save_snapshot(&sim, "output/nbody/snapshot_0000.csv");

    let mut time_series = Vec::new();
    let initial_seg = sim.segregation_distance();
    time_series.push((0.0, initial_seg, sim.kinetic_energy()));

    // Compute initial energy (expensive but important for validation)
    println!("\nComputing initial energy...");
    let ke0 = sim.kinetic_energy();
    let initial_ke = ke0;
    println!("  Initial KE = {:.4e}", ke0);

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Running simulation...");
    println!("══════════════════════════════════════════════════════════════════\n");

    println!("{:>8}  {:>12}  {:>12}  {:>14}  {:>12}", "Step", "Time", "Segregation", "KE", "KE/KE0");
    println!("{:-<70}", "");

    let start_time = std::time::Instant::now();

    for step in 1..=n_steps {
        sim.step(dt);

        if step % output_freq == 0 || step == n_steps {
            let seg = sim.segregation_distance();
            let ke = sim.kinetic_energy();
            let ke_ratio = ke / initial_ke;
            time_series.push((sim.time, seg, ke));

            println!("{:>8}  {:>12.4}  {:>12.4}  {:>14.4e}  {:>12.2}", step, sim.time, seg, ke, ke_ratio);

            // Energy stability check
            if ke_ratio > 1000.0 {
                println!("\n⚠️  WARNING: KE has increased by {}x — possible instability!", ke_ratio as u64);
                println!("    Consider reducing dt or increasing softening.");
            }

            let filename = format!("output/nbody/snapshot_{:04}.csv", step);
            save_snapshot(&sim, &filename);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Simulation complete in {:.2}s", elapsed.as_secs_f64());
    println!("══════════════════════════════════════════════════════════════════");

    // Save time series
    let mut ts_file = fs::File::create("output/nbody/time_series.csv").unwrap();
    writeln!(ts_file, "time,segregation,kinetic_energy").unwrap();
    for (t, seg, ke) in &time_series {
        writeln!(ts_file, "{:.6},{:.6},{:.6e}", t, seg, ke).unwrap();
    }
    println!("\n✓ output/nbody/time_series.csv");

    // Compute final statistics
    let (com_plus, com_minus) = sim.centers_of_mass();
    let final_seg = sim.segregation_distance();

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                  ");
    println!("══════════════════════════════════════════════════════════════════");
    println!("\n  Initial segregation: {:.4}", initial_seg);
    println!("  Final segregation:   {:.4}", final_seg);
    println!("  Change:              {:.2}%", (final_seg - initial_seg) / initial_seg * 100.0);
    println!("\n  Center of mass (+): ({:.2}, {:.2}, {:.2})", com_plus.x, com_plus.y, com_plus.z);
    println!("  Center of mass (-): ({:.2}, {:.2}, {:.2})", com_minus.x, com_minus.y, com_minus.z);

    // Final energy diagnostics
    let final_ke = sim.kinetic_energy();
    println!("\n  Energy diagnostics:");
    println!("    Initial KE: {:.4e}", initial_ke);
    println!("    Final KE:   {:.4e}", final_ke);
    println!("    KE ratio:   {:.2}x", final_ke / initial_ke);

    // Summary JSON
    let summary = format!(r#"{{
  "model": "Janus N-body",
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {:.4},
  "n_steps": {},
  "dt": {},
  "box_size": {:.2},
  "softening": {:.6},
  "initial_segregation": {:.6},
  "final_segregation": {:.6},
  "initial_ke": {:.6e},
  "final_ke": {:.6e},
  "ke_ratio": {:.4},
  "elapsed_seconds": {:.2}
}}"#,
        n_particles, n_positive, n_negative, eta, n_steps, dt, box_size,
        sim.softening, initial_seg, final_seg, initial_ke, final_ke,
        final_ke / initial_ke, elapsed.as_secs_f64());

    fs::write("output/nbody/summary.json", &summary).unwrap();
    println!("\n✓ output/nbody/summary.json");

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Done. Results in output/nbody/");
    println!("══════════════════════════════════════════════════════════════════\n");
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    for i in 0..args.len() - 1 {
        if args[i] == name {
            if let Ok(val) = args[i + 1].parse() {
                return val;
            }
        }
    }
    default
}

fn save_snapshot(sim: &NBodySimulation, filename: &str) {
    let mut file = fs::File::create(filename).unwrap();
    writeln!(file, "x,y,z,vx,vy,vz,mass,sign").unwrap();

    // Sample if too many particles
    let sample_rate = if sim.particles.len() > 10000 {
        sim.particles.len() / 10000
    } else {
        1
    };

    for (i, p) in sim.particles.iter().enumerate() {
        if i % sample_rate == 0 {
            let sign = match p.sign {
                MassSign::Positive => 1,
                MassSign::Negative => -1,
            };
            writeln!(file, "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z, p.mass, sign).unwrap();
        }
    }
}
