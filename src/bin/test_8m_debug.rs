//! Compare α_direct vs α_sampled for 100K particles

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0;
    
    println!("=== Comparison: α_direct vs α_sampled (100K) ===\n");
    
    // Create simulation WITHOUT analytical virialization
    // Use new_with_state with uniform positions and v_init=1.0
    println!("Creating simulation with uniform ICs (no virialization)...");
    
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let v_init = 1.0;
    
    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);
    
    for _ in 0..n_positive {
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        signs.push(1);
    }
    for _ in 0..n_negative {
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        positions.push((rng.random::<f64>() - 0.5) * box_size);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        velocities.push((rng.random::<f64>() - 0.5) * v_init);
        signs.push(-1);
    }
    
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive, n_negative, box_size,
        positions, velocities, signs
    )?;
    
    let ke_init = sim.kinetic_energy()?;
    println!("N+ = {}, N- = {}", n_positive, n_negative);
    println!("KE_initial = {:.4e}\n", ke_init);
    
    // Compute PE_direct using full Barnes-Hut (O(N log N))
    println!("--- Computing PE_direct (full Barnes-Hut) ---");
    let t0 = std::time::Instant::now();
    let pe_direct = sim.potential_energy_binding()?;
    println!("PE_direct = {:.4e} ({:.2}s)", pe_direct, t0.elapsed().as_secs_f64());
    
    let ke_target_direct = pe_direct.abs() / 2.0;
    let alpha_direct = (ke_target_direct / ke_init).sqrt();
    println!("KE_target (direct) = {:.4e}", ke_target_direct);
    println!("α_direct = {:.6}\n", alpha_direct);
    
    // Compute PE_sampled
    println!("--- Computing PE_sampled (5000 samples per sign) ---");
    let t0 = std::time::Instant::now();
    let pe_sampled = sim.potential_energy_binding_sampled(5000)?;
    println!("Time: {:.2}s", t0.elapsed().as_secs_f64());
    
    let ke_target_sampled = pe_sampled.abs() / 2.0;
    let alpha_sampled = (ke_target_sampled / ke_init).sqrt();
    println!("KE_target (sampled) = {:.4e}", ke_target_sampled);
    println!("α_sampled = {:.6}\n", alpha_sampled);
    
    // Comparison
    println!("=== COMPARISON ===");
    println!("PE_direct  = {:.4e}", pe_direct);
    println!("PE_sampled = {:.4e}", pe_sampled);
    println!("Ratio PE_sampled/PE_direct = {:.4}", pe_sampled / pe_direct);
    println!();
    println!("α_direct  = {:.6}", alpha_direct);
    println!("α_sampled = {:.6}", alpha_sampled);
    println!("Ratio α_sampled/α_direct = {:.4}", alpha_sampled / alpha_direct);
    println!();
    println!("Expected α ≈ 4-5 for proper Janus virialization");
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
