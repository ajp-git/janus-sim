//! Debug test: 8M full simulation loop identical to run_8m_zeldovich

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Normal, Distribution};
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const ZELDOVICH_AMPLITUDE: f64 = 1e-3;
const ZELDOVICH_LAMBDA: f64 = 100.0;
const ZELDOVICH_SIGMA: f64 = 0.1;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;

fn generate_zeldovich_ics(n_total: usize, box_size: f64, seed: u64, eta: f64)
    -> (Vec<f64>, Vec<f64>, Vec<i32>)
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_per_axis = (n_total as f64).powf(1.0/3.0).ceil() as usize;
    let cell_size = box_size / n_per_axis as f64;
    let box_half = box_size / 2.0;
    let normal = Normal::new(0.0, ZELDOVICH_SIGMA).unwrap();
    let k = 2.0 * std::f64::consts::PI / ZELDOVICH_LAMBDA;
    
    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);
    
    let mut count = 0;
    'outer: for ix in 0..n_per_axis {
        for iy in 0..n_per_axis {
            for iz in 0..n_per_axis {
                if count >= n_total { break 'outer; }
                let x0 = (ix as f64 + 0.5) * cell_size - box_half;
                let y0 = (iy as f64 + 0.5) * cell_size - box_half;
                let z0 = (iz as f64 + 0.5) * cell_size - box_half;
                let phase_x = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let phase_y = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let phase_z = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let amp: f64 = normal.sample(&mut rng);
                let dx = ZELDOVICH_AMPLITUDE * amp * (k * x0 + phase_x).sin();
                let dy = ZELDOVICH_AMPLITUDE * amp * (k * y0 + phase_y).sin();
                let dz = ZELDOVICH_AMPLITUDE * amp * (k * z0 + phase_z).sin();
                let mut x = x0 + dx;
                let mut y = y0 + dy;
                let mut z = z0 + dz;
                if x > box_half { x -= box_size; } if x < -box_half { x += box_size; }
                if y > box_half { y -= box_size; } if y < -box_half { y += box_size; }
                if z > box_half { z -= box_size; } if z < -box_half { z += box_size; }
                positions.push(x); positions.push(y); positions.push(z);
                velocities.push(0.0); velocities.push(0.0); velocities.push(0.0);
                let sign = if rng.random::<f64>() < (1.0 / (1.0 + eta)) { 1 } else { -1 };
                signs.push(sign);
                count += 1;
            }
        }
    }
    (positions, velocities, signs)
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    
    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);
    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;
    let mut tau = cosmo.tau_start;
    
    let (positions, velocities, signs) = generate_zeldovich_ics(n, box_size, 12345, eta);
    let mut sim = GpuNBodySimulation::new_with_state(n_positive, n_negative, box_size, positions, velocities, signs)?;
    sim.set_theta(0.5);
    
    // File I/O
    fs::create_dir_all("/app/output/test_debug")?;
    let mut ts_file = BufWriter::new(File::create("/app/output/test_debug/time_series.csv")?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;
    
    // Initial diagnostics
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation_distance()?;
    println!("Initial: KE={:.4e}, S={:.6}", ke_init, seg_init);
    
    let ke_ref = 1.0;  // Avoid div by 0 for cold start
    
    for step in 1..=10 {
        let t_step = std::time::Instant::now();
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;
        
        println!("Step {} (z={:.2}) step...", step, z);
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;
        
        println!("Step {} diagnostics...", step);
        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;
        
        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke / ke_ref, seg, step_time)?;
        
        if step % 5 == 0 || step <= 5 {
            println!("Step {} | z={:.2} | KE={:.4e} | S={:.6} | {}ms", step, z, ke, seg, step_time as u64);
            ts_file.flush()?;
        }
    }
    
    println!("Test complete - no hang detected");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
