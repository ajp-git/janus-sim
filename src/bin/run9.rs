//! Single run: eta=0.95, lambda=8

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::path::Path;
use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const N_PARTICLES: usize = 2_000_000;
const BOX_SIZE: f64 = 1000.0;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 1500;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const SEED: u64 = 42;
const DT: f64 = 0.01;

fn main() {
    #[cfg(feature = "cuda")]
    run_single();
}

#[cfg(feature = "cuda")]
fn run_single() {
    let eta = 0.95;
    let lambda = 8.0;
    
    println!("RUN 9/9: eta={:.2}, lambda={:.0}", eta, lambda);
    
    let run_dir = Path::new("/app/output/scan_refined/eta0.95_lam8");
    let ts_path = run_dir.join("time_series.csv");
    let mut ts_file = BufWriter::new(File::create(&ts_path).unwrap());
    writeln!(ts_file, "step,z,a,purity,ke_ratio").unwrap();
    
    let start = Instant::now();
    let (pos_data, vel_data, signs_data) = generate_ics();
    
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_data, vel_data, signs_data, BOX_SIZE).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(lambda);
    
    let ke0 = sim.kinetic_energy().unwrap_or(1.0).max(1e-20);
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);
    
    let snapshot_steps = [300, 750, 1200, 1500];
    let mut purs = vec![0.0; 4];
    let mut snap_idx = 0;
    
    for step in 1..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end { cosmo.get_params_at_tau(tau) } else { (1.0, 0.0) };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };
        
        sim.set_current_z(z);
        sim.step_dkd(DT, h, dtau_per_dt).unwrap();
        
        if step % 10 == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let ke_ratio = sim.kinetic_energy().unwrap_or(0.0) / ke0;
            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4e}", step, z, a, purity, ke_ratio).unwrap();
        }
        
        if snapshot_steps.contains(&step) {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            purs[snap_idx] = purity;
            snap_idx += 1;
            println!("  step {:4} | z={:.3} | P={:.4} <<<", step, z, purity);
        }
        
        if step % 100 == 0 && !snapshot_steps.contains(&step) {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            println!("  step {:4} | z={:.2} | P={:.4}", step, z, purity);
        }
    }
    
    ts_file.flush().unwrap();
    let elapsed = start.elapsed().as_secs_f64() / 60.0;
    println!("Completed in {:.1} min, P(z=0) = {:.4}", elapsed, purs[3]);
    
    // Append to summary
    let mut summary = fs::OpenOptions::new().append(true).open("/app/output/scan_refined/summary.csv").unwrap();
    writeln!(summary, "9,{:.2},{:.0},{:.4},{:.4},{:.4},{:.4},{:.1}", eta, lambda, purs[0], purs[1], purs[2], purs[3], elapsed).unwrap();
}

#[cfg(feature = "cuda")]
fn generate_ics() -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    use std::f64::consts::PI;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let n_grid = (N_PARTICLES as f64 / 2.0).powf(1.0/3.0).ceil() as usize;
    let cell = BOX_SIZE / n_grid as f64;
    let box_half = BOX_SIZE / 2.0;
    
    let h = 0.7; let omega_m = 0.3; let n_s = 0.965;
    let theta = 2.725 / 2.7;
    let k_eq = 0.0746 * omega_m * h * h / (theta * theta);
    let a = 1.0 / (1.0 + Z_INIT);
    let omega_m_z = omega_m / (omega_m + (1.0 - omega_m) * a.powi(3));
    let d_z = a * (omega_m_z.powf(4.0/7.0) - (1.0 - omega_m_z) + (1.0 + omega_m_z/2.0) * (1.0 + (1.0-omega_m_z)/70.0)).powf(-1.0);
    let f_omega = omega_m.powf(0.55);
    let h0_gyr = 0.0715;
    let e_z = ((omega_m * (1.0 + Z_INIT).powi(3)) + (1.0 - omega_m)).sqrt();
    let h_z = h0_gyr * e_z;
    let vel_factor = h_z * f_omega;
    let disp_scale = 25.0 * d_z;
    
    let mut pos_data = Vec::with_capacity(N_PARTICLES * 3);
    let mut vel_data = Vec::with_capacity(N_PARTICLES * 3);
    let mut signs_data = Vec::with_capacity(N_PARTICLES);
    let mut count = 0;
    
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                if count >= N_PARTICLES { break; }
                let x0 = (ix as f64 + 0.5) * cell - box_half;
                let y0 = (iy as f64 + 0.5) * cell - box_half;
                let z0 = (iz as f64 + 0.5) * cell - box_half;
                let k = 2.0 * PI / BOX_SIZE * ((ix*ix + iy*iy + iz*iz) as f64).sqrt().max(1.0);
                let pk = k.powf(n_s) / (1.0 + (k/k_eq).powi(2)).powf(2.0);
                let amp = (pk.sqrt() * disp_scale * cell / BOX_SIZE).min(cell * 0.5);
                let psi_x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_y: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_z: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let (dx, dy, dz) = (psi_x * amp, psi_y * amp, psi_z * amp);
                pos_data.extend_from_slice(&[(x0 + dx) as f32, (y0 + dy) as f32, (z0 + dz) as f32]);
                vel_data.extend_from_slice(&[(dx * vel_factor) as f32, (dy * vel_factor) as f32, (dz * vel_factor) as f32]);
                signs_data.push(if rng.gen::<bool>() { 1 } else { -1 });
                count += 1;
            }
        }
    }
    while count < N_PARTICLES {
        let x: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        let y: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        let z: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel_data.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs_data.push(if rng.gen::<bool>() { 1 } else { -1 });
        count += 1;
    }
    (pos_data, vel_data, signs_data)
}
