//! T_floor test: T_floor_minus = 10000 K (×100)
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::sph_pressure_gpu::GpuSphPressure;

const N_PLUS: usize = 250_000;
const N_MINUS: usize = 250_000;
const BOX_SIZE: f64 = 100.0;
const DT: f64 = 0.001;
const STEPS: usize = 500;

const T_INIT: f64 = 10000.0;
const T_FLOOR_PLUS: f64 = 100.0;
const T_FLOOR_MINUS: f64 = 10000.0;  // ×100
const PARTICLE_MASS: f64 = 1e10;

fn main() {
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    let output_dir = "/app/output/tfloor_test_10k";
    create_dir_all(output_dir).unwrap();

    println!("T_floor_minus = 10000 K (×100)");

    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE).unwrap();
    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(0.5);
    gpu_sim.set_c_ratio(1.0);

    let device = Arc::new(CudaDevice::new(0).unwrap());
    let mut sph_plus = GpuSphPressure::new(Arc::clone(&device), N_PLUS, PARTICLE_MASS, BOX_SIZE).unwrap();
    let mut sph_minus = GpuSphPressure::new(Arc::clone(&device), N_MINUS, PARTICLE_MASS, BOX_SIZE).unwrap();

    let temp_plus = vec![T_INIT.max(T_FLOOR_PLUS); N_PLUS];
    let temp_minus = vec![T_INIT.max(T_FLOOR_MINUS); N_MINUS];

    let csv_path = format!("{}/evolution.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,ratio,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus").unwrap();

    for step in 0..=STEPS {
        if step % 50 == 0 {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs = gpu_sim.get_signs().unwrap();

            let (v_plus, v_minus) = compute_vrms(&vel, &signs);
            let (rho_plus, rho_minus) = compute_rho_max(&pos, &signs, BOX_SIZE);
            let ratio = if v_plus > 0.0 { (v_minus / v_plus).max(v_plus / v_minus) } else { 1.0 };

            println!("step {:4} | ratio={:.4} | ρ+={:.0} ρ-={:.0} | v+={:.0} v-={:.0}",
                step, ratio, rho_plus, rho_minus, v_plus, v_minus);
            writeln!(csv, "{},{:.6},{:.1},{:.1},{:.1},{:.1}", step, ratio, rho_plus, rho_minus, v_plus, v_minus).unwrap();
        }

        if step == STEPS { break; }
        apply_sph_kick(&mut gpu_sim, &mut sph_plus, &mut sph_minus, &temp_plus, &temp_minus, DT);
        let _ = gpu_sim.step(DT);
    }

    csv.flush().unwrap();
    println!("\n=== T_floor_minus=10000K DONE ===");
}

#[cfg(feature = "cuda")]
fn compute_vrms(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let (mut sp, mut sm, mut np, mut nm) = (0.0, 0.0, 0usize, 0usize);
    for i in 0..signs.len() {
        let v2 = vel[i*3].powi(2) + vel[i*3+1].powi(2) + vel[i*3+2].powi(2);
        if signs[i] > 0 { sp += v2; np += 1; } else { sm += v2; nm += 1; }
    }
    ((sp / np as f64).sqrt() * 977.8, (sm / nm as f64).sqrt() * 977.8)
}

#[cfg(feature = "cuda")]
fn compute_rho_max(pos: &[f64], signs: &[i32], box_size: f64) -> (f64, f64) {
    let ng = 64usize;
    let mut gp = vec![0u32; ng*ng*ng];
    let mut gm = vec![0u32; ng*ng*ng];
    for i in 0..signs.len() {
        let ix = ((pos[i*3]/box_size + 0.5) * ng as f64).floor() as usize % ng;
        let iy = ((pos[i*3+1]/box_size + 0.5) * ng as f64).floor() as usize % ng;
        let iz = ((pos[i*3+2]/box_size + 0.5) * ng as f64).floor() as usize % ng;
        let idx = ix*ng*ng + iy*ng + iz;
        if signs[i] > 0 { gp[idx] += 1; } else { gm[idx] += 1; }
    }
    (*gp.iter().max().unwrap() as f64, *gm.iter().max().unwrap() as f64)
}

#[cfg(feature = "cuda")]
fn apply_sph_kick(gpu_sim: &mut GpuNBodySimulation, sph_plus: &mut GpuSphPressure, sph_minus: &mut GpuSphPressure, temp_plus: &[f64], temp_minus: &[f64], dt: f64) {
    let pos = gpu_sim.get_positions().unwrap();
    let mut vel = gpu_sim.get_velocities().unwrap();
    let signs = gpu_sim.get_signs().unwrap();

    let (mut idx_p, mut idx_m) = (Vec::new(), Vec::new());
    for i in 0..signs.len() { if signs[i] > 0 { idx_p.push(i); } else { idx_m.push(i); } }

    let mut pos_p = Vec::with_capacity(idx_p.len() * 3);
    let mut pos_m = Vec::with_capacity(idx_m.len() * 3);
    for &i in &idx_p { pos_p.extend_from_slice(&pos[i*3..i*3+3]); }
    for &i in &idx_m { pos_m.extend_from_slice(&pos[i*3..i*3+3]); }

    let acc_p = sph_plus.compute_pressure_accelerations(&pos_p, temp_plus).unwrap();
    let acc_m = sph_minus.compute_pressure_accelerations(&pos_m, temp_minus).unwrap();

    for (j, &i) in idx_p.iter().enumerate() { for k in 0..3 { vel[i*3+k] += acc_p[j*3+k] * dt; } }
    for (j, &i) in idx_m.iter().enumerate() { for k in 0..3 { vel[i*3+k] += acc_m[j*3+k] * dt; } }

    gpu_sim.set_velocities(&vel).unwrap();
}
