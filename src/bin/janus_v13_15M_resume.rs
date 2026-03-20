//! JANUS V13 15M — Resume from snapshot 2300

use std::fs::{self, File, OpenOptions};
use std::io::{Read as IoRead, Write, BufWriter, BufReader};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const L_BOX: f64 = 500.0;
const ETA: f64 = 1.045;
const EPSILON: f64 = 0.35;
const THETA: f64 = 0.7;
const R_CUT: f64 = 40.0;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 5000;
const SNAP_INT: usize = 100;
const LOG_INT: usize = 10;
const RESUME_STEP: usize = 2300;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let out = "/app/output/janus_v13_500Mpc_15M";
    let snap_path = format!("{}/snapshots/snap_{:06}.bin", out, RESUME_STEP);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  JANUS V13 15M — RESUME from step {}                    ║", RESUME_STEP);
    println!("╚═══════════════════════════════════════════════════════════╝");

    println!("Loading {}...", snap_path);
    let (pos, vel, signs, n) = load_snap(&snap_path);
    println!("  Loaded {} particles", n);

    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let tau_resume = cosmo.tau_start + (RESUME_STEP as f64) * DT * dtau;
    let (a_resume, h_resume) = cosmo.get_params_at_tau(tau_resume);
    let z_resume = 1.0/a_resume - 1.0;
    println!("Resume at z={:.3} a={:.5} H={:.5}", z_resume, a_resume, h_resume);

    println!("Init GPU...");
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(2);

    let ke0 = sim.kinetic_energy().unwrap_or(1e-20).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);
    println!("KE={:.4e} Seg={:.4}", ke0, seg0);

    let mut ts = OpenOptions::new()
        .append(true)
        .open(format!("{}/time_series.csv", out))
        .unwrap();

    println!("\n{:>5} {:>7} {:>7} {:>10} {:>7} {:>6} {:>6} {:>5}",
             "Step", "z", "a", "KE/KE0", "Seg", "σ_P", "L_J", "ms");
    println!("{}", "─".repeat(65));

    let start = Instant::now();
    let mut seg_max = seg0;

    for step in (RESUME_STEP + 1)..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau;
        let (a, h) = if tau <= cosmo.tau_end { cosmo.get_params_at_tau(tau) } else { (1.0, 0.0) };
        let z = if a > 0.0 { 1.0/a - 1.0 } else { 0.0 };

        let t0 = Instant::now();
        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, h, dtau) {
            println!("ERROR step {}: {}", step, e);
            break;
        }
        let ms = t0.elapsed().as_millis();

        if step % LOG_INT == 0 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let seg = sim.segregation().unwrap_or(0.0);
            if seg > seg_max { seg_max = seg; }

            let (sp, lj, xi) = if step % SNAP_INT == 0 { metrics(&sim, 64) } else { (0.0, 0.0, 0.0) };

            writeln!(ts, "{},{:.4},{:.5},{:.5},{:.4e},{:.4},{:.4},{:.2},{:.2}",
                     step, z, a, h, ke/ke0, seg, sp, lj, xi).unwrap();

            if step % SNAP_INT == 0 {
                println!("{:>5} {:>7.3} {:>7.5} {:>10.3e} {:>7.4} {:>6.3} {:>6.1} {:>5}",
                         step, z, a, ke/ke0, seg, sp, lj, ms);
            } else {
                println!("{:>5} {:>7.3} {:>7.5} {:>10.3e} {:>7.4} {:>6} {:>6} {:>5}",
                         step, z, a, ke/ke0, seg, "-", "-", ms);
            }
        }

        if step % SNAP_INT == 0 {
            save_snap(&sim, &format!("{}/snapshots/snap_{:06}.bin", out, step), n);
            save_field(&sim, &format!("{}/fields/field_{:06}.bin", out, step), 64);
            ts.flush().unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let done = step - RESUME_STEP;
            let remaining = STEPS - step;
            let eta = elapsed / done as f64 * remaining as f64 / 3600.0;
            println!("  → snap_{:06} | ETA: {:.1}h", step, eta);
        }
    }

    let total = start.elapsed().as_secs_f64();
    let seg_f = sim.segregation().unwrap_or(0.0);
    let (sp_f, lj_f, xi_f) = metrics(&sim, 64);

    println!("\n═══ COMPLETE ═══");
    println!("Time: {:.1}h ({:.0}ms/step)", total/3600.0, 1000.0*total/(STEPS - RESUME_STEP) as f64);
    println!("Seg: {:.4} → {:.4} (max {:.4})", seg0, seg_f, seg_max);
    println!("σ_P={:.4} L_J={:.1} ξ={:.1}", sp_f, lj_f, xi_f);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn load_snap(path: &str) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let f = File::open(path).unwrap();
    let mut r = BufReader::new(f);
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;
    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);
    let mut buf4 = [0u8; 4];
    for _ in 0..n {
        for _ in 0..3 { r.read_exact(&mut buf4).unwrap(); pos.push(f32::from_le_bytes(buf4) as f64); }
        for _ in 0..3 { r.read_exact(&mut buf4).unwrap(); vel.push(f32::from_le_bytes(buf4) as f64); }
        r.read_exact(&mut buf4).unwrap();
        signs.push(if f32::from_le_bytes(buf4) > 0.0 { 1 } else { -1 });
    }
    (pos, vel, signs, n)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn metrics(sim: &GpuNBodyTwoPass, g: usize) -> (f64, f64, f64) {
    let (pos, _, signs) = match sim.get_particles() { Ok(d) => d, Err(_) => return (0.0, 0.0, 0.0) };
    let cell = L_BOX / g as f64;
    let mut rp = vec![0.0f64; g*g*g];
    let mut rm = vec![0.0f64; g*g*g];
    for i in 0..pos.len()/3 {
        let x = pos[i*3] as f64 + L_BOX/2.0;
        let y = pos[i*3+1] as f64 + L_BOX/2.0;
        let z = pos[i*3+2] as f64 + L_BOX/2.0;
        let ix = ((x/cell) as usize).min(g-1);
        let iy = ((y/cell) as usize).min(g-1);
        let iz = ((z/cell) as usize).min(g-1);
        if signs[i] > 0 { rp[ix + iy*g + iz*g*g] += 1.0; } else { rm[ix + iy*g + iz*g*g] += 1.0; }
    }
    let pf: Vec<f64> = rp.iter().zip(&rm).map(|(&p, &m)| if p+m > 0.0 { (p-m)/(p+m) } else { 0.0 }).collect();
    let pm: f64 = pf.iter().sum::<f64>() / pf.len() as f64;
    let sp = (pf.iter().map(|&p| (p-pm).powi(2)).sum::<f64>() / pf.len() as f64).sqrt();
    let mut gs = 0.0;
    for iz in 0..g { for iy in 0..g { for ix in 0..g {
        let gx = (pf[((ix+1)%g) + iy*g + iz*g*g] - pf[((ix+g-1)%g) + iy*g + iz*g*g]) / (2.0*cell);
        let gy = (pf[ix + ((iy+1)%g)*g + iz*g*g] - pf[ix + ((iy+g-1)%g)*g + iz*g*g]) / (2.0*cell);
        let gz = (pf[ix + iy*g + ((iz+1)%g)*g*g] - pf[ix + iy*g + ((iz+g-1)%g)*g*g]) / (2.0*cell);
        gs += (gx*gx + gy*gy + gz*gz).sqrt();
    }}}
    let mg = gs / (g*g*g) as f64;
    let lj = if mg > 0.0 { sp / mg } else { 0.0 };
    let ssat = 2.0 * sp * sp;
    let mut xi = g as f64 / 2.0 * cell;
    for r in 1..g/2 {
        let mut sr = 0.0;
        for iz in 0..g { for iy in 0..g { for ix in 0..g {
            sr += (pf[ix + iy*g + iz*g*g] - pf[((ix+r)%g) + iy*g + iz*g*g]).powi(2);
        }}}
        sr /= (g*g*g) as f64;
        if sr > 0.9 * ssat { xi = r as f64 * cell; break; }
    }
    (sp, lj, xi)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snap(sim: &GpuNBodyTwoPass, path: &str, n: usize) {
    let (pos, vel, signs) = sim.get_particles().unwrap();
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    for i in 0..pos.len()/3 {
        w.write_all(&pos[i*3].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+1].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+2].to_le_bytes()).unwrap();
        w.write_all(&vel[i*3].to_le_bytes()).unwrap();
        w.write_all(&vel[i*3+1].to_le_bytes()).unwrap();
        w.write_all(&vel[i*3+2].to_le_bytes()).unwrap();
        w.write_all(&(if signs[i] > 0 { 1.0f32 } else { -1.0f32 }).to_le_bytes()).unwrap();
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_field(sim: &GpuNBodyTwoPass, path: &str, g: usize) {
    let (pos, _, signs) = match sim.get_particles() { Ok(d) => d, Err(_) => return };
    let cell = L_BOX / g as f64;
    let mut rp = vec![0.0f32; g*g*g];
    let mut rm = vec![0.0f32; g*g*g];
    for i in 0..pos.len()/3 {
        let x = pos[i*3] as f64 + L_BOX/2.0;
        let y = pos[i*3+1] as f64 + L_BOX/2.0;
        let z = pos[i*3+2] as f64 + L_BOX/2.0;
        let ix = ((x/cell) as usize).min(g-1);
        let iy = ((y/cell) as usize).min(g-1);
        let iz = ((z/cell) as usize).min(g-1);
        if signs[i] > 0 { rp[ix+iy*g+iz*g*g] += 1.0; } else { rm[ix+iy*g+iz*g*g] += 1.0; }
    }
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    w.write_all(&(g as u32).to_le_bytes()).unwrap();
    for &v in &rp { w.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &rm { w.write_all(&v.to_le_bytes()).unwrap(); }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() { println!("Need --features \"cuda cufft\""); }
