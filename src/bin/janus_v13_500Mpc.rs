//! JANUS V13 — Large Box Test
//! 5M particles, L=500 Mpc, z=5→0

use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const L_BOX: f64 = 500.0;
const N_GRID: usize = 171;  // 171³ = 5,000,211
const ETA: f64 = 1.045;
const ALPHA_IC: f64 = 1.6;
const EPSILON: f64 = 0.45;  // scaled: 0.18 * 500/200
const THETA: f64 = 0.7;
const R_CUT: f64 = 45.0;    // scaled: 18 * 500/200
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 5000;
const SNAP_INT: usize = 100;
const LOG_INT: usize = 10;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let n = N_GRID * N_GRID * N_GRID;
    let out = "/app/output/janus_v13_500Mpc";
    fs::create_dir_all(format!("{}/snapshots", out)).unwrap();
    fs::create_dir_all(format!("{}/fields", out)).unwrap();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  JANUS V13 — LARGE BOX | 5M | L=500 | z=5→0               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("N={} η={} ε={} θ={} R_cut={} dt={}", n, ETA, EPSILON, THETA, R_CUT, DT);

    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);
    let (a0, h0) = cosmo.get_params_at_tau(cosmo.tau_start);
    println!("τ=[{:.4},{:.4}] dτ/dt={:.4} z_init={:.2} H_init={:.4}",
             cosmo.tau_start, cosmo.tau_end, dtau, 1.0/a0-1.0, h0);

    println!("Generating ICs...");
    let t0 = Instant::now();
    let (pos, vel, signs) = gen_ics(42, N_GRID);
    println!("  Done in {:.1}s", t0.elapsed().as_secs_f64());

    let np = signs.iter().filter(|&&s| s > 0).count();
    let nm = n - np;
    println!("  N+={} N-={} ratio={:.4}", np, nm, nm as f64/np as f64);

    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    println!("Init GPU...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(2);

    let ke0 = sim.kinetic_energy().unwrap_or(1e-20).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);
    println!("KE0={:.4e} Seg0={:.4}", ke0, seg0);

    let mut ts = File::create(format!("{}/time_series.csv", out)).unwrap();
    writeln!(ts, "step,z,a,H,KE_ratio,seg,sigma_P,L_J,xi").unwrap();

    save_snap(&sim, &format!("{}/snapshots/snap_000000.bin", out), n);
    let (sp, lj, xi) = metrics(&sim, 64);
    save_field(&sim, &format!("{}/fields/field_000000.bin", out), 64);
    writeln!(ts, "0,{:.4},{:.5},{:.5},1.0,{:.4},{:.4},{:.2},{:.2}", 1.0/a0-1.0, a0, h0, seg0, sp, lj, xi).unwrap();

    println!("\n{:>5} {:>7} {:>7} {:>10} {:>7} {:>6} {:>6} {:>5}",
             "Step", "z", "a", "KE/KE0", "Seg", "σ_P", "L_J", "ms");
    println!("{}", "─".repeat(65));

    let start = Instant::now();
    let mut seg_max = seg0;

    for step in 1..=STEPS {
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
            let eta = elapsed / step as f64 * (STEPS - step) as f64 / 3600.0;
            println!("  → snap_{:06} | ETA: {:.1}h", step, eta);
        }
    }

    let total = start.elapsed().as_secs_f64();
    let seg_f = sim.segregation().unwrap_or(0.0);
    let (sp_f, lj_f, xi_f) = metrics(&sim, 64);

    println!("\n═══ COMPLETE ═══");
    println!("Time: {:.1}h ({:.0}ms/step)", total/3600.0, 1000.0*total/STEPS as f64);
    println!("Seg: {:.4} → {:.4} (max {:.4})", seg0, seg_f, seg_max);
    println!("σ_P={:.4} L_J={:.1} ξ={:.1}", sp_f, lj_f, xi_f);
    println!("Output: {}", out);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn gen_ics(seed: u64, ng: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = ng * ng * ng;
    let cell = L_BOX / ng as f64;
    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let p_plus = 1.0 / (1.0 + ETA);

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let x0 = (ix as f64 + 0.5) * cell - L_BOX / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - L_BOX / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - L_BOX / 2.0;

                let k = 2.0 * PI / L_BOX;
                let px: f64 = rng.random::<f64>() * 2.0 * PI;
                let py: f64 = rng.random::<f64>() * 2.0 * PI;
                let pz: f64 = rng.random::<f64>() * 2.0 * PI;

                let (mut dx, mut dy, mut dz) = (0.0, 0.0, 0.0);
                let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);

                for m in 1..=8 {
                    let km = m as f64 * k;
                    if km < 0.5 {
                        let amp = 0.015 * (km / k).powf(-1.0);
                        dx += amp * (km * x0 + px).sin();
                        dy += amp * (km * y0 + py).sin();
                        dz += amp * (km * z0 + pz).sin();
                        vx += amp * km * (km * x0 + px).cos() * ALPHA_IC;
                        vy += amp * km * (km * y0 + py).cos() * ALPHA_IC;
                        vz += amp * km * (km * z0 + pz).cos() * ALPHA_IC;
                    }
                }

                dx += norm.sample(&mut rng) * cell * 0.05;
                dy += norm.sample(&mut rng) * cell * 0.05;
                dz += norm.sample(&mut rng) * cell * 0.05;

                pos.push(((x0 + dx) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0);
                pos.push(((y0 + dy) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0);
                pos.push(((z0 + dz) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0);
                vel.push(vx); vel.push(vy); vel.push(vz);
                signs.push(if rng.random::<f64>() < p_plus { 1 } else { -1 });
            }
        }
    }
    (pos, vel, signs)
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
        let idx = ix + iy*g + iz*g*g;
        if signs[i] > 0 { rp[idx] += 1.0; } else { rm[idx] += 1.0; }
    }

    let mut pf: Vec<f64> = rp.iter().zip(&rm).map(|(&p, &m)| if p+m > 0.0 { (p-m)/(p+m) } else { 0.0 }).collect();
    let pm: f64 = pf.iter().sum::<f64>() / pf.len() as f64;
    let sp = (pf.iter().map(|&p| (p-pm).powi(2)).sum::<f64>() / pf.len() as f64).sqrt();

    let mut gs = 0.0;
    for iz in 0..g { for iy in 0..g { for ix in 0..g {
        let idx = ix + iy*g + iz*g*g;
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
            let idx = ix + iy*g + iz*g*g;
            let idxr = ((ix+r)%g) + iy*g + iz*g*g;
            sr += (pf[idx] - pf[idxr]).powi(2);
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
