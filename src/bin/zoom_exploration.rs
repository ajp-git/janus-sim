//! Zoom Exploration — parametric runs for filament detection
//! Usage: zoom_exploration --box 50 --n 100000 --k-min 20 --eps 0.08 --amp 0.5 --output /path

use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use clap::Parser;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[derive(Parser)]
#[command(name = "zoom_exploration")]
struct Args {
    #[arg(long, default_value = "50.0")]
    box_size: f64,
    #[arg(long, default_value = "100000")]
    n: usize,
    #[arg(long, default_value = "20")]
    k_min: usize,
    #[arg(long, default_value = "0.08")]
    eps: f64,
    #[arg(long, default_value = "0.5")]
    amp: f64,
    #[arg(long)]
    output: String,
    #[arg(long, default_value = "2000")]
    steps: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
}

const ETA: f64 = 1.045;
const THETA: f64 = 0.5;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const SNAP_INT: usize = 100;
const LOG_INT: usize = 10;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args = Args::parse();

    let l_box = args.box_size;
    let n_target = args.n;
    let k_min = args.k_min;
    let eps = args.eps;
    let amp = args.amp;
    let steps = args.steps;
    let out = &args.output;

    // Compute grid size from N
    let ng = (n_target as f64).cbrt().round() as usize;
    let n = ng * ng * ng;

    // r_cut scaled to box
    let r_cut = 0.09 * l_box;  // 9% of box

    fs::create_dir_all(format!("{}/snapshots", out)).unwrap();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  ZOOM EXPLORATION | N={:>7} | L={:>3} Mpc               ║", n, l_box as i32);
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("k_min={} ε={:.3} amp={:.2} θ={} r_cut={:.1}", k_min, eps, amp, THETA, r_cut);

    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (steps as f64 * DT);
    let (a0, h0) = cosmo.get_params_at_tau(cosmo.tau_start);
    println!("z_init={:.2} H_init={:.4}", 1.0/a0-1.0, h0);

    println!("Generating Zel'dovich ICs (k_min={})...", k_min);
    let t0 = Instant::now();
    let (pos, vel, signs) = gen_ics_zeldovich(args.seed, ng, l_box, k_min, amp);
    println!("  Done in {:.1}s", t0.elapsed().as_secs_f64());

    let np = signs.iter().filter(|&&s| s > 0).count();
    let nm = n - np;
    println!("  N+={} N-={}", np, nm);

    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    println!("Init GPU...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, l_box).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(eps);
    sim.set_pm_k_min(2);

    // CRITICAL: Apply proper Janus virialization (α ≈ 4-6)
    let n_sample = (n / 10).max(1000).min(50000);  // 10% of particles, capped
    println!("Applying virialize_sampled({})...", n_sample);
    if let Err(e) = sim.virialize_sampled(n_sample) {
        println!("  WARNING: virialize_sampled failed: {}", e);
    }

    let ke0 = sim.kinetic_energy().unwrap_or(1e-20).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);
    println!("KE0={:.4e} Seg0={:.4}", ke0, seg0);

    let mut ts = File::create(format!("{}/time_series.csv", out)).unwrap();
    writeln!(ts, "step,z,a,H,KE_ratio,seg,sigma_P,L_J").unwrap();

    save_snap(&sim, &format!("{}/snapshots/snap_000000.bin", out), n);
    let (sp, lj) = metrics(&sim, 32, l_box);
    writeln!(ts, "0,{:.4},{:.5},{:.5},1.0,{:.4},{:.4},{:.2}", 1.0/a0-1.0, a0, h0, seg0, sp, lj).unwrap();

    println!("\n{:>5} {:>7} {:>7} {:>10} {:>7} {:>6} {:>6} {:>5}",
             "Step", "z", "a", "KE/KE0", "Seg", "σ_P", "L_J", "ms");
    println!("{}", "─".repeat(60));

    let start = Instant::now();
    let mut ke_max = 1.0f64;

    for step in 1..=steps {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau;
        let (a, h) = if tau <= cosmo.tau_end { cosmo.get_params_at_tau(tau) } else { (1.0, 0.0) };
        let z = if a > 0.0 { 1.0/a - 1.0 } else { 0.0 };

        let t0 = Instant::now();
        if let Err(e) = sim.step_treepm_gpu(DT, r_cut, h, dtau) {
            println!("ERROR step {}: {}", step, e);
            break;
        }
        let ms = t0.elapsed().as_millis();

        if step % LOG_INT == 0 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke0;
            if ke_ratio > ke_max { ke_max = ke_ratio; }
            let seg = sim.segregation().unwrap_or(0.0);

            let (sp, lj) = if step % SNAP_INT == 0 { metrics(&sim, 32, l_box) } else { (0.0, 0.0) };

            writeln!(ts, "{},{:.4},{:.5},{:.5},{:.4e},{:.4},{:.4},{:.2}",
                     step, z, a, h, ke_ratio, seg, sp, lj).unwrap();

            if step % SNAP_INT == 0 {
                println!("{:>5} {:>7.3} {:>7.5} {:>10.3e} {:>7.4} {:>6.3} {:>6.1} {:>5}",
                         step, z, a, ke_ratio, seg, sp, lj, ms);
                save_snap(&sim, &format!("{}/snapshots/snap_{:06}.bin", out, step), n);
                ts.flush().unwrap();
            } else {
                println!("{:>5} {:>7.3} {:>7.5} {:>10.3e} {:>7.4} {:>6} {:>6} {:>5}",
                         step, z, a, ke_ratio, seg, "-", "-", ms);
            }

            // Early termination if KE explodes
            if ke_ratio > 5.0 {
                println!("KE EXPLOSION - ABORT");
                break;
            }
        }
    }

    let total = start.elapsed().as_secs_f64();
    let seg_f = sim.segregation().unwrap_or(0.0);
    let (sp_f, lj_f) = metrics(&sim, 32, l_box);
    let ke_f = sim.kinetic_energy().unwrap_or(0.0) / ke0;

    // Save final summary
    let mut summary = File::create(format!("{}/summary.json", out)).unwrap();
    writeln!(summary, r#"{{"box":{:.1},"n":{},"k_min":{},"eps":{:.4},"amp":{:.2},"seg_final":{:.4},"sigma_P":{:.4},"L_J":{:.2},"KE_max":{:.3},"KE_final":{:.3},"time_s":{:.1}}}"#,
             l_box, n, k_min, eps, amp, seg_f, sp_f, lj_f, ke_max, ke_f, total).unwrap();

    println!("\n═══ COMPLETE ═══");
    println!("Time: {:.1}min ({:.0}ms/step)", total/60.0, 1000.0*total/steps as f64);
    println!("Seg: {:.4} → {:.4}", seg0, seg_f);
    println!("σ_P={:.4} L_J={:.1}", sp_f, lj_f);
    println!("KE_max={:.3} KE_final={:.3}", ke_max, ke_f);
    println!("Output: {}", out);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn gen_ics_zeldovich(seed: u64, ng: usize, l_box: f64, k_min: usize, amp: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    use rand::seq::SliceRandom;

    let mut rng = StdRng::seed_from_u64(seed);
    let n = ng * ng * ng;
    let cell = l_box / ng as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let p_plus = 1.0 / (1.0 + ETA);
    let k_fund = 2.0 * PI / l_box;

    // Initial velocity scale (will be rescaled by virialize_sampled)
    // Use small non-zero values so α scaling works
    let v_init = 1.0;

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let x0 = (ix as f64 + 0.5) * cell - l_box / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - l_box / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - l_box / 2.0;

                // Random phases for this particle
                let phase_x: f64 = rng.random::<f64>() * 2.0 * PI;
                let phase_y: f64 = rng.random::<f64>() * 2.0 * PI;
                let phase_z: f64 = rng.random::<f64>() * 2.0 * PI;

                let (mut dx, mut dy, mut dz) = (0.0, 0.0, 0.0);
                let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);

                // Add Zel'dovich displacement for modes from k_min to ng/2
                let k_max = (ng / 2).min(64);
                for m in k_min..=k_max {
                    let km = m as f64 * k_fund;

                    // Amplitude with P(k) ~ k^-1 scaling
                    let a_mode = amp * cell * 0.1 * (k_min as f64 / m as f64);

                    dx += a_mode * (km * x0 + phase_x).sin();
                    dy += a_mode * (km * y0 + phase_y).sin();
                    dz += a_mode * (km * z0 + phase_z).sin();

                    // Zel'dovich velocities (proportional to displacement gradient)
                    vx += a_mode * km * (km * x0 + phase_x).cos() * v_init;
                    vy += a_mode * km * (km * y0 + phase_y).cos() * v_init;
                    vz += a_mode * km * (km * z0 + phase_z).cos() * v_init;
                }

                // Small random jitter
                dx += norm.sample(&mut rng) * cell * 0.02;
                dy += norm.sample(&mut rng) * cell * 0.02;
                dz += norm.sample(&mut rng) * cell * 0.02;

                // Wrap to box
                let mut x = x0 + dx;
                let mut y = y0 + dy;
                let mut z = z0 + dz;
                if x > l_box / 2.0 { x -= l_box; }
                if x < -l_box / 2.0 { x += l_box; }
                if y > l_box / 2.0 { y -= l_box; }
                if y < -l_box / 2.0 { y += l_box; }
                if z > l_box / 2.0 { z -= l_box; }
                if z < -l_box / 2.0 { z += l_box; }

                pos.push(x); pos.push(y); pos.push(z);
                vel.push(vx); vel.push(vy); vel.push(vz);

                // Sign based on eta ratio
                signs.push(if rng.random::<f64>() < p_plus { 1 } else { -1 });
            }
        }
    }

    // Shuffle signs for random spatial distribution
    signs.shuffle(&mut rng);

    (pos, vel, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn metrics(sim: &GpuNBodyTwoPass, g: usize, l_box: f64) -> (f64, f64) {
    let (pos, _, signs) = match sim.get_particles() { Ok(d) => d, Err(_) => return (0.0, 0.0) };
    let cell = l_box / g as f64;
    let mut rp = vec![0.0f64; g*g*g];
    let mut rm = vec![0.0f64; g*g*g];

    for i in 0..pos.len()/3 {
        let x = pos[i*3] as f64 + l_box/2.0;
        let y = pos[i*3+1] as f64 + l_box/2.0;
        let z = pos[i*3+2] as f64 + l_box/2.0;
        let ix = ((x/cell) as usize).min(g-1);
        let iy = ((y/cell) as usize).min(g-1);
        let iz = ((z/cell) as usize).min(g-1);
        let idx = ix + iy*g + iz*g*g;
        if signs[i] > 0 { rp[idx] += 1.0; } else { rm[idx] += 1.0; }
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

    (sp, lj)
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

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() { println!("Need --features \"cuda cufft\""); }
