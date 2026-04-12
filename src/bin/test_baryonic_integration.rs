//! Tests d'intégration physique baryonique
//! OBLIGATOIRES avant tout run Janus
//! Lancer : cargo run --release --bin test_baryonic_integration

use janus::baryonic::cooling::apply_cooling;
use janus::baryonic::pressure::sound_speed;
use rand::Rng;

fn main() {
    println!("=== TESTS D'INTÉGRATION BARYONIQUE ===\n");

    let mut all_passed = true;

    all_passed &= test_cold_collapse();
    all_passed &= test_hot_stability();
    all_passed &= test_energy_conservation();
    all_passed &= test_momentum_conservation();

    println!("\n=== RÉSULTAT FINAL ===");
    if all_passed {
        println!("✓ TOUS LES TESTS PASSENT → Run Janus autorisé");
    } else {
        println!("✗ TESTS ÉCHOUÉS → STOP. Corriger avant de continuer.");
        std::process::exit(1);
    }
}

// ─────────────────────────────────────────────
// TEST 1 — Effondrement isotherme (sans Janus)
// ─────────────────────────────────────────────
fn test_cold_collapse() -> bool {
    println!("TEST 1 — Effondrement isotherme");
    println!("  N=10k m+ seulement, Box=2 Mpc, T=100K");
    println!("  Surdensité centrale, Steps=100");

    // Config (réduit pour vitesse)
    let n = 10_000usize;
    let box_size = 2.0f64;
    let t_init = 100.0f64;
    let g_code = 4.499e-15f64;
    let steps = 100usize;
    let dt = 0.002f64;

    let mut rng = rand::thread_rng();

    // Positions : 50% dans sphère centrale, 50% uniformes
    let r_core = 0.3;
    let n_core = n / 2;

    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, z) = if i < n_core {
            loop {
                let x = rng.gen_range(-r_core..r_core);
                let y = rng.gen_range(-r_core..r_core);
                let z = rng.gen_range(-r_core..r_core);
                if x*x + y*y + z*z < r_core * r_core {
                    break (x, y, z);
                }
            }
        } else {
            (
                rng.gen_range(-box_size/2.0..box_size/2.0),
                rng.gen_range(-box_size/2.0..box_size/2.0),
                rng.gen_range(-box_size/2.0..box_size/2.0),
            )
        };
        pos.push([x, y, z]);
    }

    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut temp: Vec<f64> = vec![t_init; n];
    let mass = 1e10f64;
    let mean_density = n as f64 * mass / box_size.powi(3);

    let mut rho_max_ratio = 1.0f64;
    let grid = 16usize;
    let cell = box_size / grid as f64;
    let cell_vol = cell.powi(3);

    for step in 0..steps {
        // Calculer densité sur grille
        let mut density_grid = vec![0.0f64; grid.pow(3)];

        for p in &pos {
            let ix = ((p[0] + box_size/2.0) / cell) as usize;
            let iy = ((p[1] + box_size/2.0) / cell) as usize;
            let iz = ((p[2] + box_size/2.0) / cell) as usize;
            if ix < grid && iy < grid && iz < grid {
                density_grid[ix * grid * grid + iy * grid + iz] += mass;
            }
        }

        let rho_max = density_grid.iter().cloned().fold(0.0f64, f64::max) / cell_vol;
        rho_max_ratio = rho_max / mean_density;

        if step % 20 == 0 {
            println!("  step {:4} : ρmax/ρ̄ = {:.2}", step, rho_max_ratio);
        }

        // Gravité depuis gradient de densité
        for i in 0..n {
            let ix = ((pos[i][0] + box_size/2.0) / cell) as usize;
            let iy = ((pos[i][1] + box_size/2.0) / cell) as usize;
            let iz = ((pos[i][2] + box_size/2.0) / cell) as usize;
            if ix == 0 || iy == 0 || iz == 0
                || ix >= grid-1 || iy >= grid-1 || iz >= grid-1 {
                continue;
            }

            let idx = |x: usize, y: usize, z: usize| x*grid*grid + y*grid + z;
            let rho_xp = density_grid[idx(ix+1, iy, iz)];
            let rho_xm = density_grid[idx(ix-1, iy, iz)];
            let rho_yp = density_grid[idx(ix, iy+1, iz)];
            let rho_ym = density_grid[idx(ix, iy-1, iz)];
            let rho_zp = density_grid[idx(ix, iy, iz+1)];
            let rho_zm = density_grid[idx(ix, iy, iz-1)];

            let ax = -g_code * (rho_xp - rho_xm) / (2.0 * cell);
            let ay = -g_code * (rho_yp - rho_ym) / (2.0 * cell);
            let az = -g_code * (rho_zp - rho_zm) / (2.0 * cell);

            vel[i][0] += ax * dt;
            vel[i][1] += ay * dt;
            vel[i][2] += az * dt;

            // Cooling (z=0 for this test)
            let overdensity = density_grid[idx(ix, iy, iz)] / cell_vol / mean_density;
            temp[i] = apply_cooling(temp[i], overdensity.max(1.0), 0.0, dt);
        }

        // Drift
        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
                if pos[i][k] > box_size/2.0 { pos[i][k] -= box_size; }
                if pos[i][k] < -box_size/2.0 { pos[i][k] += box_size; }
            }
        }
    }

    let passed = rho_max_ratio > 5.0;
    if passed {
        println!("  ✓ PASS : ρmax/ρ̄ = {:.2} > 5.0", rho_max_ratio);
    } else {
        println!("  ✗ FAIL : ρmax/ρ̄ = {:.2} (attendu > 5.0)", rho_max_ratio);
    }
    passed
}

// ─────────────────────────────────────────────
// TEST 2 — Stabilité gaz chaud (sans Janus)
// ─────────────────────────────────────────────
fn test_hot_stability() -> bool {
    println!("\nTEST 2 — Stabilité gaz chaud");
    println!("  N=5k m+ seulement, Box=5 Mpc, T=1e7K, Steps=100");

    let n = 5_000usize;
    let box_size = 5.0f64;
    let t_init = 1e7f64;
    let g_code = 4.499e-15f64;
    let steps = 100usize;
    let dt = 0.001f64;

    let mut rng = rand::thread_rng();

    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mass = 1e10f64;
    let mean_density = n as f64 * mass / box_size.powi(3);
    let cs = sound_speed(t_init);

    let mut rho_max_ratio = 1.0f64;
    let grid = 10usize;
    let cell = box_size / grid as f64;
    let cell_vol = cell.powi(3);

    for step in 0..steps {
        let mut density_grid = vec![0.0f64; grid.pow(3)];

        for p in &pos {
            let ix = ((p[0] + box_size/2.0) / cell) as usize;
            let iy = ((p[1] + box_size/2.0) / cell) as usize;
            let iz = ((p[2] + box_size/2.0) / cell) as usize;
            if ix < grid && iy < grid && iz < grid {
                density_grid[ix*grid*grid + iy*grid + iz] += mass;
            }
        }

        let rho_max = density_grid.iter().cloned().fold(0.0f64, f64::max) / cell_vol;
        rho_max_ratio = rho_max / mean_density;

        for i in 0..n {
            let ix = ((pos[i][0] + box_size/2.0) / cell) as usize;
            let iy = ((pos[i][1] + box_size/2.0) / cell) as usize;
            let iz = ((pos[i][2] + box_size/2.0) / cell) as usize;
            if ix == 0 || iy == 0 || iz == 0
                || ix >= grid-1 || iy >= grid-1 || iz >= grid-1 {
                continue;
            }

            let idx = |x: usize, y: usize, z: usize| x*grid*grid + y*grid + z;
            let rho_xp = density_grid[idx(ix+1, iy, iz)];
            let rho_xm = density_grid[idx(ix-1, iy, iz)];
            let rho_yp = density_grid[idx(ix, iy+1, iz)];
            let rho_ym = density_grid[idx(ix, iy-1, iz)];
            let rho_zp = density_grid[idx(ix, iy, iz+1)];
            let rho_zm = density_grid[idx(ix, iy, iz-1)];

            // Gravité
            let ax_grav = -g_code * (rho_xp - rho_xm) / (2.0 * cell);
            let ay_grav = -g_code * (rho_yp - rho_ym) / (2.0 * cell);
            let az_grav = -g_code * (rho_zp - rho_zm) / (2.0 * cell);

            // Pression (oppose gravité)
            let rho_i = density_grid[idx(ix, iy, iz)] / cell_vol;
            let press_factor = if rho_i > 0.0 { cs * cs / rho_i } else { 0.0 };
            let ax_press = -press_factor * (rho_xp - rho_xm) / (2.0 * cell);
            let ay_press = -press_factor * (rho_yp - rho_ym) / (2.0 * cell);
            let az_press = -press_factor * (rho_zp - rho_zm) / (2.0 * cell);

            vel[i][0] += (ax_grav + ax_press) * dt;
            vel[i][1] += (ay_grav + ay_press) * dt;
            vel[i][2] += (az_grav + az_press) * dt;
        }

        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
                if pos[i][k] > box_size/2.0 { pos[i][k] -= box_size; }
                if pos[i][k] < -box_size/2.0 { pos[i][k] += box_size; }
            }
        }

        if step % 25 == 0 {
            println!("  step {:4} : ρmax/ρ̄ = {:.3}", step, rho_max_ratio);
        }
    }

    let passed = rho_max_ratio < 5.0;
    if passed {
        println!("  ✓ PASS : ρmax/ρ̄ = {:.3} < 5.0 (stable)", rho_max_ratio);
    } else {
        println!("  ✗ FAIL : ρmax/ρ̄ = {:.3} (instable!)", rho_max_ratio);
    }
    passed
}

// ─────────────────────────────────────────────
// TEST 3 — Conservation de l'énergie
// ─────────────────────────────────────────────
fn test_energy_conservation() -> bool {
    println!("\nTEST 3 — Conservation de l'énergie");
    println!("  N=500, Box=10 Mpc, Steps=50, sans cooling");

    let n = 500usize;
    let box_size = 10.0f64;
    let g_code = 4.499e-15f64;
    let steps = 50usize;
    let dt = 0.001f64;
    let mass = 1e12f64;
    let softening = 0.1f64;

    let mut rng = rand::thread_rng();

    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    // Énergie initiale
    let e_kin_0: f64 = vel.iter()
        .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    let e_pot_0 = compute_potential_energy(&pos, mass, g_code, softening);
    let e_tot_0 = e_kin_0 + e_pot_0;

    // Intégration leapfrog
    for _ in 0..steps {
        // Kick (demi-pas)
        for i in 0..n {
            let mut ax = 0.0;
            let mut ay = 0.0;
            let mut az = 0.0;
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                let r = r2.sqrt();
                let f = g_code * mass / (r * r2);
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            vel[i][0] += ax * dt * 0.5;
            vel[i][1] += ay * dt * 0.5;
            vel[i][2] += az * dt * 0.5;
        }

        // Drift
        for i in 0..n {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }

        // Kick (demi-pas)
        for i in 0..n {
            let mut ax = 0.0;
            let mut ay = 0.0;
            let mut az = 0.0;
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                let r = r2.sqrt();
                let f = g_code * mass / (r * r2);
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            vel[i][0] += ax * dt * 0.5;
            vel[i][1] += ay * dt * 0.5;
            vel[i][2] += az * dt * 0.5;
        }
    }

    let e_kin_f: f64 = vel.iter()
        .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    let e_pot_f = compute_potential_energy(&pos, mass, g_code, softening);
    let e_tot_f = e_kin_f + e_pot_f;

    let variation = if e_tot_0.abs() > 1e-30 {
        (e_tot_f - e_tot_0).abs() / e_tot_0.abs()
    } else { 0.0 };

    println!("  E_tot(0) = {:.4e}", e_tot_0);
    println!("  E_tot(f) = {:.4e}", e_tot_f);
    println!("  Variation = {:.4}%", variation * 100.0);

    let passed = variation < 0.05;
    if passed {
        println!("  ✓ PASS : énergie conservée à {:.2}%", variation * 100.0);
    } else {
        println!("  ✗ FAIL : variation = {:.2}% > 5%", variation * 100.0);
    }
    passed
}

fn compute_potential_energy(pos: &[[f64; 3]], mass: f64, g: f64, soft: f64) -> f64 {
    let n = pos.len();
    let mut e = 0.0f64;
    for i in 0..n {
        for j in (i+1)..n {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r = (dx*dx + dy*dy + dz*dz + soft*soft).sqrt();
            e -= g * mass * mass / r;
        }
    }
    e
}

// ─────────────────────────────────────────────
// TEST 4 — Conservation de l'impulsion
// ─────────────────────────────────────────────
fn test_momentum_conservation() -> bool {
    println!("\nTEST 4 — Conservation de l'impulsion");
    println!("  N=500, Steps=50");

    let n = 500usize;
    let box_size = 10.0f64;
    let g_code = 4.499e-15f64;
    let steps = 50usize;
    let dt = 0.001f64;
    let mass = 1e12f64;
    let softening = 0.1f64;

    let mut rng = rand::thread_rng();

    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    for _ in 0..steps {
        for i in 0..n {
            let mut ax = 0.0;
            let mut ay = 0.0;
            let mut az = 0.0;
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                let r = r2.sqrt();
                let f = g_code * mass / (r * r2);
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            vel[i][0] += ax * dt;
            vel[i][1] += ay * dt;
            vel[i][2] += az * dt;
        }
        for i in 0..n {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }
    }

    let px: f64 = vel.iter().map(|v| mass * v[0]).sum();
    let py: f64 = vel.iter().map(|v| mass * v[1]).sum();
    let pz: f64 = vel.iter().map(|v| mass * v[2]).sum();
    let p_total = (px*px + py*py + pz*pz).sqrt();

    println!("  |P_final| = {:.4e}", p_total);

    let passed = p_total < mass * n as f64 * 1e-6;
    if passed {
        println!("  ✓ PASS : impulsion conservée");
    } else {
        println!("  ✗ FAIL : |P| = {:.4e} (trop grand)", p_total);
    }
    passed
}
