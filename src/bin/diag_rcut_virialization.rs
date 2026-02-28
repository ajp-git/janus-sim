//! Diagnostic: Compare virialization parameters
//!
//! Hypothesis: r_cut/box_size is too small → few pairs → PE underestimated → α too large
//!
//! Compare:
//!   - Reference: N=2M, α=4.83
//!   - Current: N=100K, α=48.9
//!
//! Test different r_cut values for virialization

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn compute_pe_binding_with_rcut(
    pos_data: &[f32],
    signs_data: &[i8],
    box_size: f64,
    r_cut: f64,
    softening: f64,
) -> (f64, usize, usize) {
    let n_total = signs_data.len();
    let half_box = box_size / 2.0;
    let soft_sq = softening * softening;
    let r_cut_sq = r_cut * r_cut;
    let mass = 1.0_f64;
    let g_code = 1.0_f64;

    let mut pe_binding = 0.0_f64;
    let mut n_pairs_rcut = 0usize;
    let mut n_pairs_total = 0usize;

    for i in 0..n_total {
        let xi = pos_data[i * 3] as f64;
        let yi = pos_data[i * 3 + 1] as f64;
        let zi = pos_data[i * 3 + 2] as f64;
        let si = signs_data[i];

        for j in (i + 1)..n_total {
            if signs_data[j] != si { continue; }
            n_pairs_total += 1;

            let xj = pos_data[j * 3] as f64;
            let yj = pos_data[j * 3 + 1] as f64;
            let zj = pos_data[j * 3 + 2] as f64;

            // Minimum image convention
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
            if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
            if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

            let r_sq = dx*dx + dy*dy + dz*dz;

            // Count pairs within r_cut
            if r_sq < r_cut_sq {
                n_pairs_rcut += 1;
            }

            // PE uses ALL pairs (no r_cut)
            let r_soft = (r_sq + soft_sq).sqrt();
            pe_binding -= g_code * mass * mass / r_soft;
        }
    }

    (pe_binding, n_pairs_rcut, n_pairs_total)
}

fn main() {
    println!("=== Diagnostic: r_cut and Virialization ===\n");

    // Test different box sizes and N
    let configs = [
        ("Small (N=10K, box=100)", 10_000usize, 100.0_f64),
        ("Medium (N=50K, box=100)", 50_000, 100.0),
        ("Current (N=100K, box=100)", 100_000, 100.0),
    ];

    let softening = 0.1_f64;

    for (name, n_total, box_size) in configs {
        println!("=== {} ===", name);
        println!("  N = {}, box = {} Mpc", n_total, box_size);

        // Generate random uniform positions
        let mut rng = StdRng::seed_from_u64(42);
        let half = box_size / 2.0;

        let mut pos_data: Vec<f32> = Vec::with_capacity(n_total * 3);
        let mut vel_data: Vec<f32> = Vec::with_capacity(n_total * 3);
        let mut signs_data: Vec<i8> = vec![-1i8; n_total];

        for _ in 0..n_total {
            let x = rng.random::<f64>() * box_size - half;
            let y = rng.random::<f64>() * box_size - half;
            let z = rng.random::<f64>() * box_size - half;
            pos_data.extend([x as f32, y as f32, z as f32]);

            let v_init = 1.0;
            let vx = (rng.random::<f64>() - 0.5) * v_init;
            let vy = (rng.random::<f64>() - 0.5) * v_init;
            let vz = (rng.random::<f64>() - 0.5) * v_init;
            vel_data.extend([vx as f32, vy as f32, vz as f32]);
        }

        // Assign signs (first half +, second half -)
        for i in 0..(n_total / 2) {
            signs_data[i] = 1;
        }

        // Compute KE
        let mass = 1.0_f64;
        let ke: f64 = vel_data.chunks(3)
            .map(|v| 0.5 * mass * (v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64 + v[2] as f64 * v[2] as f64))
            .sum();

        println!("  KE_initial = {:.4e}", ke);
        println!();

        // Test different r_cut values
        let r_cut_fractions = [
            ("box/4", box_size / 4.0),
            ("box/8", box_size / 8.0),
            ("box/16", box_size / 16.0),
            ("full (no r_cut)", box_size * 10.0),  // Effectively no cutoff
        ];

        println!("  {:15} {:>12} {:>15} {:>12} {:>10}",
                 "r_cut", "r_cut (Mpc)", "pairs r<r_cut", "PE_binding", "α");
        println!("  {}", "-".repeat(70));

        for (label, r_cut) in r_cut_fractions {
            let (pe_binding, n_pairs_rcut, _n_pairs_total) =
                compute_pe_binding_with_rcut(&pos_data, &signs_data, box_size, r_cut, softening);

            let ke_target = pe_binding.abs() / 2.0;
            let alpha = if ke > 1e-20 { (ke_target / ke).sqrt() } else { 1.0 };

            let pairs_per_particle = n_pairs_rcut as f64 / n_total as f64;

            println!("  {:15} {:>12.2} {:>15.1} {:>12.4e} {:>10.2}",
                     label, r_cut, pairs_per_particle, pe_binding, alpha);
        }
        println!();
    }

    println!("=== Analysis ===");
    println!();
    println!("If PE_binding uses r_cut restriction:");
    println!("  - box/16: Very few pairs → PE underestimated → α too large");
    println!("  - box/8:  More pairs → Better PE estimate → α closer to expected");
    println!("  - No r_cut: All pairs → Full PE → Correct α");
    println!();
    println!("Reference α = 4.83 suggests PE_binding was computed with ALL pairs (no r_cut).");
    println!("If current code uses r_cut in PE calculation, that's the bug.");
}
