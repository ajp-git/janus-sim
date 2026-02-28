//! Diagnostic: Compare analytical vs exact PE_binding
//!
//! The analytical formula uses mean nearest-neighbor distance:
//!   mean_sep = 0.554 × L / N^(1/3)
//!   PE = -G × m² × N_pairs / mean_sep
//!
//! The exact O(N²) calculates:
//!   PE = -Σ G × m² / r_ij for all same-sign pairs
//!
//! If these differ significantly, the analytical formula is wrong.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("=== Analytical vs Exact PE_binding ===\n");

    let n_total = 10_000;  // Small for quick O(N²)
    let n_positive = n_total / 2;
    let n_negative = n_total - n_positive;
    let box_size = 100.0_f64;
    let mass = 1.0_f64;
    let g_code = 1.0_f64;
    let softening = 0.1_f64;

    println!("Parameters:");
    println!("  N = {} ({} each sign)", n_total, n_total / 2);
    println!("  box = {} Mpc", box_size);
    println!("  m = {}, G = {}", mass, g_code);
    println!();

    // Generate random uniform positions
    let mut rng = StdRng::seed_from_u64(42);
    let half = box_size / 2.0;

    let mut pos_data: Vec<f64> = Vec::with_capacity(n_total * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n_total);

    for i in 0..n_total {
        let x = rng.random::<f64>() * box_size - half;
        let y = rng.random::<f64>() * box_size - half;
        let z = rng.random::<f64>() * box_size - half;
        pos_data.extend([x, y, z]);
        signs.push(if i < n_positive { 1 } else { -1 });
    }

    // === ANALYTICAL PE ===
    println!("=== Analytical Formula ===");
    let mean_sep_plus = 0.554 * box_size / (n_positive as f64).cbrt();
    let mean_sep_minus = 0.554 * box_size / (n_negative as f64).cbrt();
    let pe_plus_analytical = -g_code * mass * mass
        * ((n_positive * (n_positive - 1)) / 2) as f64 / mean_sep_plus;
    let pe_minus_analytical = -g_code * mass * mass
        * ((n_negative * (n_negative - 1)) / 2) as f64 / mean_sep_minus;
    let pe_analytical = pe_plus_analytical + pe_minus_analytical;

    println!("  mean_sep (nearest neighbor) = {:.3} Mpc", mean_sep_plus);
    println!("  N_pairs (each sign) = {}", n_positive * (n_positive - 1) / 2);
    println!("  PE_+ analytical = {:.4e}", pe_plus_analytical);
    println!("  PE_- analytical = {:.4e}", pe_minus_analytical);
    println!("  PE_binding analytical = {:.4e}", pe_analytical);
    println!();

    // === EXACT O(N²) PE ===
    println!("=== Exact O(N²) ===");
    let mut pe_exact = 0.0_f64;
    let mut n_pairs_counted = 0u64;
    let mut sum_r = 0.0_f64;
    let mut sum_inv_r = 0.0_f64;
    let soft_sq = softening * softening;

    for i in 0..n_total {
        let xi = pos_data[i * 3];
        let yi = pos_data[i * 3 + 1];
        let zi = pos_data[i * 3 + 2];
        let si = signs[i];

        for j in (i + 1)..n_total {
            if signs[j] != si { continue; }

            let xj = pos_data[j * 3];
            let yj = pos_data[j * 3 + 1];
            let zj = pos_data[j * 3 + 2];

            // Minimum image convention
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            if dx > half { dx -= box_size; } else if dx < -half { dx += box_size; }
            if dy > half { dy -= box_size; } else if dy < -half { dy += box_size; }
            if dz > half { dz -= box_size; } else if dz < -half { dz += box_size; }

            let r_sq = dx*dx + dy*dy + dz*dz;
            let r = r_sq.sqrt();
            let r_soft = (r_sq + soft_sq).sqrt();

            pe_exact -= g_code * mass * mass / r_soft;
            n_pairs_counted += 1;
            sum_r += r;
            sum_inv_r += 1.0 / r_soft;
        }
    }

    let mean_r = sum_r / n_pairs_counted as f64;
    let mean_inv_r = sum_inv_r / n_pairs_counted as f64;

    println!("  N_pairs counted = {}", n_pairs_counted);
    println!("  mean(r) = {:.3} Mpc", mean_r);
    println!("  mean(1/r) = {:.6}", mean_inv_r);
    println!("  1/mean(r) = {:.6}", 1.0 / mean_r);
    println!("  Ratio: mean(1/r) / (1/mean(r)) = {:.3}", mean_inv_r * mean_r);
    println!();
    println!("  PE_binding exact = {:.4e}", pe_exact);
    println!();

    // === COMPARISON ===
    println!("=== Comparison ===");
    let ratio = pe_analytical / pe_exact;
    println!("  PE_analytical / PE_exact = {:.2}", ratio);
    println!();

    if ratio > 10.0 {
        println!("CRITICAL: Analytical overestimates PE by {:.0}x!", ratio);
        println!();
        println!("The analytical formula uses mean NEAREST-NEIGHBOR distance (~{:.1} Mpc)",
                 mean_sep_plus);
        println!("But the actual mean ALL-PAIRS distance is ~{:.1} Mpc", mean_r);
        println!();
        println!("This means:");
        println!("  - Analytical α ≈ {:.2}", (pe_analytical.abs() / (2.0 * 12500.0_f64)).sqrt());
        println!("  - Exact α ≈ {:.2}", (pe_exact.abs() / (2.0 * 12500.0_f64)).sqrt());
        println!();
        println!("The reference BH run used WRONG analytical virialization!");
        println!("S_max=0.694 was from gravitational collapse, not Janus physics.");
    }
}
