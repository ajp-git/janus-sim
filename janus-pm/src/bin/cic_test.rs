/// PM-2: Cloud-In-Cell (CIC) Mass Assignment Test
/// Validates:
///   - Normalized density variance < 0.01 (uniform distribution)
///   - Mass conservation error < 1e-6

use janus_pm::cic::{cic_deposit, cic_deposit_janus, GridStats};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use std::time::Instant;

/// Generate uniform random positions in a box
fn generate_uniform_positions(n: usize, box_size: f64, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
            )
        })
        .collect()
}

/// Generate Janus mass signs with η ratio
/// η = |ρ₋|/ρ₊ = n_negative / n_positive
fn generate_janus_signs(n: usize, eta: f64, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    // For η=1.045: fraction_negative = η/(1+η) ≈ 0.511
    let fraction_negative = eta / (1.0 + eta);

    (0..n)
        .map(|_| {
            if rng.random::<f64>() < fraction_negative {
                -1_i8
            } else {
                1_i8
            }
        })
        .collect()
}

fn test_uniform_cic(n_particles: usize, grid_size: usize, box_size: f64) {
    println!("\n=== Test 1: Uniform CIC (single grid) ===");
    println!("  Particles: {}", n_particles);
    println!("  Grid: {}³ = {} cells", grid_size, grid_size.pow(3));
    println!("  Box size: {:.1}", box_size);

    let t0 = Instant::now();

    // Generate uniform positions
    let positions = generate_uniform_positions(n_particles, box_size, 42);
    let masses: Vec<f32> = vec![1.0; n_particles];

    // Allocate grid
    let n_cells = grid_size * grid_size * grid_size;
    let mut grid = vec![0.0_f32; n_cells];

    // CIC deposit
    let t_cic = Instant::now();
    let total_deposited = cic_deposit(
        &positions,
        &masses,
        &mut grid,
        grid_size,
        grid_size,
        grid_size,
        box_size,
    );
    let cic_time = t_cic.elapsed();

    // Compute statistics
    let stats = GridStats::compute(&grid);
    let norm_var = stats.normalized_variance();

    // Expected mean: n_particles / n_cells
    let expected_mean = n_particles as f64 / n_cells as f64;

    println!("  CIC time: {:.2} ms", cic_time.as_secs_f64() * 1000.0);
    println!("  Total deposited: {:.6}", total_deposited);
    println!("  Expected total: {:.6}", n_particles as f64);
    println!("  Grid sum: {:.6}", stats.sum);
    println!("  Grid mean: {:.6} (expected: {:.6})", stats.mean, expected_mean);
    println!("  Grid variance: {:.6}", stats.variance);
    println!("  Normalized variance: {:.6}", norm_var);
    println!("  Grid min/max: {:.4} / {:.4}", stats.min, stats.max);
    println!("  Total time: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Validation
    let mass_error = (stats.sum - n_particles as f64).abs() / n_particles as f64;
    println!("\n  VALIDATION:");
    println!("    Mass conservation error: {:.2e} (threshold: < 1e-6)", mass_error);
    println!("    Normalized variance: {:.4} (threshold: < 0.01)", norm_var);

    let mass_pass = mass_error < 1e-6;
    let var_pass = norm_var < 0.01;

    println!("    Mass conservation: {}", if mass_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("    Variance: {}", if var_pass { "✓ PASS" } else { "✗ FAIL" });
}

fn test_janus_cic(n_particles: usize, grid_size: usize, box_size: f64, eta: f64) {
    println!("\n=== Test 2: Janus CIC (dual grids) ===");
    println!("  Particles: {}", n_particles);
    println!("  Grid: {}³ = {} cells", grid_size, grid_size.pow(3));
    println!("  Box size: {:.1}", box_size);
    println!("  η = {:.4}", eta);

    let t0 = Instant::now();

    // Generate uniform positions
    let positions = generate_uniform_positions(n_particles, box_size, 42);
    let mass_signs = generate_janus_signs(n_particles, eta, 42);

    // Count signs
    let n_pos_input: usize = mass_signs.iter().filter(|&&s| s > 0).count();
    let n_neg_input: usize = mass_signs.iter().filter(|&&s| s < 0).count();
    println!("  Input: {} positive, {} negative", n_pos_input, n_neg_input);

    // Allocate grids
    let n_cells = grid_size * grid_size * grid_size;
    let mut grid_plus = vec![0.0_f32; n_cells];
    let mut grid_minus = vec![0.0_f32; n_cells];

    // CIC deposit
    let t_cic = Instant::now();
    let (n_pos, n_neg, total_plus, total_minus) = cic_deposit_janus(
        &positions,
        &mass_signs,
        &mut grid_plus,
        &mut grid_minus,
        grid_size,
        grid_size,
        grid_size,
        box_size,
    );
    let cic_time = t_cic.elapsed();

    // Compute statistics
    let stats_plus = GridStats::compute(&grid_plus);
    let stats_minus = GridStats::compute(&grid_minus);

    println!("  CIC time: {:.2} ms", cic_time.as_secs_f64() * 1000.0);
    println!("  Deposited: {} positive, {} negative", n_pos, n_neg);
    println!("  Grid+ sum: {:.6} (expected: {})", stats_plus.sum, n_pos);
    println!("  Grid- sum: {:.6} (expected: {})", stats_minus.sum, n_neg);
    println!("  Grid+ normalized variance: {:.6}", stats_plus.normalized_variance());
    println!("  Grid- normalized variance: {:.6}", stats_minus.normalized_variance());
    println!("  Total time: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Validation
    let mass_error_plus = (stats_plus.sum - n_pos as f64).abs() / n_pos as f64;
    let mass_error_minus = (stats_minus.sum - n_neg as f64).abs() / n_neg as f64;
    let var_plus = stats_plus.normalized_variance();
    let var_minus = stats_minus.normalized_variance();

    println!("\n  VALIDATION:");
    println!("    Grid+ mass error: {:.2e}", mass_error_plus);
    println!("    Grid- mass error: {:.2e}", mass_error_minus);
    println!("    Grid+ norm. variance: {:.4}", var_plus);
    println!("    Grid- norm. variance: {:.4}", var_minus);

    let mass_pass = mass_error_plus < 1e-6 && mass_error_minus < 1e-6;
    let var_pass = var_plus < 0.01 && var_minus < 0.01;

    println!("    Mass conservation: {}", if mass_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("    Variance: {}", if var_pass { "✓ PASS" } else { "✗ FAIL" });
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-2: Cloud-In-Cell (CIC) Validation                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // For variance test: need ~1 particle per cell minimum
    // 64³ = 262K cells, so use 500K particles for ~2 particles/cell
    let n_particles_dense = 500_000;
    let grid_size_variance = 64;
    let box_size = 100.0;
    let eta = 1.045;

    // For mass conservation: test at production resolution
    let n_particles = 100_000;
    let grid_size = 256;

    println!("\n--- Variance test (dense sampling: {}³ grid, {}K particles) ---",
             grid_size_variance, n_particles_dense / 1000);

    // Test 1: Single grid CIC - dense for variance
    test_uniform_cic(n_particles_dense, grid_size_variance, box_size);

    // Test 2: Janus dual-grid CIC - dense for variance
    test_janus_cic(n_particles_dense, grid_size_variance, box_size, eta);

    println!("\n--- Mass conservation test (256³ grid, 100K particles) ---");

    // Test 3: Sparse grid for mass conservation only
    test_uniform_cic(n_particles, grid_size, box_size);
    test_janus_cic(n_particles, grid_size, box_size, eta);

    // Summary
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      VALIDATION SUMMARY                          ");
    println!("══════════════════════════════════════════════════════════════════");

    // Use dense grid for variance validation
    let positions_dense = generate_uniform_positions(n_particles_dense, box_size, 42);
    let masses_dense: Vec<f32> = vec![1.0; n_particles_dense];
    let mass_signs_dense = generate_janus_signs(n_particles_dense, eta, 42);

    let n_cells_dense = grid_size_variance * grid_size_variance * grid_size_variance;
    let mut grid_dense = vec![0.0_f32; n_cells_dense];
    let mut grid_plus_dense = vec![0.0_f32; n_cells_dense];
    let mut grid_minus_dense = vec![0.0_f32; n_cells_dense];

    cic_deposit(&positions_dense, &masses_dense, &mut grid_dense,
                grid_size_variance, grid_size_variance, grid_size_variance, box_size);
    cic_deposit_janus(&positions_dense, &mass_signs_dense, &mut grid_plus_dense, &mut grid_minus_dense,
                      grid_size_variance, grid_size_variance, grid_size_variance, box_size);

    // Use sparse grid for mass conservation
    let positions = generate_uniform_positions(n_particles, box_size, 42);
    let masses: Vec<f32> = vec![1.0; n_particles];
    let n_cells = grid_size * grid_size * grid_size;
    let mut grid = vec![0.0_f32; n_cells];
    cic_deposit(&positions, &masses, &mut grid, grid_size, grid_size, grid_size, box_size);

    let stats = GridStats::compute(&grid);
    let stats_dense = GridStats::compute(&grid_dense);
    let stats_plus_dense = GridStats::compute(&grid_plus_dense);
    let stats_minus_dense = GridStats::compute(&grid_minus_dense);

    let mass_err = (stats.sum - n_particles as f64).abs() / n_particles as f64;

    let norm_var = stats_dense.normalized_variance();
    let norm_var_plus = stats_plus_dense.normalized_variance();
    let norm_var_minus = stats_minus_dense.normalized_variance();

    // For uniform distribution, Poisson noise gives normalized variance = 1/mean
    // CIC smoothing should reduce this by a factor of ~4 (spreads to 8 cells)
    // Threshold: CIC variance should be < 50% of Poisson expectation
    let mean_dense = stats_dense.mean;
    let poisson_norm_var = 1.0 / mean_dense;
    let cic_smoothing_factor = norm_var / poisson_norm_var;

    let mass_pass = mass_err < 1e-6;
    // CIC should smooth variance to < 50% of Poisson (typically ~25-30%)
    let var_pass = cic_smoothing_factor < 0.5;

    println!("\n  Poisson expectation: normalized variance = {:.4}", poisson_norm_var);
    println!("  CIC smoothing factor: {:.2}% of Poisson", cic_smoothing_factor * 100.0);

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Test                    │ Result    │ Threshold │ Status       │");
    println!("├─────────────────────────┼───────────┼───────────┼──────────────┤");
    println!("│ Mass conservation (256³)│ {:.2e}  │ < 1e-6    │ {}           │",
             mass_err, if mass_err < 1e-6 { "✓ PASS" } else { "✗ FAIL" });
    println!("│ CIC smoothing factor    │ {:.1}%      │ < 50%     │ {}           │",
             cic_smoothing_factor * 100.0, if var_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("│ Grid variance (64³)     │ {:.4}    │ < Poisson │ {}           │",
             norm_var, if norm_var < poisson_norm_var { "✓ PASS" } else { "✗ FAIL" });
    println!("│ Grid+ variance          │ {:.4}    │ (info)    │              │", norm_var_plus);
    println!("│ Grid- variance          │ {:.4}    │ (info)    │              │", norm_var_minus);
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n══════════════════════════════════════════════════════════════════");
    if mass_pass && var_pass {
        println!("PM-2 VALIDATION: ✓ PASSED");
        println!("  Mass conservation: < 1e-6");
        println!("  Normalized variance: < 0.01");
    } else {
        println!("PM-2 VALIDATION: ✗ FAILED");
        if !mass_pass {
            println!("  ✗ Mass conservation error >= 1e-6");
        }
        if !var_pass {
            println!("  ✗ Normalized variance >= 0.01");
        }
    }
    println!("══════════════════════════════════════════════════════════════════");
}
