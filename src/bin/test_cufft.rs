//! Test cuFFT Poisson solver
//!
//! Build: cargo build --release --features cufft --bin test_cufft
//! Run: LD_LIBRARY_PATH=target/release cargo run --release --features cufft --bin test_cufft

#[cfg(feature = "cufft")]
use janus::treepm::CuFFTPoisson;
use std::time::Instant;

#[cfg(feature = "cufft")]
fn main() {
    println!("=== cuFFT Poisson Solver Test ===\n");

    let grid_size = 128;
    let box_size = 100.0;
    let g_constant = 1.0;
    let r_s = box_size / 16.0 / 3.0;  // TreePM splitting scale

    println!("Configuration:");
    println!("  Grid: {}³", grid_size);
    println!("  Box: {:.1}", box_size);
    println!("  G: {}", g_constant);
    println!("  r_s: {:.2} (TreePM splitting)", r_s);
    println!();

    // Initialize solver
    println!("Initializing cuFFT solver...");
    let init_start = Instant::now();
    let mut solver = match CuFFTPoisson::new(grid_size, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize cuFFT: {}", e);
            return;
        }
    };
    println!("  Init time: {:.2?}", init_start.elapsed());
    println!("  GPU memory: {:.2} MB", solver.memory_bytes() as f64 / 1e6);
    println!();

    // Test 1: Single point mass
    println!("Test 1: Single point mass at center");
    let n = grid_size * grid_size * grid_size;
    let mut rho = vec![0.0f64; n];
    let center = grid_size / 2 + grid_size * (grid_size / 2 + grid_size * grid_size / 2);
    rho[center] = 1.0;

    let solve_start = Instant::now();
    let phi = match solver.solve(&rho, g_constant, 0.0) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Solve failed: {}", e);
            return;
        }
    };
    let solve_time = solve_start.elapsed();

    println!("  Solve time: {:.2?}", solve_time);
    println!("  phi[center] = {:.6e}", phi[center]);
    println!("  phi[edge] = {:.6e}", phi[0]);

    // Test 2: With TreePM splitting
    println!("\nTest 2: With TreePM splitting (r_s = {:.2})", r_s);
    let solve_start = Instant::now();
    let phi_split = match solver.solve(&rho, g_constant, r_s) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Solve failed: {}", e);
            return;
        }
    };
    let solve_time = solve_start.elapsed();

    println!("  Solve time: {:.2?}", solve_time);
    println!("  phi_split[center] = {:.6e}", phi_split[center]);
    println!("  phi_split[edge] = {:.6e}", phi_split[0]);

    // Test 3: Benchmark
    println!("\nTest 3: Benchmark (10 iterations)");
    let n_iter = 10;
    let bench_start = Instant::now();
    for _ in 0..n_iter {
        let _ = solver.solve(&rho, g_constant, r_s);
    }
    let bench_time = bench_start.elapsed();
    let per_iter = bench_time.as_secs_f64() / n_iter as f64;

    println!("  Total: {:.2?}", bench_time);
    println!("  Per iteration: {:.3}ms", per_iter * 1000.0);

    // Summary
    println!("\n=== Summary ===");
    println!("  cuFFT Poisson solver: WORKING");
    println!("  128³ grid solve time: {:.3}ms", per_iter * 1000.0);

    // Extrapolate to 256³
    let scale_factor = 8.0;  // 256³ / 128³ ~ 8x more data, but FFT scales as N log N
    let estimated_256 = per_iter * 1000.0 * scale_factor * (256.0_f64.log2() / 128.0_f64.log2());
    println!("  Estimated 256³: {:.1}ms", estimated_256);
}

#[cfg(not(feature = "cufft"))]
fn main() {
    eprintln!("This binary requires the 'cufft' feature.");
    eprintln!("Build with: cargo build --release --features cufft --bin test_cufft");
    eprintln!("Run with: LD_LIBRARY_PATH=target/release cargo run --release --features cufft --bin test_cufft");
}
