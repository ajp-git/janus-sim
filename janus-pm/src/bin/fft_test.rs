/// PM-1: FFT Round-trip Test
/// Validates: FFT → IFFT reconstruction error < 1e-4
///
/// Uses rustfft (CPU) for algorithm validation.
/// GPU CuFFT will be added later for production 512³ grids.

use rustfft::{FftPlanner, num_complex::Complex};
use std::time::Instant;

/// 3D FFT using rustfft (row-major order)
/// Performs 1D FFT along each dimension sequentially
fn fft_3d(data: &mut [Complex<f32>], nx: usize, ny: usize, nz: usize, inverse: bool) {
    let mut planner = FftPlanner::new();

    // FFT along Z (innermost, contiguous)
    let fft_z = if inverse { planner.plan_fft_inverse(nz) } else { planner.plan_fft_forward(nz) };
    for ix in 0..nx {
        for iy in 0..ny {
            let start = ix * ny * nz + iy * nz;
            let slice = &mut data[start..start + nz];
            fft_z.process(slice);
        }
    }

    // FFT along Y
    let fft_y = if inverse { planner.plan_fft_inverse(ny) } else { planner.plan_fft_forward(ny) };
    let mut buffer = vec![Complex::new(0.0, 0.0); ny];
    for ix in 0..nx {
        for iz in 0..nz {
            // Gather Y-line
            for iy in 0..ny {
                buffer[iy] = data[ix * ny * nz + iy * nz + iz];
            }
            fft_y.process(&mut buffer);
            // Scatter Y-line
            for iy in 0..ny {
                data[ix * ny * nz + iy * nz + iz] = buffer[iy];
            }
        }
    }

    // FFT along X (outermost)
    let fft_x = if inverse { planner.plan_fft_inverse(nx) } else { planner.plan_fft_forward(nx) };
    let mut buffer = vec![Complex::new(0.0, 0.0); nx];
    for iy in 0..ny {
        for iz in 0..nz {
            // Gather X-line
            for ix in 0..nx {
                buffer[ix] = data[ix * ny * nz + iy * nz + iz];
            }
            fft_x.process(&mut buffer);
            // Scatter X-line
            for ix in 0..nx {
                data[ix * ny * nz + iy * nz + iz] = buffer[ix];
            }
        }
    }

    // Normalize for inverse FFT
    if inverse {
        let norm = 1.0 / (nx * ny * nz) as f32;
        for val in data.iter_mut() {
            *val *= norm;
        }
    }
}

/// Create 3D Gaussian centered in the grid
fn create_gaussian(nx: usize, ny: usize, nz: usize, sigma: f32) -> Vec<Complex<f32>> {
    let mut data = vec![Complex::new(0.0, 0.0); nx * ny * nz];
    let cx = nx as f32 / 2.0;
    let cy = ny as f32 / 2.0;
    let cz = nz as f32 / 2.0;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let dx = ix as f32 - cx;
                let dy = iy as f32 - cy;
                let dz = iz as f32 - cz;
                let r2 = dx * dx + dy * dy + dz * dz;
                let val = (-r2 * inv_2sigma2).exp();
                data[ix * ny * nz + iy * nz + iz] = Complex::new(val, 0.0);
            }
        }
    }
    data
}

/// Compute max absolute error between two arrays
fn max_error(a: &[Complex<f32>], b: &[Complex<f32>]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f32, |acc, e| acc.max(e))
}

/// Compute RMS error
fn rms_error(a: &[Complex<f32>], b: &[Complex<f32>]) -> f32 {
    let sum_sq: f32 = a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum();
    (sum_sq / a.len() as f32).sqrt()
}

fn test_fft_roundtrip(n: usize) -> (f32, f32, f64) {
    println!("\n=== Testing {}³ grid ({} elements) ===", n, n * n * n);

    // Create Gaussian input
    let sigma = n as f32 / 8.0;
    let original = create_gaussian(n, n, n, sigma);
    let mut data = original.clone();

    println!("  Gaussian σ = {:.1}", sigma);
    println!("  Max value: {:.6}", original.iter().map(|c| c.re).fold(0.0_f32, f32::max));

    // Forward FFT
    let t0 = Instant::now();
    fft_3d(&mut data, n, n, n, false);
    let t_fft = t0.elapsed();

    // Inverse FFT
    let t1 = Instant::now();
    fft_3d(&mut data, n, n, n, true);
    let t_ifft = t1.elapsed();

    let total_ms = (t_fft + t_ifft).as_secs_f64() * 1000.0;

    // Compute errors
    let max_err = max_error(&original, &data);
    let rms_err = rms_error(&original, &data);

    println!("  FFT time:  {:.2} ms", t_fft.as_secs_f64() * 1000.0);
    println!("  IFFT time: {:.2} ms", t_ifft.as_secs_f64() * 1000.0);
    println!("  Total:     {:.2} ms", total_ms);
    println!("  Max error: {:.2e}", max_err);
    println!("  RMS error: {:.2e}", rms_err);

    (max_err, rms_err, total_ms)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-1: FFT Round-trip Validation (rustfft CPU)                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Test 64³
    let (max_err_64, _, time_64) = test_fft_roundtrip(64);

    // Test 128³
    let (max_err_128, _, time_128) = test_fft_roundtrip(128);

    // Test 256³
    let (max_err_256, _, time_256) = test_fft_roundtrip(256);

    // Validation summary
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      VALIDATION SUMMARY                          ");
    println!("══════════════════════════════════════════════════════════════════\n");

    let err_threshold = 1e-4;
    let time_threshold = 500.0; // ms for 256³

    let err_pass = max_err_256 < err_threshold;
    let time_pass = time_256 < time_threshold;

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Test                    │ Result    │ Threshold │ Status       │");
    println!("├─────────────────────────┼───────────┼───────────┼──────────────┤");
    println!("│ 64³ reconstruction err  │ {:.2e}  │ < 1e-4    │ {}           │",
             max_err_64, if max_err_64 < err_threshold { "✓ PASS" } else { "✗ FAIL" });
    println!("│ 128³ reconstruction err │ {:.2e}  │ < 1e-4    │ {}           │",
             max_err_128, if max_err_128 < err_threshold { "✓ PASS" } else { "✗ FAIL" });
    println!("│ 256³ reconstruction err │ {:.2e}  │ < 1e-4    │ {}           │",
             max_err_256, if max_err_256 < err_threshold { "✓ PASS" } else { "✗ FAIL" });
    println!("│ 256³ FFT+IFFT time      │ {:.1} ms   │ < 500 ms  │ {}           │",
             time_256, if time_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n══════════════════════════════════════════════════════════════════");
    if err_pass && time_pass {
        println!("PM-1 VALIDATION: ✓ PASSED");
        println!("  Reconstruction error: {:.2e} < 1e-4", max_err_256);
        println!("  FFT 256³ time: {:.1} ms < 500 ms", time_256);
    } else {
        println!("PM-1 VALIDATION: ✗ FAILED");
        if !err_pass {
            println!("  ✗ Reconstruction error {:.2e} >= 1e-4", max_err_256);
        }
        if !time_pass {
            println!("  ✗ FFT time {:.1} ms >= 500 ms (CPU expected, GPU needed)", time_256);
        }
    }
    println!("══════════════════════════════════════════════════════════════════");

    println!("\nNote: Using rustfft (CPU). For 512³ production grids,");
    println!("      CuFFT (GPU) will be needed for acceptable performance.");
}
