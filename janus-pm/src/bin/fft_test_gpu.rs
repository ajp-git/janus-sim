/// PM-1b: GPU FFT Round-trip Test (CuFFT)
/// Validates: FFT → IFFT reconstruction error < 1e-4, time < 500ms
///
/// Uses direct CuFFT FFI bindings for C2C 3D transforms.

use janus_pm::cufft::Cufft3dC2C;
use janus_pm::cufft_ffi::CufftComplex;
use std::time::Instant;

/// Create 3D Gaussian centered in the grid
fn create_gaussian(nx: usize, ny: usize, nz: usize, sigma: f32) -> Vec<CufftComplex> {
    let mut data = vec![CufftComplex::default(); nx * ny * nz];
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
                data[ix * ny * nz + iy * nz + iz] = CufftComplex::new(val, 0.0);
            }
        }
    }
    data
}

/// Compute max absolute error between two arrays
fn max_error(a: &[CufftComplex], b: &[CufftComplex]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| {
            let dx = x.x - y.x;
            let dy = x.y - y.y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0_f32, f32::max)
}

/// Compute RMS error
fn rms_error(a: &[CufftComplex], b: &[CufftComplex]) -> f32 {
    let sum_sq: f32 = a.iter().zip(b.iter())
        .map(|(x, y)| {
            let dx = x.x - y.x;
            let dy = x.y - y.y;
            dx * dx + dy * dy
        })
        .sum();
    (sum_sq / a.len() as f32).sqrt()
}

fn test_fft_roundtrip(n: usize) -> Result<(f32, f32, f64), String> {
    println!("\n=== Testing {}³ grid ({} elements) ===", n, n * n * n);

    // Create FFT plan (includes GPU memory allocation)
    let t_plan = Instant::now();
    let plan = Cufft3dC2C::new(n, n, n)?;
    println!("  Plan creation: {:.2} ms", t_plan.elapsed().as_secs_f64() * 1000.0);

    // Create Gaussian input
    let sigma = n as f32 / 8.0;
    let original = create_gaussian(n, n, n, sigma);
    let mut data = original.clone();

    println!("  Gaussian σ = {:.1}", sigma);
    println!("  Max value: {:.6}", original.iter().map(|c| c.x).fold(0.0_f32, f32::max));

    // Execute round-trip (forward + inverse + normalize)
    let t0 = Instant::now();
    plan.roundtrip(&mut data)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Compute errors
    let max_err = max_error(&original, &data);
    let rms_err = rms_error(&original, &data);

    println!("  FFT+IFFT time: {:.2} ms", total_ms);
    println!("  Max error: {:.2e}", max_err);
    println!("  RMS error: {:.2e}", rms_err);

    Ok((max_err, rms_err, total_ms))
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-1b: FFT Round-trip Validation (CuFFT GPU)                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Test 64³
    let result_64 = test_fft_roundtrip(64);
    let (max_err_64, _, time_64) = match &result_64 {
        Ok(r) => (r.0, r.1, r.2),
        Err(e) => {
            println!("\n✗ 64³ test FAILED: {}", e);
            std::process::exit(1);
        }
    };

    // Test 128³
    let result_128 = test_fft_roundtrip(128);
    let (max_err_128, _, time_128) = match &result_128 {
        Ok(r) => (r.0, r.1, r.2),
        Err(e) => {
            println!("\n✗ 128³ test FAILED: {}", e);
            std::process::exit(1);
        }
    };

    // Test 256³
    let result_256 = test_fft_roundtrip(256);
    let (max_err_256, _, time_256) = match &result_256 {
        Ok(r) => (r.0, r.1, r.2),
        Err(e) => {
            println!("\n✗ 256³ test FAILED: {}", e);
            std::process::exit(1);
        }
    };

    // Test 512³ (production size)
    let result_512 = test_fft_roundtrip(512);
    let (max_err_512, _, time_512) = match &result_512 {
        Ok(r) => (r.0, r.1, r.2),
        Err(e) => {
            println!("\n✗ 512³ test FAILED: {}", e);
            // Not a hard failure - just report
            (f32::NAN, f32::NAN, f64::NAN)
        }
    };

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
    println!("│ 256³ FFT+IFFT time      │ {:>6.1} ms │ < 500 ms  │ {}           │",
             time_256, if time_pass { "✓ PASS" } else { "✗ FAIL" });
    if !time_512.is_nan() {
        println!("│ 512³ FFT+IFFT time      │ {:>6.1} ms │ (info)    │ (prod size)  │", time_512);
    }
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n══════════════════════════════════════════════════════════════════");
    if err_pass && time_pass {
        println!("PM-1b VALIDATION: ✓ PASSED");
        println!("  Reconstruction error: {:.2e} < 1e-4", max_err_256);
        println!("  FFT 256³ time: {:.1} ms < 500 ms", time_256);
        println!("  Speedup vs CPU (rustfft): ~{:.0}×", 1424.0 / time_256);
    } else {
        println!("PM-1b VALIDATION: ✗ FAILED");
        if !err_pass {
            println!("  ✗ Reconstruction error {:.2e} >= 1e-4", max_err_256);
        }
        if !time_pass {
            println!("  ✗ FFT time {:.1} ms >= 500 ms", time_256);
        }
    }
    println!("══════════════════════════════════════════════════════════════════");
}
