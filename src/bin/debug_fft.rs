//! Debug FFT — Test inverse FFT on trivial input
//!
//! Test: single frequency k=(1,0,0) should give sinusoid after IFFT

use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

const N: usize = 16;  // Small grid for debugging

fn main() {
    println!("=== FFT Debug Test ===\n");

    let n3 = N * N * N;
    let dk = 2.0 * PI / 100.0;  // Box size 100

    // Test 1: Single frequency delta_k
    println!("Test 1: Single frequency k=(1,0,0)");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    // Put amplitude at k=(1,0,0) only
    // For REAL field: also need conjugate at k=(-1,0,0) = (N-1,0,0)
    let idx_k1 = 0 * N * N + 0 * N + 1;  // (ix=1, iy=0, iz=0)
    let idx_km1 = 0 * N * N + 0 * N + (N - 1);  // (ix=N-1, iy=0, iz=0) = k=(-1,0,0)

    delta_k[idx_k1] = Complex::new(1.0, 0.0);   // Real amplitude
    delta_k[idx_km1] = Complex::new(1.0, 0.0);  // Hermitian conjugate

    println!("  delta_k[k=(1,0,0)] = {:?}", delta_k[idx_k1]);
    println!("  delta_k[k=(-1,0,0)] = {:?}", delta_k[idx_km1]);

    let max_delta_k = delta_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  max(|delta_k|) = {:.6}", max_delta_k);

    // Compute psi_x_k = -i * kx * delta_k / k²
    println!("\nComputing displacement field psi_x_k...");
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let half_n = N / 2;

    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let idx = iz * N * N + iy * N + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - N as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - N as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - N as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                if k2 < 1e-20 {
                    continue;
                }

                // psi = -i * k * delta_k / k²
                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
            }
        }
    }

    println!("  psi_x_k[k=(1,0,0)] = {:?}", psi_x_k[idx_k1]);
    println!("  psi_x_k[k=(-1,0,0)] = {:?}", psi_x_k[idx_km1]);

    let max_psi_k = psi_x_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  max(|psi_x_k|) = {:.6}", max_psi_k);

    // Apply IFFT
    println!("\nApplying IFFT...");
    let psi_x = ifft_3d(&mut psi_x_k.clone(), N);

    let max_psi = psi_x.iter().cloned().fold(0.0f64, |a, b| a.abs().max(b.abs()));
    let min_psi = psi_x.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    println!("  max(|psi_x|) = {:.6}", max_psi);
    println!("  min(psi_x) = {:.6}", min_psi);
    println!("  max(psi_x) = {:.6}", psi_x.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b)));

    // Print first few values along x axis
    println!("\n  psi_x along x-axis (iy=0, iz=0):");
    for ix in 0..N {
        let idx = 0 * N * N + 0 * N + ix;
        print!("  [{:2}] = {:+.6}", ix, psi_x[idx]);
        if ix % 4 == 3 { println!(); }
    }
    println!();

    // Test 2: Verify IFFT normalization with DC component
    println!("\n\nTest 2: DC component only");
    let mut dc_field: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    dc_field[0] = Complex::new(1.0, 0.0);

    let dc_result = ifft_3d(&mut dc_field.clone(), N);
    let expected_dc = 1.0 / (n3 as f64);  // After normalization
    println!("  DC input: delta_k[0] = 1.0");
    println!("  After IFFT: delta_x[0] = {:.10}", dc_result[0]);
    println!("  Expected (1/N³): {:.10}", expected_dc);
    println!("  Match: {}", (dc_result[0] - expected_dc).abs() < 1e-10);

    // Test 3: Full pipeline with simple sine wave
    println!("\n\nTest 3: Expected sine wave δ(x) = cos(kx·x)");
    let mut sine_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    // For cos(kx·x): put 0.5 at +kx and 0.5 at -kx
    sine_k[idx_k1] = Complex::new(0.5 * n3 as f64, 0.0);  // Include N³ factor since IFFT divides by N³
    sine_k[idx_km1] = Complex::new(0.5 * n3 as f64, 0.0);

    let sine_result = ifft_3d(&mut sine_k.clone(), N);
    println!("  psi_x along x-axis (should be cos(2π·ix/N)):");
    for ix in 0..N {
        let idx = 0 * N * N + 0 * N + ix;
        let expected = (2.0 * PI * ix as f64 / N as f64).cos();
        print!("  [{:2}] got {:+.4} exp {:+.4}", ix, sine_result[idx], expected);
        if ix % 2 == 1 { println!(); }
    }
    println!();

    println!("\n=== DIAGNOSIS ===");
    if max_psi < 1e-10 {
        println!("BUG CONFIRMED: IFFT output is zero!");
        println!("Check: IFFT normalization, Hermitian symmetry, complex conjugate handling");
    } else {
        println!("IFFT appears to work. Issue may be in amplitude calculation.");
    }
}

/// 3D inverse FFT (same as jour4_filaments.rs)
fn ifft_3d(data: &mut Vec<Complex<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);

    // Process along z
    for iy in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n)
                .map(|iz| data[iz * n * n + iy * n + ix])
                .collect();
            ifft.process(&mut slice);
            for iz in 0..n {
                data[iz * n * n + iy * n + ix] = slice[iz];
            }
        }
    }

    // Process along y
    for iz in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n)
                .map(|iy| data[iz * n * n + iy * n + ix])
                .collect();
            ifft.process(&mut slice);
            for iy in 0..n {
                data[iz * n * n + iy * n + ix] = slice[iy];
            }
        }
    }

    // Process along x
    for iz in 0..n {
        for iy in 0..n {
            let base = iz * n * n + iy * n;
            let mut slice: Vec<Complex<f64>> = data[base..base+n].to_vec();
            ifft.process(&mut slice);
            for ix in 0..n {
                data[base + ix] = slice[ix];
            }
        }
    }

    // Extract real part and normalize
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}
