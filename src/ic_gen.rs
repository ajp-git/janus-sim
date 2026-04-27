//! Zel'dovich Initial Conditions Generator
//!
//! Generates cosmological initial conditions using the Zel'dovich approximation.
//! Supports configurable box size, grid resolution, and cosmological parameters.

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Parameters for IC generation
#[derive(Debug, Clone)]
pub struct IcParams {
    pub box_size: f64,      // Mpc
    pub n_grid: usize,      // Particles per dimension per species
    pub z_init: f64,        // Initial redshift
    pub seed: u64,          // Random seed
    pub mu: f64,            // Mass ratio m-/m+ (Janus)
    pub n_s: f64,           // Spectral index
    pub delta_rms: f64,     // Initial overdensity RMS
}

impl Default for IcParams {
    fn default() -> Self {
        Self {
            box_size: 50.0,
            n_grid: 128,
            z_init: 10.0,
            seed: 42,
            mu: 19.0,
            n_s: 0.965,
            delta_rms: 0.10,
        }
    }
}

/// Result of IC generation
pub struct IcResult {
    pub positions: Vec<f64>,   // Flat [x0,y0,z0, x1,y1,z1, ...]
    pub velocities: Vec<f64>,  // Flat [vx0,vy0,vz0, ...]
    pub signs: Vec<i32>,       // +1 for m+, -1 for m-
    pub n_plus: usize,
    pub n_minus: usize,
}

/// Generate Zel'dovich ICs with random sign assignment on SINGLE GRID
///
/// All particles on one grid, signs randomly shuffled.
/// Minimum distance between ANY pair = spacing (guaranteed).
/// This is the correct approach matching the main production runs.
pub fn generate_zeldovich_ics(params: &IcParams) -> IcResult {
    let n_grid = params.n_grid;
    let n3 = n_grid * n_grid * n_grid;
    let n_total = n3;  // Single grid: one particle per cell

    // Fraction of positive particles: f+ = 1/(1+μ)
    // n_positive = n3 / (1 + μ)
    let n_positive = (n3 as f64 / (1.0 + params.mu)).round() as usize;
    let n_negative = n_total - n_positive;

    println!("  Grid: {}³ = {} particles (single grid)", n_grid, n3);
    println!("  Total: {} (N+ = {}, N- = {})", n_total, n_positive, n_negative);
    println!("  Box: {} Mpc, z_init = {}", params.box_size, params.z_init);
    println!("  Seed: {}, n_s = {}, δ_rms = {}", params.seed, params.n_s, params.delta_rms);

    let dk = 2.0 * PI / params.box_size;
    let half_n = n_grid / 2;
    let spacing = params.box_size / n_grid as f64;
    let half_box = params.box_size / 2.0;

    println!("  Min distance (any pair) = spacing = {:.3} Mpc", spacing);

    let a_init = 1.0 / (1.0 + params.z_init);
    let d_growth = a_init;  // Linear growth factor approximation

    // Generate displacement field
    println!("  Generating displacement field...");
    let (psi_x, psi_y, psi_z) = generate_displacement_field(
        n_grid, dk, half_n, d_growth, params.seed, params.n_s, params.delta_rms
    );

    // Scale displacements to ~30% of cell size
    let max_disp = find_max_displacement(&psi_x, &psi_y, &psi_z);
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };

    println!("  Max displacement: {:.6e} Mpc → scale = {:.4}", max_disp, scale);

    // Zel'dovich velocity: v = dD/dt × ψ = H(z) × f(z) × ψ_scaled
    let mpc_gyr_to_kms = 977.8;
    let h0_gyr = 69.9 / mpc_gyr_to_kms;  // H₀ = 70 km/s/Mpc → 0.0715 Gyr⁻¹
    let vel_scale = h0_gyr * (1.0 + params.z_init).sqrt();
    println!("  Velocity scale: {:.4e} Mpc/Gyr ({:.1} km/s/Mpc)",
             vel_scale, vel_scale * mpc_gyr_to_kms);

    // Single grid with random sign assignment
    // 1. Create vector of indices [0..n3]
    // 2. Shuffle with fixed seed
    // 3. First n_positive indices → m+ (sign = +1)
    // 4. Remaining indices → m- (sign = -1)
    println!("  Placing {} particles...", n_total);

    let mut rng_shuffle = StdRng::seed_from_u64(params.seed + 12345);
    let mut particle_indices: Vec<usize> = (0..n3).collect();
    particle_indices.shuffle(&mut rng_shuffle);

    // Create sign array: first n_positive shuffled indices are m+
    let mut sign_by_cell = vec![-1i32; n3];
    for &idx in &particle_indices[..n_positive] {
        sign_by_cell[idx] = 1;
    }

    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs: Vec<i32> = Vec::with_capacity(n_total);

    // Place particles on single grid
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let grid_idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let mut x = x0 + psi_x[grid_idx] * scale;
                let mut y = y0 + psi_y[grid_idx] * scale;
                let mut z = z0 + psi_z[grid_idx] * scale;

                // Periodic boundary
                while x > half_box { x -= params.box_size; }
                while x < -half_box { x += params.box_size; }
                while y > half_box { y -= params.box_size; }
                while y < -half_box { y += params.box_size; }
                while z > half_box { z -= params.box_size; }
                while z < -half_box { z += params.box_size; }

                positions.push(x);
                positions.push(y);
                positions.push(z);

                // Zel'dovich: v = H*f*ψ_scaled
                velocities.push(psi_x[grid_idx] * scale * vel_scale);
                velocities.push(psi_y[grid_idx] * scale * vel_scale);
                velocities.push(psi_z[grid_idx] * scale * vel_scale);

                signs.push(sign_by_cell[grid_idx]);
            }
        }
    }

    // Validation
    let n_plus_actual = signs.iter().filter(|&&s| s > 0).count();
    let n_minus_actual = signs.iter().filter(|&&s| s < 0).count();
    println!("  ✓ Generated: N+ = {}, N- = {}", n_plus_actual, n_minus_actual);

    // Check positions in box
    let all_in_box = positions.chunks(3).all(|p| {
        p[0].abs() <= half_box && p[1].abs() <= half_box && p[2].abs() <= half_box
    });
    if all_in_box {
        println!("  ✓ All positions in [{:.0}, {:.0}] Mpc", -half_box, half_box);
    } else {
        println!("  ⚠ Some positions outside box!");
    }

    IcResult {
        positions,
        velocities,
        signs,
        n_plus: n_plus_actual,
        n_minus: n_minus_actual,
    }
}

/// Generate 3D displacement field using FFT
fn generate_displacement_field(
    n_grid: usize,
    dk: f64,
    half_n: usize,
    d_growth: f64,
    seed: u64,
    n_s: f64,
    delta_rms: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Gaussian random field in Fourier space
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                if k2 > 0.0 {
                    let k = k2.sqrt();
                    // Power spectrum P(k) ∝ k^(n_s - 4) for displacements
                    let pk = k.powf(n_s - 4.0) * delta_rms.powi(2);
                    let amp = (pk / 2.0).sqrt();

                    let phase = rng.gen::<f64>() * 2.0 * PI;
                    let re = normal.sample(&mut rng) * amp * d_growth;
                    let im = normal.sample(&mut rng) * amp * d_growth;

                    let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                    delta_k[idx] = Complex::new(
                        re * phase.cos() - im * phase.sin(),
                        re * phase.sin() + im * phase.cos()
                    );
                }
            }
        }
    }

    // IFFT to get displacement field components
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    let mut psi_x = vec![0.0f64; n3];
    let mut psi_y = vec![0.0f64; n3];
    let mut psi_z = vec![0.0f64; n3];

    // ψ_i = IFFT[-i k_i δ(k) / k²]
    for dim in 0..3 {
        let mut field_k = delta_k.clone();

        for iz in 0..n_grid {
            for iy in 0..n_grid {
                for ix in 0..n_grid {
                    let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                    let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                    let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;
                    let k2 = kx*kx + ky*ky + kz*kz;

                    let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                    if k2 > 0.0 {
                        let k_i = match dim { 0 => kx, 1 => ky, _ => kz };
                        // -i k_i / k²
                        let factor = Complex::new(0.0, -k_i / k2);
                        field_k[idx] = field_k[idx] * factor;
                    } else {
                        field_k[idx] = Complex::new(0.0, 0.0);
                    }
                }
            }
        }

        // 3D IFFT via 1D IFFTs
        // X direction
        for iz in 0..n_grid {
            for iy in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|ix| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for ix in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[ix];
                }
            }
        }

        // Y direction
        for iz in 0..n_grid {
            for ix in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|iy| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for iy in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[iy];
                }
            }
        }

        // Z direction
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|iz| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for iz in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[iz];
                }
            }
        }

        // Extract real part
        let psi = match dim {
            0 => &mut psi_x,
            1 => &mut psi_y,
            _ => &mut psi_z,
        };
        for i in 0..n3 {
            psi[i] = field_k[i].re / n3 as f64;
        }
    }

    (psi_x, psi_y, psi_z)
}

/// Find maximum displacement magnitude
fn find_max_displacement(psi_x: &[f64], psi_y: &[f64], psi_z: &[f64]) -> f64 {
    let n = psi_x.len();
    let mut max_disp = 0.0f64;
    for i in 0..n {
        let d = (psi_x[i].powi(2) + psi_y[i].powi(2) + psi_z[i].powi(2)).sqrt();
        if d > max_disp { max_disp = d; }
    }
    max_disp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ic_generation_small() {
        let params = IcParams {
            box_size: 50.0,
            n_grid: 4,
            z_init: 10.0,
            seed: 42,
            mu: 19.0,
            n_s: 0.965,
            delta_rms: 0.10,
        };

        let result = generate_zeldovich_ics(&params);

        // Check particle counts
        let n_expected = 2 * 4 * 4 * 4;  // 2 × 4³ = 128
        assert_eq!(result.signs.len(), n_expected);
        assert_eq!(result.positions.len(), n_expected * 3);
        assert_eq!(result.velocities.len(), n_expected * 3);

        // Check ratio approximately 1:19
        let ratio = result.n_minus as f64 / result.n_plus as f64;
        assert!((ratio - 19.0).abs() < 2.0, "Ratio should be ~19, got {}", ratio);
    }

    #[test]
    fn test_positions_in_box() {
        let params = IcParams {
            box_size: 50.0,
            n_grid: 4,
            ..Default::default()
        };

        let result = generate_zeldovich_ics(&params);
        let half_box = params.box_size / 2.0;

        for chunk in result.positions.chunks(3) {
            assert!(chunk[0].abs() <= half_box, "x out of bounds: {}", chunk[0]);
            assert!(chunk[1].abs() <= half_box, "y out of bounds: {}", chunk[1]);
            assert!(chunk[2].abs() <= half_box, "z out of bounds: {}", chunk[2]);
        }
    }
}
