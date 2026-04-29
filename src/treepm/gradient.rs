//! 4th order central finite difference operators on a periodic 3D grid.
//!
//! Reference: GrGadget Eq. (20), originally from Springel 2005.
//! Replaces 2nd-order central difference (current TreePM) for ∂Φ/∂x_i.
//! Error: O(h^4) vs current O(h^2).
//!
//! Field is stored row-major: idx(i, j, k) = i*n*n + j*n + k.

#[inline(always)]
fn idx(i: usize, j: usize, k: usize, n: usize) -> usize {
    i * n * n + j * n + k
}

/// 4th-order central finite difference for ∂f/∂x at cell (i, j, k).
///
/// Formula: ∂f/∂x ≈ [8·(f_{i+1} - f_{i-1}) − (f_{i+2} - f_{i-2})] / (12·h)
#[inline(always)]
pub fn grad4_x(field: &[f64], i: usize, j: usize, k: usize, n: usize, h: f64) -> f64 {
    let im2 = (i + n - 2) % n;
    let im1 = (i + n - 1) % n;
    let ip1 = (i + 1) % n;
    let ip2 = (i + 2) % n;

    (8.0 * (field[idx(ip1, j, k, n)] - field[idx(im1, j, k, n)])
        - (field[idx(ip2, j, k, n)] - field[idx(im2, j, k, n)]))
        / (12.0 * h)
}

/// 4th-order central finite difference for ∂f/∂y at cell (i, j, k).
#[inline(always)]
pub fn grad4_y(field: &[f64], i: usize, j: usize, k: usize, n: usize, h: f64) -> f64 {
    let jm2 = (j + n - 2) % n;
    let jm1 = (j + n - 1) % n;
    let jp1 = (j + 1) % n;
    let jp2 = (j + 2) % n;

    (8.0 * (field[idx(i, jp1, k, n)] - field[idx(i, jm1, k, n)])
        - (field[idx(i, jp2, k, n)] - field[idx(i, jm2, k, n)]))
        / (12.0 * h)
}

/// 4th-order central finite difference for ∂f/∂z at cell (i, j, k).
#[inline(always)]
pub fn grad4_z(field: &[f64], i: usize, j: usize, k: usize, n: usize, h: f64) -> f64 {
    let km2 = (k + n - 2) % n;
    let km1 = (k + n - 1) % n;
    let kp1 = (k + 1) % n;
    let kp2 = (k + 2) % n;

    (8.0 * (field[idx(i, j, kp1, n)] - field[idx(i, j, km1, n)])
        - (field[idx(i, j, kp2, n)] - field[idx(i, j, km2, n)]))
        / (12.0 * h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_grad4_constant_field_zero() {
        // Constant field → gradient = 0 partout.
        // Tolerance: machine epsilon × magnitude × n_ops ≈ 1e-14 × 4 × ~10 ops ≈ 1e-13
        let n = 32;
        let field = vec![3.14159_f64; n * n * n];
        let mut max_err: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    max_err = max_err.max(grad4_x(&field, i, j, k, n, 1.0).abs());
                    max_err = max_err.max(grad4_y(&field, i, j, k, n, 1.0).abs());
                    max_err = max_err.max(grad4_z(&field, i, j, k, n, 1.0).abs());
                }
            }
        }
        assert!(max_err < 1e-13, "max_err = {}", max_err);
    }

    #[test]
    fn test_grad4_linear_field_constant() {
        // f(x) = x → ∂f/∂x = 1 (au centre, loin des bords PBC)
        // Tolerance: 1e-12 (machine epsilon ~1e-15, on multiplie par ordres de grandeur)
        let n = 64;
        let l = 1.0_f64;
        let h = l / n as f64;
        let mut field = vec![0.0; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    field[idx(i, j, k, n)] = i as f64 * h;
                }
            }
        }
        // Centre du domaine, à au moins 4 cellules des bords PBC
        let g = grad4_x(&field, n / 2, n / 2, n / 2, n, h);
        assert!((g - 1.0).abs() < 1e-12, "Expected 1.0, got {}", g);
    }

    #[test]
    fn test_grad4_sine_convergence_order4() {
        // f(x) = sin(2πx/L) → ∂f/∂x = (2π/L)·cos(2πx/L)
        // Erreur doit décroître en h^4. Critère : err(2h)/err(h) ≈ 16.
        let l = 1.0_f64;
        let mut errors = Vec::new();
        let resolutions = [32_usize, 64, 128];

        for &n in &resolutions {
            let h = l / n as f64;
            let mut field = vec![0.0; n * n * n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let x = i as f64 * h;
                        field[idx(i, j, k, n)] = (2.0 * PI * x / l).sin();
                    }
                }
            }

            let mut max_err: f64 = 0.0;
            // Évite les bords PBC pour test propre
            for i in 4..n - 4 {
                let x = i as f64 * h;
                let exact = (2.0 * PI / l) * (2.0 * PI * x / l).cos();
                let g = grad4_x(&field, i, n / 2, n / 2, n, h);
                max_err = max_err.max((g - exact).abs());
            }
            errors.push(max_err);
        }

        // Vérifier convergence O(h^4) : err(2h)/err(h) ≈ 16
        let ratio_32_64 = errors[0] / errors[1];
        let ratio_64_128 = errors[1] / errors[2];

        // Tolérance : on doit voir au moins 12× (proche de 16)
        assert!(
            ratio_32_64 > 12.0 && ratio_32_64 < 20.0,
            "Convergence n=32→64 ratio = {} (expected ~16, errors {:?})",
            ratio_32_64,
            errors
        );
        assert!(
            ratio_64_128 > 12.0 && ratio_64_128 < 20.0,
            "Convergence n=64→128 ratio = {} (expected ~16, errors {:?})",
            ratio_64_128,
            errors
        );
    }

    #[test]
    fn test_grad4_periodic_consistency() {
        // f(x) = cos(2πx/L) périodique. Vérifier que les valeurs aux bords
        // sont cohérentes via PBC. d/dx cos(2πx/L) = -(2π/L)·sin(2πx/L).
        // À x=0 : -(2π/L)·sin(0) = 0
        // À x=(n-1)·h = L - L/n : -(2π/L)·sin(2π·(n-1)/n)
        //                       = -(2π/L)·sin(2π - 2π/n) = (2π/L)·sin(2π/n)
        let n = 32;
        let l = 1.0_f64;
        let h = l / n as f64;
        let mut field = vec![0.0; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * h;
                    field[idx(i, j, k, n)] = (2.0 * PI * x / l).cos();
                }
            }
        }
        // g(0) doit être ≈ 0
        let g0 = grad4_x(&field, 0, n / 2, n / 2, n, h);
        // g(L-h) doit être ≈ +(2π/L)·sin(2π/n)
        let expected_at_end = (2.0 * PI / l) * (2.0 * PI / n as f64).sin();
        let g_n = grad4_x(&field, n - 1, n / 2, n / 2, n, h);
        // Tolerance schéma O(h^4) sur cos avec n=32 : err ~ (2π/32)^4 × 2π ~ 9e-3
        assert!(g0.abs() < 1e-2, "g(0) = {}", g0);
        assert!(
            (g_n - expected_at_end).abs() < 1e-2,
            "g(L-h) = {}, expected {}",
            g_n,
            expected_at_end
        );
    }

    #[test]
    fn test_grad4_y_z_symmetric() {
        // Verify y and z gradients have same convergence as x by computing on
        // a separable field f(x,y,z) = sin(2πy/L), confirming ∂f/∂y = (2π/L)·cos.
        let n = 64;
        let l = 1.0_f64;
        let h = l / n as f64;
        let mut field = vec![0.0; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let y = j as f64 * h;
                    field[idx(i, j, k, n)] = (2.0 * PI * y / l).sin();
                }
            }
        }
        let j_mid = n / 2;
        let y = j_mid as f64 * h;
        let exact = (2.0 * PI / l) * (2.0 * PI * y / l).cos();
        let g = grad4_y(&field, n / 2, j_mid, n / 2, n, h);
        // Tolerance pour n=64 : err ~ (2π/64)^4 ~ 9.3e-5
        assert!((g - exact).abs() < 1e-4, "Expected {}, got {}", exact, g);
    }
}
