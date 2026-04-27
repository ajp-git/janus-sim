//! SPH Kernel Functions
//!
//! Gaussian kernel: W(r, h) = (1/π^(3/2) h³) × exp(-r²/h²)

use std::f64::consts::PI;

pub struct SphKernel;

impl SphKernel {
    /// Gaussian SPH kernel
    /// W(r, h) = (1/π^(3/2) h³) × exp(-r²/h²)
    pub fn w(r: f64, h: f64) -> f64 {
        let q = r / h;
        (1.0 / (PI.powf(1.5) * h.powi(3))) * (-q * q).exp()
    }

    /// Gradient of kernel (radial component)
    /// dW/dr = -2r/h² × W(r, h)
    pub fn dw_dr(r: f64, h: f64) -> f64 {
        -2.0 * r / (h * h) * Self::w(r, h)
    }

    /// Adaptive smoothing length
    /// h = η × (m / ρ)^(1/3)
    pub fn smoothing_length(mass: f64, density: f64, eta: f64) -> f64 {
        eta * (mass / density).powf(1.0 / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_normalization() {
        // ∫W(r,h)dV = 1
        let h = 1.0;
        let n_shells = 1000;
        let r_max = 5.0 * h;
        let dr = r_max / n_shells as f64;
        let integral: f64 = (0..n_shells).map(|i| {
            let r = (i as f64 + 0.5) * dr;
            4.0 * PI * r * r * SphKernel::w(r, h) * dr
        }).sum();
        assert!((integral - 1.0).abs() < 0.01,
            "Kernel non normalisé: intégrale = {}", integral);
    }

    #[test]
    fn test_kernel_positive() {
        for r in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] {
            assert!(SphKernel::w(r, 1.0) >= 0.0,
                "Kernel négatif à r={}", r);
        }
    }

    #[test]
    fn test_kernel_decreasing() {
        let h = 1.0;
        let w0 = SphKernel::w(0.0, h);
        let w1 = SphKernel::w(0.5, h);
        let w2 = SphKernel::w(1.0, h);
        assert!(w0 > w1 && w1 > w2, "Kernel non décroissant");
    }

    #[test]
    fn test_gradient_negative() {
        // dW/dr < 0 (kernel décroissant)
        let dw = SphKernel::dw_dr(0.5, 1.0);
        assert!(dw < 0.0, "Gradient doit être négatif");
    }

    #[test]
    fn test_smoothing_length_scaling() {
        // h ∝ ρ^(-1/3) → ρ×8 donne h/2
        let mass = 1.0e10;
        let h1 = SphKernel::smoothing_length(mass, 1.0e-3, 1.2);
        let h2 = SphKernel::smoothing_length(mass, 8.0e-3, 1.2);
        assert!((h1 / h2 - 2.0).abs() < 0.01,
            "Scaling h incorrect: ratio = {}", h1/h2);
    }
}
