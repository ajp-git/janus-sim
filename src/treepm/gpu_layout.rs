//! Structure-of-Arrays (SoA) particle layout for GPU coalesced memory access.
//!
//! Reference: PhotoNs-GPU §3.1, Wang & Meng 2021.
//!
//! On GPU, threads in a warp access memory coalesced only when data is
//! contiguous. Storing particles as separate arrays per attribute (SoA)
//! instead of array of structures (AoS) gives ~40% reduction in memory
//! transfer time for typical N-body kernels.
//!
//! Convention de précision (cf plan §3.0):
//! - pos_x/y/z: f64 (DP) — accumulation cosmologique sur ~10^4 steps
//! - vel_x/y/z: f32 (SP) — reset partiel à chaque kick
//! - acc_x/y/z: f32 (SP) — output kernel GPU
//! - mass: f32 (SP) — toujours positive (valeur absolue)
//! - sign: i8 — +1 (m+) ou -1 (m-)

/// SoA particle arrays for GPU pipeline.
///
/// Mass is stored as ABSOLUTE value (always > 0).
/// Sign is stored separately as i8 (±1) to avoid negative-mass FP issues.
#[derive(Debug, Clone)]
pub struct ParticleArrays {
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub pos_z: Vec<f64>,

    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,

    pub acc_x: Vec<f32>,
    pub acc_y: Vec<f32>,
    pub acc_z: Vec<f32>,

    pub mass: Vec<f32>, // |m|, always > 0
    pub sign: Vec<i8>,  // +1 for m+, -1 for m-

    pub n: usize,
}

impl ParticleArrays {
    /// Create new empty arrays of capacity `n`, all values zeroed.
    pub fn new(n: usize) -> Self {
        Self {
            pos_x: vec![0.0; n],
            pos_y: vec![0.0; n],
            pos_z: vec![0.0; n],
            vel_x: vec![0.0; n],
            vel_y: vec![0.0; n],
            vel_z: vec![0.0; n],
            acc_x: vec![0.0; n],
            acc_y: vec![0.0; n],
            acc_z: vec![0.0; n],
            mass: vec![0.0; n],
            sign: vec![0; n],
            n,
        }
    }

    /// Convert AoS positions [(x, y, z), ...] to SoA arrays.
    pub fn from_aos_positions(aos: &[(f64, f64, f64)]) -> Self {
        let n = aos.len();
        let mut s = Self::new(n);
        for (i, &(x, y, z)) in aos.iter().enumerate() {
            s.pos_x[i] = x;
            s.pos_y[i] = y;
            s.pos_z[i] = z;
        }
        s
    }

    /// Convert from per-particle data (positions DP, sign, mass).
    /// Velocities and accelerations initialized to zero.
    pub fn from_particles(pos: &[(f64, f64, f64)], sign: &[i8], mass: &[f32]) -> Self {
        let n = pos.len();
        assert_eq!(sign.len(), n);
        assert_eq!(mass.len(), n);
        let mut s = Self::from_aos_positions(pos);
        s.sign.copy_from_slice(sign);
        s.mass.copy_from_slice(mass);
        s
    }

    /// Convert SoA positions back to AoS [(x, y, z), ...].
    pub fn to_aos_positions(&self) -> Vec<(f64, f64, f64)> {
        (0..self.n)
            .map(|i| (self.pos_x[i], self.pos_y[i], self.pos_z[i]))
            .collect()
    }

    /// Memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let n = self.n;
        // 3×f64 (pos) + 3×f32 (vel) + 3×f32 (acc) + f32 (mass) + i8 (sign)
        n * (3 * 8 + 3 * 4 + 3 * 4 + 4 + 1)
    }

    /// Reset accelerations to zero (between force computations).
    pub fn reset_acc(&mut self) {
        self.acc_x.fill(0.0);
        self.acc_y.fill(0.0);
        self.acc_z.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aos_to_soa_roundtrip() {
        let n = 1000;
        let aos: Vec<(f64, f64, f64)> =
            (0..n).map(|i| (i as f64, (i * 2) as f64, (i * 3) as f64)).collect();

        let soa = ParticleArrays::from_aos_positions(&aos);
        assert_eq!(soa.pos_x.len(), n);
        assert_eq!(soa.n, n);

        let back = soa.to_aos_positions();
        for i in 0..n {
            // Tolerance: 1e-15 (exact f64 roundtrip, no quantization)
            assert!((aos[i].0 - back[i].0).abs() < 1e-15);
            assert!((aos[i].1 - back[i].1).abs() < 1e-15);
            assert!((aos[i].2 - back[i].2).abs() < 1e-15);
        }
    }

    #[test]
    fn test_soa_full_init() {
        let pos = vec![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)];
        let sign = vec![1_i8, -1_i8];
        let mass = vec![1.0_f32, 2.5_f32];
        let s = ParticleArrays::from_particles(&pos, &sign, &mass);
        assert_eq!(s.n, 2);
        assert_eq!(s.pos_x[0], 1.0);
        assert_eq!(s.pos_y[1], 5.0);
        assert_eq!(s.sign[0], 1);
        assert_eq!(s.sign[1], -1);
        assert_eq!(s.mass[0], 1.0);
        assert_eq!(s.mass[1], 2.5);
    }

    #[test]
    fn test_position_dp_to_sp_roundtrip_loss() {
        // Positions cosmologiques typiques: x ∈ [0, 1000] Mpc
        // Conversion DP → SP → DP doit conserver précision suffisante.
        // Tolerance: 1e-4 Mpc (epsilon SP × x_max ≈ 1.2e-7 × 1000 ≈ 1.2e-4)
        let positions_dp: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
        let positions_sp: Vec<f32> = positions_dp.iter().map(|&x| x as f32).collect();
        let positions_back: Vec<f64> = positions_sp.iter().map(|&x| x as f64).collect();

        let max_err = positions_dp
            .iter()
            .zip(positions_back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-4, "DP→SP→DP loss = {} Mpc", max_err);
    }

    #[test]
    fn test_memory_footprint() {
        let n = 1_000_000;
        let s = ParticleArrays::new(n);
        // 3*8 + 3*4 + 3*4 + 4 + 1 = 53 B per particle
        let expected = n * 53;
        assert_eq!(s.memory_bytes(), expected);
        // For 1M particles: ~53 MB
        assert!(s.memory_bytes() < 60 * 1024 * 1024);
    }

    #[test]
    fn test_reset_acc() {
        let mut s = ParticleArrays::new(10);
        s.acc_x[3] = 1.5;
        s.acc_y[7] = -2.0;
        s.acc_z[9] = 0.1;
        s.reset_acc();
        for i in 0..10 {
            assert_eq!(s.acc_x[i], 0.0);
            assert_eq!(s.acc_y[i], 0.0);
            assert_eq!(s.acc_z[i], 0.0);
        }
    }
}
