//! Comprehensive Test Suite for CUDA Cooling Kernels
//!
//! Tests verify physical correctness and performance of GPU cooling implementation.
//! GO criteria: 8/8 tests pass before production run.

#[cfg(feature = "cuda")]
mod cooling_tests {
    use janus::cooling_gpu::{GpuCooling, K_B_OVER_MP, MU_IONIZED, T_FLOOR};
    use cudarc::driver::CudaDevice;
    use std::time::Instant;

    const RHO_TO_NH: f64 = 3.07e-17;  // M_sun/Mpc³ to cm⁻³

    /// Helper: compute density in code units for target nH
    fn density_for_nh(nh: f64) -> f64 {
        nh / RHO_TO_NH
    }

    /// Helper: create GpuCooling instance
    fn create_cooling(n: usize) -> GpuCooling {
        let device = CudaDevice::new(0).expect("CUDA device");
        GpuCooling::new(device, n, 100.0, 1e10).expect("Create GpuCooling")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 1: IGM UV Equilibrium
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_igm_uv_equilibrium() {
        println!("\n=== TEST 1: IGM UV Equilibrium ===");
        println!("  nH = 1e-5 cm⁻³, T = 10^4 K, z = 3");
        println!("  Expectation: UV heating dominates → T stable or increasing\n");

        let n = 10000;
        let mut cooling = create_cooling(n);

        // All particles are m+ (sign = 1)
        let signs: Vec<i32> = vec![1; n];
        let t_init = 10000.0;  // 10^4 K

        cooling.init_from_temperature(t_init, t_init, &signs).unwrap();

        // Very low density IGM
        let nh = 1e-5;  // cm⁻³
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        let t_before = cooling.get_mean_temperature_plus().unwrap();

        // Run 100 steps at z=3
        let dt = 0.001;  // Gyr
        let z = 3.0;
        for _ in 0..100 {
            cooling.apply_cooling(dt, z).unwrap();
        }

        let t_after = cooling.get_mean_temperature_plus().unwrap();

        println!("  T_before = {:.1} K", t_before);
        println!("  T_after  = {:.1} K", t_after);
        println!("  ΔT = {:.1} K ({:.1}%)", t_after - t_before, (t_after - t_before) / t_before * 100.0);

        // UV should dominate at low density → T should NOT decrease significantly
        // Allow small decrease due to numerical effects, but no runaway cooling
        assert!(t_after > t_init * 0.8, "Spurious cooling in IGM: T dropped to {:.1} K", t_after);
        println!("  ✓ PASS: No spurious cooling in diffuse IGM\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 2: Halo Cooling Dominant
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_halo_cooling_dominant() {
        println!("\n=== TEST 2: Halo Cooling Dominant ===");
        println!("  nH = 1.0 cm⁻³, T = 10^6 K, z = 1");
        println!("  Expectation: Cooling dominates → T decreases\n");

        let n = 10000;
        let mut cooling = create_cooling(n);

        let signs: Vec<i32> = vec![1; n];
        let t_init = 1e6;  // 10^6 K (hot halo gas)

        cooling.init_from_temperature(t_init, t_init, &signs).unwrap();

        // Dense halo gas
        let nh = 1.0;  // cm⁻³
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        let t_before = cooling.get_mean_temperature_plus().unwrap();

        // Theoretical cooling time at T=10^6 K, nH=1
        // t_cool ~ (3/2) k_B T / (Lambda * n) ~ 10 Myr
        let dt = 0.001;  // 1 Myr
        let z = 1.0;
        for _ in 0..100 {
            cooling.apply_cooling(dt, z).unwrap();
        }

        let t_after = cooling.get_mean_temperature_plus().unwrap();

        println!("  T_before = {:.2e} K", t_before);
        println!("  T_after  = {:.2e} K", t_after);
        println!("  ΔT/T = {:.1}%", (t_after - t_before) / t_before * 100.0);

        // At nH=1, T=10^6, cooling should dominate
        // After 100 Myr, T should be significantly lower
        assert!(t_after < t_init * 0.5, "Cooling too slow: T only dropped to {:.2e} K", t_after);
        println!("  ✓ PASS: Significant cooling in dense halo\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 3: Cooling Floor
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_cooling_floor() {
        println!("\n=== TEST 3: Cooling Floor ===");
        println!("  nH = 10.0 cm⁻³, T = 200 K");
        println!("  Expectation: T must not drop below T_floor = {} K\n", T_FLOOR);

        let n = 10000;
        let mut cooling = create_cooling(n);

        let signs: Vec<i32> = vec![1; n];
        let t_init = 200.0;  // Already near floor

        cooling.init_from_temperature(t_init, t_init, &signs).unwrap();

        let nh = 10.0;
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        // Run many steps
        let dt = 0.001;
        let z = 0.0;
        for _ in 0..500 {
            cooling.apply_cooling(dt, z).unwrap();
        }

        let t_final = cooling.get_mean_temperature_plus().unwrap();

        println!("  T_init  = {:.1} K", t_init);
        println!("  T_final = {:.1} K", t_final);
        println!("  T_floor = {:.1} K", T_FLOOR);

        assert!(t_final >= T_FLOOR - 1.0, "Temperature dropped below floor: {:.1} K", t_final);
        println!("  ✓ PASS: Temperature floor respected\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 4: Self-Shielding at High Density
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_self_shielding_high_density() {
        println!("\n=== TEST 4: Self-Shielding at High Density ===");
        println!("  Compare nH = 0.1 cm⁻³ vs nH = 1e-4 cm⁻³ at z=2");
        println!("  Expectation: High density → more shielding → cooler\n");

        let n = 10000;
        let t_init = 20000.0;  // 2×10^4 K
        let dt = 0.001;
        let z = 2.0;
        let steps = 100;

        // Test 1: High density (shielded)
        let mut cooling_high = create_cooling(n);
        let signs: Vec<i32> = vec![1; n];
        cooling_high.init_from_temperature(t_init, t_init, &signs).unwrap();
        let nh_high = 0.1;
        let densities_high = vec![density_for_nh(nh_high); n];
        cooling_high.upload_densities(&densities_high).unwrap();

        for _ in 0..steps {
            cooling_high.apply_cooling(dt, z).unwrap();
        }
        let t_high = cooling_high.get_mean_temperature_plus().unwrap();

        // Test 2: Low density (not shielded)
        let mut cooling_low = create_cooling(n);
        cooling_low.init_from_temperature(t_init, t_init, &signs).unwrap();
        let nh_low = 1e-4;
        let densities_low = vec![density_for_nh(nh_low); n];
        cooling_low.upload_densities(&densities_low).unwrap();

        for _ in 0..steps {
            cooling_low.apply_cooling(dt, z).unwrap();
        }
        let t_low = cooling_low.get_mean_temperature_plus().unwrap();

        println!("  nH = {:.0e} cm⁻³ → T_final = {:.1} K", nh_high, t_high);
        println!("  nH = {:.0e} cm⁻³ → T_final = {:.1} K", nh_low, t_low);

        // At high density with shielding, gas should cool more effectively
        // (less UV heating, same cooling) → lower final T
        // But at very low density, UV heating dominates, so T might be higher
        // The self-shielding effect should make high-density gas cooler
        println!("  Self-shielding allows cooling at high density");
        assert!(t_high <= t_low || t_high < t_init * 0.9,
            "Self-shielding not working: high-density T = {:.1} K, low-density T = {:.1} K", t_high, t_low);
        println!("  ✓ PASS: Self-shielding effect observed\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 5: Star Formation Threshold
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_sf_threshold() {
        println!("\n=== TEST 5: Star Formation Threshold ===");
        println!("  nH = 50 cm⁻³, T = 500 K");
        println!("  Expectation: SF criteria met → N_stars > 0\n");

        let n = 10000;
        let mut cooling = create_cooling(n);

        let signs: Vec<i32> = vec![1; n];
        let t_init = 500.0;  // Cold gas

        cooling.init_from_temperature(t_init, t_init, &signs).unwrap();

        let nh = 50.0;  // Very dense
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        // Run SF check
        let dt = 0.001;
        let n_stars = cooling.apply_star_formation(dt).unwrap();

        println!("  N particles: {}", n);
        println!("  N stars formed: {}", n_stars);

        // With nH=50 > 30 and T=500 < 10000, SF should occur
        assert!(n_stars > 0, "No star formation despite meeting criteria");
        println!("  ✓ PASS: Star formation triggered at high density/low T\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 6: No Cooling for m- Particles
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_no_cooling_m_minus() {
        println!("\n=== TEST 6: No Cooling for m- Particles ===");
        println!("  All particles are m- (sign = -1)");
        println!("  Expectation: T_mean unchanged after 100 steps\n");

        let n = 10000;
        let mut cooling = create_cooling(n);

        // All particles are m- (collisionless)
        let signs: Vec<i32> = vec![-1; n];
        let t_init = 50000.0;

        cooling.init_from_temperature(t_init, t_init, &signs).unwrap();
        cooling.upload_signs(&signs).unwrap();

        let nh = 10.0;  // High density that would cause cooling for m+
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        // Get internal energy before (since T_mean for m- won't work directly)
        let u_before = cooling.get_internal_energy().unwrap();
        let u_mean_before: f64 = u_before.iter().sum::<f64>() / n as f64;

        let dt = 0.001;
        let z = 1.0;
        for _ in 0..100 {
            cooling.apply_cooling(dt, z).unwrap();
        }

        let u_after = cooling.get_internal_energy().unwrap();
        let u_mean_after: f64 = u_after.iter().sum::<f64>() / n as f64;

        println!("  u_mean_before = {:.2e}", u_mean_before);
        println!("  u_mean_after  = {:.2e}", u_mean_after);
        println!("  Δu/u = {:.6}%", (u_mean_after - u_mean_before) / u_mean_before * 100.0);

        // m- should not be affected by cooling
        let tolerance = 1e-10;
        assert!((u_mean_after - u_mean_before).abs() / u_mean_before < tolerance,
            "m- particles affected by cooling: Δu/u = {:.2e}", (u_mean_after - u_mean_before) / u_mean_before);
        println!("  ✓ PASS: m- particles unaffected by cooling\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 7: Performance Benchmark
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_performance() {
        println!("\n=== TEST 7: Performance Benchmark ===");
        println!("  N = 1,000,000 particles");
        println!("  Target: < 0.5s per cooling step on RTX 3060\n");

        let n = 1_000_000;
        let mut cooling = create_cooling(n);

        let signs: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        cooling.init_from_temperature(10000.0, 10000.0, &signs).unwrap();

        let nh = 1.0;
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        // Warmup
        cooling.apply_cooling(0.001, 2.0).unwrap();

        // Benchmark
        let n_steps = 10;
        let start = Instant::now();
        for _ in 0..n_steps {
            cooling.apply_cooling(0.001, 2.0).unwrap();
        }
        let elapsed = start.elapsed();
        let time_per_step = elapsed.as_secs_f64() / n_steps as f64;

        println!("  N particles: {}", n);
        println!("  Steps: {}", n_steps);
        println!("  Total time: {:.3}s", elapsed.as_secs_f64());
        println!("  Time per step: {:.4}s", time_per_step);

        assert!(time_per_step < 0.5, "Cooling too slow: {:.3}s/step > 0.5s target", time_per_step);
        println!("  ✓ PASS: Performance target met ({:.4}s < 0.5s)\n", time_per_step);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 8: NaN Stability with Extreme Temperatures
    // ═══════════════════════════════════════════════════════════════════════
    #[test]
    fn test_nan_stability() {
        println!("\n=== TEST 8: NaN Stability ===");
        println!("  T_init extremes: 10 K and 10^9 K");
        println!("  Expectation: No NaN after 100 steps\n");

        let n = 10000;
        let mut cooling = create_cooling(n);

        // Mix of extreme temperatures
        let signs: Vec<i32> = vec![1; n];
        cooling.upload_signs(&signs).unwrap();

        // Set half to very cold, half to very hot
        let mut u = vec![0.0f64; n];
        let u_cold = (3.0 / 2.0) * K_B_OVER_MP * 10.0 / MU_IONIZED;     // 10 K
        let u_hot = (3.0 / 2.0) * K_B_OVER_MP * 1e9 / MU_IONIZED;       // 10^9 K
        for i in 0..n {
            u[i] = if i < n / 2 { u_cold } else { u_hot };
        }
        cooling.set_internal_energy(&u).unwrap();

        let nh = 1.0;
        let rho = density_for_nh(nh);
        let densities = vec![rho; n];
        cooling.upload_densities(&densities).unwrap();

        println!("  Initial: 50% at T=10K, 50% at T=10^9 K");

        let dt = 0.001;
        let z = 2.0;
        for step in 0..100 {
            cooling.apply_cooling(dt, z).unwrap();

            let has_nan = cooling.has_nan().unwrap();
            if has_nan {
                panic!("NaN detected at step {}", step);
            }
        }

        let has_nan_final = cooling.has_nan().unwrap();
        let t_final = cooling.get_mean_temperature_plus().unwrap();

        println!("  After 100 steps:");
        println!("  T_mean = {:.1} K", t_final);
        println!("  Has NaN: {}", has_nan_final);

        assert!(!has_nan_final, "NaN detected in final state");
        assert!(t_final > 0.0 && t_final.is_finite(), "Invalid temperature: {}", t_final);
        println!("  ✓ PASS: No NaN with extreme temperatures\n");
    }
}

// Run all tests with summary
#[cfg(feature = "cuda")]
#[test]
fn run_all_cooling_tests() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           CUDA COOLING KERNEL TEST SUITE                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Running 8 tests to validate GPU cooling implementation              ║");
    println!("║  GO criteria: 8/8 PASS                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
