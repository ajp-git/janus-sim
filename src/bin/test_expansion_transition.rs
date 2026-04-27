//! Diagnostic 3: Pure expansion test crossing z=4.51 transition
//!
//! Tests Janus H(z) transition without gravity - pure expansion only.
//! Criteria:
//! - No NaN
//! - a(t) monotone increasing
//! - z(t) monotone decreasing
//! - No jump da > 10% in 1 step
//! - Explicit log at z=4.51 crossing

const ETA: f64 = 1.045;
const ALPHA_SQ_JANUS: f64 = 0.1815456201;  // (1+z_trans)^-2
const Z_TRANSITION: f64 = 4.51;

/// Janus H(z) in Gyr^-1
fn h_janus(z: f64) -> f64 {
    let h0 = 0.0699;  // 69.9 km/s/Mpc in Gyr^-1
    let a = 1.0 / (1.0 + z);

    if a <= ALPHA_SQ_JANUS {
        // Gauge process era: H = H0 * sqrt(η/α²)
        h0 * (ETA / (a * a)).sqrt()
    } else {
        // Matter era: H = H0 * sqrt(Ω_m/a³)
        // Ω_m ≈ 0.03 at z=0 for Janus
        let omega_m = 0.03;
        h0 * (omega_m / (a * a * a)).sqrt()
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DIAGNOSTIC 3: Pure Expansion Test Crossing z=4.51          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let z_init = 4.6;  // Start in gauge era
    let dt = 0.001;    // Gyr
    let n_steps = 100;

    let mut a = 1.0 / (1.0 + z_init);
    let mut a_prev = a;
    let mut t = 0.0;

    let mut crossed = false;
    let mut has_nan = false;
    let mut monotone_a = true;
    let mut monotone_z = true;
    let mut max_da_frac = 0.0_f64;
    let mut z_prev = z_init;

    println!("Initial: z = {:.4}, a = {:.6}", z_init, a);
    println!("dt = {} Gyr, {} steps\n", dt, n_steps);
    println!("{:>6} {:>10} {:>12} {:>12} {:>12} {:>8}",
             "Step", "z", "a", "H(z)", "da/a [%]", "Era");
    println!("{}", "-".repeat(70));

    for step in 0..=n_steps {
        let z = 1.0 / a - 1.0;
        let h = h_janus(z);
        let era = if a <= ALPHA_SQ_JANUS { "gauge" } else { "matter" };

        // Check for NaN
        if a.is_nan() || h.is_nan() || z.is_nan() {
            has_nan = true;
            println!("  ⚠️  NaN DETECTED at step {}", step);
            break;
        }

        // Check monotonicity
        if step > 0 {
            if a < a_prev {
                monotone_a = false;
            }
            if z > z_prev {
                monotone_z = false;
            }

            let da_frac = (a - a_prev) / a_prev;
            max_da_frac = max_da_frac.max(da_frac);
        }

        // Detect transition crossing
        if !crossed && z < Z_TRANSITION && z_prev >= Z_TRANSITION {
            crossed = true;
            println!("\n  🔄 TRANSITION CROSSED between step {} and {}", step - 1, step);
            println!("     z_prev = {:.6}, z_now = {:.6}", z_prev, z);
            println!("     a_prev = {:.6}, a_now = {:.6}", a_prev, a);
            println!("     da/a at transition = {:.4}%", (a - a_prev) / a_prev * 100.0);
            println!();
        }

        // Log every 10 steps or around transition
        if step % 10 == 0 || (z < 4.6 && z > 4.4) {
            let da_pct = if step > 0 { (a - a_prev) / a_prev * 100.0 } else { 0.0 };
            println!("{:>6} {:>10.4} {:>12.8} {:>12.6} {:>12.4} {:>8}",
                     step, z, a, h, da_pct, era);
        }

        // Update for next step
        z_prev = z;
        a_prev = a;

        // Leapfrog: da = a * H * dt
        let da = a * h * dt;
        a += da;
        t += dt;
    }

    let z_final = 1.0 / a - 1.0;

    println!("\n{}", "=".repeat(70));
    println!("RESULTS:");
    println!("  Final: z = {:.4}, a = {:.6}", z_final, a);
    println!("  Transition crossed: {}", if crossed { "YES ✓" } else { "NO" });
    println!();

    // Criteria check
    println!("CRITERIA CHECK:");
    let c1 = !has_nan;
    let c2 = monotone_a;
    let c3 = monotone_z;
    let c4 = max_da_frac < 0.10;  // < 10% jump

    println!("  [{}] No NaN detected", if c1 { "✓" } else { "✗" });
    println!("  [{}] a(t) monotone increasing", if c2 { "✓" } else { "✗" });
    println!("  [{}] z(t) monotone decreasing", if c3 { "✓" } else { "✗" });
    println!("  [{}] No da/a > 10% in 1 step (max = {:.2}%)",
             if c4 { "✓" } else { "✗" }, max_da_frac * 100.0);

    println!();
    if c1 && c2 && c3 && c4 {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  ALL CRITERIA PASS ✓ — Expansion survives transition");
        println!("═══════════════════════════════════════════════════════════════");
    } else {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  SOME CRITERIA FAIL ✗ — Review H(z) implementation");
        println!("═══════════════════════════════════════════════════════════════");
    }
}
