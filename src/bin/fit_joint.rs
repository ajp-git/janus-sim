/// Joint fit of Pantheon+ SNIa and CC+BAO H(z) data
/// Two free parameters: η and H₀
///
/// Usage: cargo run --release --bin fit_joint

use janus::friedmann::{JanusParams, CosmoInterpolator};
use janus::analysis::load_pantheon;
use std::fs;
use std::path::Path;

/// Speed of light (km/s)
const C_KM_S: f64 = 299792.458;

/// CC+BAO H(z) observations: (z, H, σ_H) in km/s/Mpc
const HZ_DATA: [(f64, f64, f64); 22] = [
    (0.07,  69.0,  19.6),
    (0.09,  69.0,  12.0),
    (0.12,  68.6,  26.2),
    (0.17,  83.0,  8.0),
    (0.20,  72.9,  29.6),
    (0.27,  77.0,  14.0),
    (0.28,  88.8,  36.6),
    (0.35,  82.7,  8.4),
    (0.40,  95.0,  17.0),
    (0.48, 101.0,  27.0),
    (0.57, 100.3,  3.7),
    (0.59, 104.0,  13.0),
    (0.60,  87.9,  6.1),
    (0.73,  97.3,  7.0),
    (0.78, 105.0,  12.0),
    (0.88,  90.0,  40.0),
    (0.90, 117.0,  23.0),
    (1.30, 168.0,  17.0),
    (1.43, 177.0,  18.0),
    (1.53, 140.0,  14.0),
    (1.75, 202.0,  40.0),
    (2.34, 222.0,  7.0),
];

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Joint Fit: Pantheon+ SNIa + CC+BAO H(z)                      ║");
    println!("║   Free parameters: η and H₀                                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Load Pantheon+ data
    let data_path = Path::new("data/Pantheon+SH0ES.dat");
    let sne = match load_pantheon(data_path) {
        Ok(data) => {
            println!("✓ Loaded {} supernovae from Pantheon+", data.len());
            data
        }
        Err(e) => {
            eprintln!("✗ Failed to load Pantheon+ data: {}", e);
            std::process::exit(1);
        }
    };
    println!("✓ CC+BAO H(z) data: {} points\n", HZ_DATA.len());

    // Grid parameters
    let eta_min = 0.90;
    let eta_max = 1.20;
    let eta_step = 0.005;
    let h0_min = 60.0;
    let h0_max = 85.0;
    let h0_step = 1.0;

    let n_eta = ((eta_max - eta_min) / eta_step) as usize + 1;
    let n_h0 = ((h0_max - h0_min) / h0_step) as usize + 1;

    println!("Grid search:");
    println!("  η  : {:.3} → {:.3}, step {:.3} ({} values)", eta_min, eta_max, eta_step, n_eta);
    println!("  H₀ : {:.1} → {:.1}, step {:.1} ({} values)", h0_min, h0_max, h0_step, n_h0);
    println!("  Total: {} grid points\n", n_eta * n_h0);

    // Results storage
    let mut results: Vec<(f64, f64, f64, f64, f64)> = Vec::with_capacity(n_eta * n_h0);

    // Track minima
    let mut min_total = (f64::INFINITY, 0.0, 0.0, 0.0, 0.0); // (chi2, eta, h0, chi2_snia, chi2_hz)
    let mut min_snia = (f64::INFINITY, 0.0, 0.0);
    let mut min_hz = (f64::INFINITY, 0.0, 0.0);

    println!("Computing χ² grid...\n");

    // Loop over η (outer loop - compute CosmoInterpolator once per η)
    for i_eta in 0..n_eta {
        let eta = eta_min + i_eta as f64 * eta_step;

        // Compute CosmoInterpolator for this η
        let params = JanusParams::from_eta(eta);
        let cosmo = CosmoInterpolator::new(&params, 5.0);

        // Precompute h(z)/h0 for all redshifts we need
        // For SNIa: need integral ∫dz'/h(z')
        // For H(z): need h(z) directly

        // Loop over H₀
        for i_h0 in 0..n_h0 {
            let h0 = h0_min + i_h0 as f64 * h0_step;

            // Compute χ²_snia
            let chi2_snia = compute_chi2_snia(&sne, &cosmo, eta, h0);

            // Compute χ²_hz
            let chi2_hz = compute_chi2_hz(&cosmo, h0);

            let chi2_total = chi2_snia + chi2_hz;

            results.push((eta, h0, chi2_snia, chi2_hz, chi2_total));

            // Track minima
            if chi2_total < min_total.0 {
                min_total = (chi2_total, eta, h0, chi2_snia, chi2_hz);
            }
            if chi2_snia < min_snia.0 {
                min_snia = (chi2_snia, eta, h0);
            }
            if chi2_hz < min_hz.0 {
                min_hz = (chi2_hz, eta, h0);
            }
        }

        // Progress
        if i_eta % 10 == 0 || i_eta == n_eta - 1 {
            print!("\r  η = {:.3} ({}/{})", eta, i_eta + 1, n_eta);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!("\n");

    // Save results to CSV
    fs::create_dir_all("output").ok();
    let mut csv = String::from("eta,H0,chi2_snia,chi2_hz,chi2_total\n");
    for (eta, h0, chi2_snia, chi2_hz, chi2_total) in &results {
        csv.push_str(&format!("{:.4},{:.1},{:.2},{:.4},{:.2}\n",
                              eta, h0, chi2_snia, chi2_hz, chi2_total));
    }
    fs::write("output/chi2_map.csv", &csv).expect("Write chi2_map.csv");
    println!("✓ Saved output/chi2_map.csv\n");

    // Report results
    let n_sne = sne.len();
    let dof_snia = n_sne - 2; // 2 free params
    let dof_hz = HZ_DATA.len() - 2;
    let dof_total = n_sne + HZ_DATA.len() - 2;

    println!("══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                  ");
    println!("══════════════════════════════════════════════════════════════════");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ GLOBAL MINIMUM (χ²_total = χ²_snia + χ²_hz)                     │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  η      = {:.4}                                               │", min_total.1);
    println!("│  H₀     = {:.1} km/s/Mpc                                       │", min_total.2);
    println!("│  χ²_tot = {:.1}                                               │", min_total.0);
    println!("│  χ²_snia= {:.1}  (χ²/dof = {:.3})                            │",
             min_total.3, min_total.3 / dof_snia as f64);
    println!("│  χ²_hz  = {:.1}  (χ²/dof = {:.3})                             │",
             min_total.4, min_total.4 / dof_hz as f64);
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PANTHEON+ ONLY MINIMUM                                          │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  η      = {:.4}                                               │", min_snia.1);
    println!("│  H₀     = {:.1} km/s/Mpc                                       │", min_snia.2);
    println!("│  χ²_snia= {:.1}  (χ²/dof = {:.3})                            │",
             min_snia.0, min_snia.0 / dof_snia as f64);
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ CC+BAO H(z) ONLY MINIMUM                                        │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  η      = {:.4}                                               │", min_hz.1);
    println!("│  H₀     = {:.1} km/s/Mpc                                       │", min_hz.2);
    println!("│  χ²_hz  = {:.1}  (χ²/dof = {:.3})                             │",
             min_hz.0, min_hz.0 / dof_hz as f64);
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Check for compatible region
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("COMPATIBILITY CHECK: χ²/dof < 2 for both datasets?");
    println!("══════════════════════════════════════════════════════════════════\n");

    let threshold_snia = 2.0 * dof_snia as f64;
    let threshold_hz = 2.0 * dof_hz as f64;

    let compatible: Vec<_> = results.iter()
        .filter(|(_, _, chi2_snia, chi2_hz, _)|
                *chi2_snia < threshold_snia && *chi2_hz < threshold_hz)
        .collect();

    if compatible.is_empty() {
        println!("✗ NO region found where both χ²/dof < 2");
        println!("\n  Best compromise at global minimum:");
        println!("    χ²_snia/dof = {:.3}", min_total.3 / dof_snia as f64);
        println!("    χ²_hz/dof   = {:.3}", min_total.4 / dof_hz as f64);
    } else {
        println!("✓ Found {} grid points with both χ²/dof < 2:\n", compatible.len());
        println!("  {:>8}  {:>8}  {:>12}  {:>12}", "η", "H₀", "χ²_snia/dof", "χ²_hz/dof");
        println!("  {:->8}  {:->8}  {:->12}  {:->12}", "", "", "", "");
        for (eta, h0, chi2_snia, chi2_hz, _) in compatible.iter().take(10) {
            println!("  {:>8.4}  {:>8.1}  {:>12.3}  {:>12.3}",
                     eta, h0, chi2_snia / dof_snia as f64, chi2_hz / dof_hz as f64);
        }
        if compatible.len() > 10 {
            println!("  ... and {} more", compatible.len() - 10);
        }
    }

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Done.");
    println!("══════════════════════════════════════════════════════════════════\n");
}

/// Compute χ² for Pantheon+ SNIa
/// μ(z) = 5 × log10(d_L(z)/Mpc) + 25
/// d_L(z) = (c/H₀) × (1+z) × ∫₀ᶻ dz'/h(z')
fn compute_chi2_snia(
    sne: &[janus::analysis::Supernova],
    cosmo: &CosmoInterpolator,
    _eta: f64,
    h0: f64
) -> f64 {
    let mut chi2 = 0.0;

    for sn in sne {
        let mu_model = compute_mu(cosmo, sn.z, h0);
        if mu_model.is_finite() {
            let residual = sn.mu - mu_model;
            chi2 += (residual * residual) / (sn.mu_err * sn.mu_err);
        } else {
            return f64::INFINITY;
        }
    }

    chi2
}

/// Compute distance modulus μ(z) using CosmoInterpolator
fn compute_mu(cosmo: &CosmoInterpolator, z: f64, h0: f64) -> f64 {
    // Integrate 1/h(z') from 0 to z using trapezoidal rule
    let n_steps = 500;
    let dz = z / n_steps as f64;

    let mut integral = 0.0;
    for i in 0..n_steps {
        let z1 = i as f64 * dz;
        let z2 = (i + 1) as f64 * dz;

        let h1 = get_h_at_z(cosmo, z1);
        let h2 = get_h_at_z(cosmo, z2);

        if h1 <= 0.0 || h2 <= 0.0 {
            return f64::NAN;
        }

        integral += 0.5 * dz * (1.0 / h1 + 1.0 / h2);
    }

    // d_L in Mpc: (c/H₀) × (1+z) × integral
    // c in km/s, H₀ in km/s/Mpc → c/H₀ in Mpc
    let d_l_mpc = (C_KM_S / h0) * (1.0 + z) * integral;

    if d_l_mpc <= 0.0 {
        return f64::NAN;
    }

    // μ = 5 × log10(d_L/Mpc) + 25
    5.0 * d_l_mpc.log10() + 25.0
}

/// Get h(z) = H(z)/H₀ from CosmoInterpolator
fn get_h_at_z(cosmo: &CosmoInterpolator, z: f64) -> f64 {
    if z < 0.0 { return 1.0; }
    if z > 5.0 { return f64::NAN; }

    // Linear interpolation: z=0 → tau_end, z=5 → tau_start
    let tau = cosmo.tau_end + (cosmo.tau_start - cosmo.tau_end) * (z / 5.0);
    let (a, h) = cosmo.get_params_at_tau(tau);

    // h = H/H₀ where H₀ = H(z=0)
    // But CosmoInterpolator already gives H in units where H(z=0) = √Ω₊
    // We need to normalize: h(z) = H(z) / H(z=0)
    let (_, h0_internal) = cosmo.get_params_at_tau(cosmo.tau_end);

    h / h0_internal
}

/// Compute χ² for CC+BAO H(z) data
fn compute_chi2_hz(cosmo: &CosmoInterpolator, h0: f64) -> f64 {
    let mut chi2 = 0.0;

    // Get internal H₀ for normalization
    let (_, h0_internal) = cosmo.get_params_at_tau(cosmo.tau_end);

    for &(z, h_obs, h_err) in HZ_DATA.iter() {
        // Get H(z) from CosmoInterpolator
        let tau = cosmo.tau_end + (cosmo.tau_start - cosmo.tau_end) * (z / 5.0);
        let (_, h_internal) = cosmo.get_params_at_tau(tau);

        // H(z) in km/s/Mpc = h0 × (H_internal / H0_internal)
        let h_model = h0 * (h_internal / h0_internal);

        let residual = h_obs - h_model;
        chi2 += (residual * residual) / (h_err * h_err);
    }

    chi2
}
