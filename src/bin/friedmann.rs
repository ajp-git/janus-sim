/// Janus Friedmann Integration — Magnitude-Redshift Fit with Pantheon+
///
/// Reproduces D'Agostini & Petit (2018) Astrophys. Space Sci. 363:139
/// Fits ~1500 SNIa from Pantheon+ catalog with single free parameter η = |ρ̄|/ρ
///
/// Usage: cargo run --release --bin friedmann

use janus::friedmann::*;
use janus::analysis::*;
use janus::constants::*;
use std::fs;
use std::path::Path;

fn main() {
    env_logger::init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus Cosmological Model — Friedmann Integration             ║");
    println!("║   Single parameter fit: η = |ρ̄₀|/ρ₀                            ║");
    println!("║   Reference: Petit, Margnat & Zejli (2024) EPJC 84:1226        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Load Pantheon+ data
    let data_path = Path::new("data/Pantheon+SH0ES.dat");
    let sne = match load_pantheon(data_path) {
        Ok(data) => {
            println!("✓ Loaded {} supernovae from Pantheon+", data.len());
            println!("  z range: {:.4} — {:.4}",
                data.iter().map(|s| s.z).fold(f64::INFINITY, f64::min),
                data.iter().map(|s| s.z).fold(0.0, f64::max));
            data
        }
        Err(e) => {
            eprintln!("✗ Failed to load Pantheon+ data: {}", e);
            eprintln!("  Expected at: {}", data_path.display());
            eprintln!("  Download from: https://github.com/PantheonPlusSH0ES/DataRelease");
            std::process::exit(1);
        }
    };

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Phase 1: Parameter scan η ∈ [0.5, 5.0]");
    println!("══════════════════════════════════════════════════════════════════\n");

    // Coarse scan
    let (best_eta_coarse, min_chi2_coarse, coarse_results) = scan_eta(&sne, 0.5, 5.0, 50);

    println!("{:>8}  {:>12}  {:>12}", "η", "χ²", "χ²/dof");
    println!("{:-<38}", "");

    for (eta, chi2) in coarse_results.iter().step_by(5) {
        let chi2_dof = chi2 / (sne.len() as f64 - 1.0);
        println!("{:>8.3}  {:>12.1}  {:>12.4}", eta, chi2, chi2_dof);
    }

    println!("\nCoarse minimum: η = {:.3}, χ² = {:.1}", best_eta_coarse, min_chi2_coarse);

    // Fine scan around minimum
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Phase 2: Fine scan η ∈ [{:.2}, {:.2}]",
        (best_eta_coarse - 0.3).max(0.1), best_eta_coarse + 0.3);
    println!("══════════════════════════════════════════════════════════════════\n");

    let eta_low = (best_eta_coarse - 0.3).max(0.1);
    let eta_high = best_eta_coarse + 0.3;
    let (best_eta, min_chi2, fine_results) = scan_eta(&sne, eta_low, eta_high, 50);

    println!("{:>8}  {:>12}  {:>12}", "η", "χ²", "χ²/dof");
    println!("{:-<38}", "");

    for (eta, chi2) in fine_results.iter().step_by(5) {
        let chi2_dof = chi2 / (sne.len() as f64 - 1.0);
        println!("{:>8.4}  {:>12.1}  {:>12.4}", eta, chi2, chi2_dof);
    }

    let chi2_dof = min_chi2 / (sne.len() as f64 - 1.0);

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                  ");
    println!("══════════════════════════════════════════════════════════════════");
    println!("\n  Best fit:  η = {:.4}", best_eta);
    println!("  χ²       = {:.1}", min_chi2);
    println!("  χ²/dof   = {:.4}  (dof = {})", chi2_dof, sne.len() - 1);
    println!("  N_SNIa   = {}", sne.len());

    // Compare with ΛCDM
    let chi2_lcdm = chi_squared_lcdm(&sne);
    let chi2_dof_lcdm = chi2_lcdm / (sne.len() as f64 - 2.0); // 2 params: Ωm, ΩΛ
    println!("\n  ΛCDM comparison (Ωm=0.3, ΩΛ=0.7):");
    println!("  χ²_ΛCDM  = {:.1}", chi2_lcdm);
    println!("  χ²/dof   = {:.4}  (dof = {})", chi2_dof_lcdm, sne.len() - 2);

    // Physical interpretation
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                    Physical Parameters                           ");
    println!("══════════════════════════════════════════════════════════════════");

    let params = JanusParams::from_eta(best_eta);
    println!("\n  H₀        = {:.1} km/s/Mpc", H0_KM_S_MPC);
    println!("  Ω_+       = {:.3}", params.omega_plus);
    println!("  Ω_-       = {:.3}", params.omega_minus);
    println!("  η = |ρ₋|/ρ₊ = {:.4}", best_eta);
    println!("  w_eff     = {:.4} (effective EoS)", -1.0 / best_eta);

    // Generate Hubble diagram
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                    Generating Outputs                            ");
    println!("══════════════════════════════════════════════════════════════════\n");

    fs::create_dir_all("output").ok();

    // Output 1: Chi² scan results
    let mut scan_csv = String::from("eta,chi2,chi2_dof\n");
    for (eta, chi2) in &fine_results {
        let dof = chi2 / (sne.len() as f64 - 1.0);
        scan_csv.push_str(&format!("{:.6},{:.2},{:.6}\n", eta, chi2, dof));
    }
    fs::write("output/chi2_scan.csv", &scan_csv).expect("Write chi2_scan.csv");
    println!("✓ output/chi2_scan.csv");

    // Output 2: Hubble diagram with best fit
    let mut hubble_csv = String::from("z,mu_janus,mu_lcdm\n");
    for i in 1..=200 {
        let z = i as f64 * 0.01;
        let mu_j = mu_janus(z, best_eta);
        let mu_l = compute_mu_lcdm(z);
        hubble_csv.push_str(&format!("{:.4},{:.4},{:.4}\n", z, mu_j, mu_l));
    }
    fs::write("output/hubble_diagram.csv", &hubble_csv).expect("Write hubble_diagram.csv");
    println!("✓ output/hubble_diagram.csv");

    // Output 3: Residuals vs Pantheon+
    let mut residuals_csv = String::from("z,mu_obs,mu_err,mu_janus,residual\n");
    for sn in &sne {
        let mu_j = mu_janus(sn.z, best_eta);
        let residual = sn.mu - mu_j;
        residuals_csv.push_str(&format!("{:.5},{:.4},{:.4},{:.4},{:.4}\n",
            sn.z, sn.mu, sn.mu_err, mu_j, residual));
    }
    fs::write("output/residuals.csv", &residuals_csv).expect("Write residuals.csv");
    println!("✓ output/residuals.csv");

    // Output 4: Summary JSON
    let summary = format!(r#"{{
  "model": "Janus bimetric",
  "reference": "Petit, Margnat & Zejli (2024) EPJC 84:1226",
  "data": "Pantheon+ (Scolnic+ 2022)",
  "n_sne": {},
  "best_eta": {:.6},
  "w_eff": {:.6},
  "chi2": {:.2},
  "chi2_dof": {:.6},
  "chi2_lcdm": {:.2},
  "chi2_dof_lcdm": {:.6},
  "H0_km_s_Mpc": {:.1},
  "omega_plus": {:.4},
  "omega_minus": {:.4}
}}"#,
        sne.len(), best_eta, -1.0/best_eta, min_chi2, chi2_dof,
        chi2_lcdm, chi2_dof_lcdm,
        H0_KM_S_MPC, params.omega_plus, params.omega_minus);

    fs::write("output/summary.json", &summary).expect("Write summary.json");
    println!("✓ output/summary.json");

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Done. Results in output/");
    println!("══════════════════════════════════════════════════════════════════\n");
}

/// Compute χ² for flat ΛCDM (Ωm=0.3, ΩΛ=0.7)
fn chi_squared_lcdm(sne: &[Supernova]) -> f64 {
    let mut chi2 = 0.0;

    for sn in sne {
        let mu_lcdm = compute_mu_lcdm(sn.z);
        let residual = sn.mu - mu_lcdm;
        chi2 += (residual * residual) / (sn.mu_err * sn.mu_err);
    }

    chi2
}

/// Compute distance modulus for flat ΛCDM
fn compute_mu_lcdm(z: f64) -> f64 {
    let omega_m = 0.3_f64;
    let omega_lambda = 0.7_f64;
    let n = 1000;

    let mut integral = 0.0;
    let dz = z / n as f64;

    for i in 0..n {
        let zi = (i as f64 + 0.5) * dz;
        let e_z = (omega_m * (1.0 + zi).powi(3) + omega_lambda).sqrt();
        integral += dz / e_z;
    }

    let d_l_m = (1.0 + z) * C / H0 * integral;
    distance_modulus(d_l_m)
}
