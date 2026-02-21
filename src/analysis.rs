/// Analysis and comparison with observational data
/// χ² fitting vs Pantheon+ supernova catalog
///
/// Reference: Scolnic+ (2022) ApJ 938, 113
/// Pantheon+ contains 1701 SNIa with distance moduli

use crate::friedmann::mu_janus;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Single supernova observation
#[derive(Debug, Clone)]
pub struct Supernova {
    pub name: String,
    pub z: f64,        // redshift (zHD)
    pub mu: f64,       // distance modulus (MU_SH0ES)
    pub mu_err: f64,   // error on distance modulus
}

/// Load Pantheon+ data from file
/// Format: whitespace-separated, columns zHD(3), MU_SH0ES(11), MU_SH0ES_ERR_DIAG(12)
pub fn load_pantheon(path: &Path) -> Result<Vec<Supernova>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let reader = BufReader::new(file);
    let mut sne = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;

        // Skip header
        if i == 0 { continue; }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 12 { continue; }

        let name = fields[0].to_string();
        let z: f64 = fields[2].parse().unwrap_or(f64::NAN);      // zHD
        let mu: f64 = fields[10].parse().unwrap_or(f64::NAN);    // MU_SH0ES
        let mu_err: f64 = fields[11].parse().unwrap_or(f64::NAN); // MU_SH0ES_ERR_DIAG

        // Filter invalid entries
        if z.is_nan() || mu.is_nan() || mu_err.is_nan() { continue; }
        if z <= 0.0 || mu_err <= 0.0 { continue; }

        // Pantheon+ has some calibrator SNe with z < 0.01, skip for cosmological fit
        if z < 0.01 { continue; }

        sne.push(Supernova { name, z, mu, mu_err });
    }

    Ok(sne)
}

/// Compute χ² between Janus model and Pantheon+ data
/// χ² = Σ [(μ_obs - μ_model)² / σ²]
pub fn chi_squared(sne: &[Supernova], eta: f64) -> f64 {
    let mut chi2 = 0.0;

    for sn in sne {
        let mu_model = mu_janus(sn.z, eta);
        if mu_model.is_finite() {
            let residual = sn.mu - mu_model;
            chi2 += (residual * residual) / (sn.mu_err * sn.mu_err);
        } else {
            return f64::INFINITY;
        }
    }

    chi2
}

/// Compute reduced χ² (χ²/dof)
pub fn reduced_chi_squared(sne: &[Supernova], eta: f64) -> f64 {
    let chi2 = chi_squared(sne, eta);
    let dof = sne.len() as f64 - 1.0; // 1 free parameter (η)
    chi2 / dof
}

/// Scan η parameter to find minimum χ²
/// Returns (best_eta, min_chi2, all_results)
pub fn scan_eta(sne: &[Supernova], eta_min: f64, eta_max: f64, n_points: usize)
    -> (f64, f64, Vec<(f64, f64)>)
{
    let mut results = Vec::with_capacity(n_points);
    let mut best_eta = eta_min;
    let mut min_chi2 = f64::INFINITY;

    for i in 0..n_points {
        let eta = eta_min + (eta_max - eta_min) * i as f64 / (n_points - 1) as f64;
        let chi2 = chi_squared(sne, eta);

        results.push((eta, chi2));

        if chi2 < min_chi2 {
            min_chi2 = chi2;
            best_eta = eta;
        }
    }

    (best_eta, min_chi2, results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_pantheon() {
        let path = Path::new("data/Pantheon+SH0ES.dat");
        if path.exists() {
            let sne = load_pantheon(path).unwrap();
            assert!(sne.len() > 1000);
            assert!(sne.iter().all(|sn| sn.z > 0.0));
        }
    }
}
