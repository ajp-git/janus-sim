//! Diagnostic: Check if new() ICs have index-position correlation
//! This determines if February 2M run was real physics or artifact

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Diagnostic: ICs index-position correlation");
    println!("═══════════════════════════════════════════════════════════\n");

    // February 2M reference parameters
    let n_particles = 2_000_000usize;
    let eta = 1.045f64;
    let box_size = 271.0f64;

    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;

    println!("Parameters (February 2M reference):");
    println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
    println!("  Box = {} Mpc", box_size);
    println!("  η = {}", eta);
    println!();

    // Create simulation with new() - same as February
    println!("Creating ICs with new()...");
    let sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
        .expect("Failed to create simulation");

    // Get positions
    let positions = sim.get_positions().expect("get_positions failed");
    let n = positions.len() / 3;

    // Extract x, y, z
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);

    for i in 0..n {
        x.push(positions[i * 3]);
        y.push(positions[i * 3 + 1]);
        z.push(positions[i * 3 + 2]);
    }

    // Compute correlations
    let idx: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let corr_x = pearson_correlation(&idx, &x);
    let corr_y = pearson_correlation(&idx, &y);
    let corr_z = pearson_correlation(&idx, &z);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  RÉSULTATS");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Corrélation index ↔ position:");
    println!("  corr(idx, x) = {:.4}", corr_x);
    println!("  corr(idx, y) = {:.4}", corr_y);
    println!("  corr(idx, z) = {:.4}", corr_z);

    // Compute mean positions for + and -
    let mean_x_pos: f64 = x[..n_positive].iter().sum::<f64>() / n_positive as f64;
    let mean_y_pos: f64 = y[..n_positive].iter().sum::<f64>() / n_positive as f64;
    let mean_z_pos: f64 = z[..n_positive].iter().sum::<f64>() / n_positive as f64;

    let mean_x_neg: f64 = x[n_positive..].iter().sum::<f64>() / n_negative as f64;
    let mean_y_neg: f64 = y[n_positive..].iter().sum::<f64>() / n_negative as f64;
    let mean_z_neg: f64 = z[n_positive..].iter().sum::<f64>() / n_negative as f64;

    println!("\nPosition moyenne:");
    println!("  <pos>_+ = ({:.1}, {:.1}, {:.1}) Mpc", mean_x_pos, mean_y_pos, mean_z_pos);
    println!("  <pos>_- = ({:.1}, {:.1}, {:.1}) Mpc", mean_x_neg, mean_y_neg, mean_z_neg);

    // Compute Seg₀
    let dx = mean_x_pos - mean_x_neg;
    let dy = mean_y_pos - mean_y_neg;
    let dz = mean_z_pos - mean_z_neg;
    let separation = (dx*dx + dy*dy + dz*dz).sqrt();
    let seg_0 = separation / box_size;

    println!("\nSégrégation initiale:");
    println!("  |COM+ - COM-| = {:.2} Mpc", separation);
    println!("  Seg₀ = {:.4}", seg_0);

    // Verdict
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════\n");

    let max_corr = corr_x.abs().max(corr_y.abs()).max(corr_z.abs());

    if max_corr > 0.5 {
        println!("  ❌ corr > 0.5 : ARTIFACT GÉOMÉTRIQUE");
        println!("     Le run février était probablement aussi un artefact.");
        println!("     S_max = 0.694 était pré-imposé par les ICs, pas dynamique.");
    } else if max_corr > 0.1 {
        println!("  ⚠️  corr ∈ [0.1, 0.5] : CORRÉLATION MODÉRÉE");
        println!("     Biais partiel dans les ICs.");
    } else {
        println!("  ✓ corr < 0.1 : ICs CORRECTES");
        println!("     Pas de corrélation index-position.");
        println!("     new() a changé entre février et maintenant.");
    }

    if seg_0 > 0.01 {
        println!("\n  ⚠️  Seg₀ = {:.4} > 0.01 : ségrégation pré-imposée", seg_0);
    } else {
        println!("\n  ✓ Seg₀ = {:.4} ≈ 0 : pas de ségrégation initiale", seg_0);
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
