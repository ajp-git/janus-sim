//! Cluster Analysis: FOF clustering, velocity profiles, and mass relations
//!
//! Identifies Janus proto-galaxy CLUSTERS (Mpc scale) from proto-star distributions.
//! Note: At 500 Mpc box / 10M particles, individual galaxies (kpc) are NOT resolved.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use kiddo::KdTree;
use rayon::prelude::*;

// FOF parameters
const LINKING_LENGTH: f32 = 0.5; // Mpc - identifies CLUSTERS (not individual galaxies)
const MIN_CLUSTER_PARTICLES: usize = 100; // Minimum proto-stars to be a cluster
const K_NEIGHBORS: usize = 32;
const OVERDENSITY_THRESHOLD: f64 = 5.0;

// Velocity profile binning
const N_RADIAL_BINS: usize = 50;
const R_MAX: f32 = 10.0; // Mpc - max radius for velocity profiles

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let snap_path = args.get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/mnt/T2/janus-sim/output/janus_baryonic_calibrated/snapshots/snap_03930.bin".to_string());

    let output_dir = args.get(2)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/mnt/T2/janus-sim/output/janus_baryonic_calibrated/galaxy_analysis".to_string());

    // Create output directory
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    println!("======================================================================");
    println!("CLUSTER ANALYSIS — FOF + Velocity Profiles + Mass Relations");
    println!("======================================================================");
    println!("Snapshot: {}", snap_path);
    println!("Output: {}", output_dir);
    println!("Linking length: {} Mpc (cluster scale)", LINKING_LENGTH);
    println!("Min cluster particles: {}", MIN_CLUSTER_PARTICLES);
    println!("NOTE: Objects are CLUSTERS (Mpc), not galaxies (kpc)");
    println!("======================================================================\n");

    // Read snapshot
    println!("Reading snapshot...");
    let (n, a, t, pos, vel, signs) = read_snapshot(&PathBuf::from(&snap_path))
        .expect("Failed to read snapshot");
    let z = 1.0 / a - 1.0;
    let l_box = 500.0f32;

    println!("  N = {}, z = {:.3}, t = {:.2} Gyr", n, z, t);

    // Separate m+ and m- particles
    let mut pos_plus: Vec<[f32; 3]> = Vec::new();
    let mut vel_plus: Vec<[f32; 3]> = Vec::new();
    let mut idx_plus: Vec<usize> = Vec::new();

    let mut pos_minus: Vec<[f32; 3]> = Vec::new();
    let mut idx_minus: Vec<usize> = Vec::new();

    for i in 0..n {
        let p = [pos[i*3], pos[i*3+1], pos[i*3+2]];
        let v = [vel[i*3], vel[i*3+1], vel[i*3+2]];

        if signs[i] > 0 {
            pos_plus.push(p);
            vel_plus.push(v);
            idx_plus.push(i);
        } else {
            pos_minus.push(p);
            idx_minus.push(i);
        }
    }

    println!("  N+ = {}, N- = {}", pos_plus.len(), pos_minus.len());

    // Step 1: Identify proto-stars (overdense + collapsing m+ particles)
    println!("\nStep 1: Identifying proto-stars...");
    let proto_star_indices = identify_proto_stars(&pos_plus, &vel_plus, l_box);
    println!("  Found {} proto-stars ({:.1}%)",
             proto_star_indices.len(),
             100.0 * proto_star_indices.len() as f64 / pos_plus.len() as f64);

    // Step 2: FOF clustering on proto-stars
    println!("\nStep 2: FOF clustering on proto-stars...");
    let proto_star_positions: Vec<[f32; 3]> = proto_star_indices.iter()
        .map(|&i| pos_plus[i])
        .collect();

    let (galaxy_labels, n_galaxies) = fof_clustering(&proto_star_positions, LINKING_LENGTH, l_box);

    // Count particles per galaxy
    let mut galaxy_sizes: HashMap<i32, usize> = HashMap::new();
    for &label in &galaxy_labels {
        if label >= 0 {
            *galaxy_sizes.entry(label).or_insert(0) += 1;
        }
    }

    // Filter clusters by minimum size
    let mut valid_clusters: Vec<(i32, usize)> = galaxy_sizes.into_iter()
        .filter(|&(_, size)| size >= MIN_CLUSTER_PARTICLES)
        .collect();
    valid_clusters.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by size descending

    println!("  Found {} FOF groups, {} clusters (N >= {})",
             n_galaxies, valid_clusters.len(), MIN_CLUSTER_PARTICLES);

    if valid_clusters.is_empty() {
        println!("No clusters found! Try reducing MIN_CLUSTER_PARTICLES or LINKING_LENGTH");
        return;
    }

    // Step 3: Compute cluster properties
    println!("\nStep 3: Computing cluster properties...");
    let mut clusters: Vec<Cluster> = Vec::new();

    for (cl_id, n_stars) in valid_clusters.iter().take(100) {
        // Get proto-star positions for this cluster
        let cl_proto_indices: Vec<usize> = galaxy_labels.iter()
            .enumerate()
            .filter(|(_, &l)| l == *cl_id)
            .map(|(i, _)| proto_star_indices[i])
            .collect();

        // Compute center of mass
        let mut com = [0.0f64; 3];
        for &i in &cl_proto_indices {
            com[0] += pos_plus[i][0] as f64;
            com[1] += pos_plus[i][1] as f64;
            com[2] += pos_plus[i][2] as f64;
        }
        let n_cl = cl_proto_indices.len() as f64;
        com[0] /= n_cl;
        com[1] /= n_cl;
        com[2] /= n_cl;

        // Compute half-mass radius
        let mut radii: Vec<f32> = cl_proto_indices.iter()
            .map(|&i| {
                let dx = pos_plus[i][0] as f64 - com[0];
                let dy = pos_plus[i][1] as f64 - com[1];
                let dz = pos_plus[i][2] as f64 - com[2];
                (dx*dx + dy*dy + dz*dz).sqrt() as f32
            })
            .collect();
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r_half = radii[radii.len() / 2];

        // Estimate stellar mass
        // Particle mass: M_box / N = (Ω_m × ρ_crit × h² × V) / N
        // ρ_crit = 2.78e11 M_sun/Mpc³, h=0.7, Ω_m=0.3
        // V = 500³ = 1.25e8 Mpc³, N = 10M
        // m_particle = 0.3 × 2.78e11 × 0.49 × 1.25e8 / 1e7 = 5.1e11 M_sun
        let m_particle = 5.1e11; // M_sun per particle
        let m_stellar = *n_stars as f64 * m_particle;

        clusters.push(Cluster {
            id: *cl_id,
            n_proto_stars: *n_stars,
            center: [com[0] as f32, com[1] as f32, com[2] as f32],
            r_half,
            m_stellar,
            proto_star_indices: cl_proto_indices,
        });
    }

    println!("  Top 5 clusters:");
    for (i, cl) in clusters.iter().take(5).enumerate() {
        println!("    #{}: N*={}, R_half={:.2} Mpc, M*={:.2e} M_sun, center=({:.1},{:.1},{:.1})",
                 i+1, cl.n_proto_stars, cl.r_half, cl.m_stellar,
                 cl.center[0], cl.center[1], cl.center[2]);
    }

    // Step 4: Velocity profiles for top 5 clusters
    println!("\nStep 4: Computing velocity profiles for top 5 clusters...");

    // Build KD-trees for m+ and m- particles
    let tree_plus: KdTree<f32, 3> = pos_plus.iter()
        .enumerate()
        .map(|(i, p)| {
            let shifted = [
                p[0] + l_box/2.0,
                p[1] + l_box/2.0,
                p[2] + l_box/2.0,
            ];
            (shifted, i as u64)
        })
        .collect();

    let tree_minus: KdTree<f32, 3> = pos_minus.iter()
        .enumerate()
        .map(|(i, p)| {
            let shifted = [
                p[0] + l_box/2.0,
                p[1] + l_box/2.0,
                p[2] + l_box/2.0,
            ];
            (shifted, i as u64)
        })
        .collect();

    let mut velocity_profiles: Vec<VelocityProfile> = Vec::new();

    for (i, cl) in clusters.iter().take(5).enumerate() {
        println!("  Cluster #{}: computing radial profiles...", i+1);

        let vp = compute_velocity_profile(cl, &pos_plus, &vel_plus, &pos_minus,
                                          &tree_plus, &tree_minus, l_box);
        velocity_profiles.push(vp);
    }

    // Step 5: Save results
    println!("\nStep 5: Saving results...");

    // Save cluster catalog
    let catalog_path = format!("{}/cluster_catalog.csv", output_dir);
    let mut f = BufWriter::new(File::create(&catalog_path).unwrap());
    writeln!(f, "id,n_proto_stars,x,y,z,r_half_mpc,m_stellar_msun").unwrap();
    for cl in &clusters {
        writeln!(f, "{},{},{:.3},{:.3},{:.3},{:.4},{:.2e}",
                 cl.id, cl.n_proto_stars,
                 cl.center[0], cl.center[1], cl.center[2],
                 cl.r_half, cl.m_stellar).unwrap();
    }
    println!("  Saved: {}", catalog_path);

    // Save velocity profiles
    for (i, vp) in velocity_profiles.iter().enumerate() {
        let vp_path = format!("{}/velocity_profile_cl{}.csv", output_dir, i+1);
        let mut f = BufWriter::new(File::create(&vp_path).unwrap());
        writeln!(f, "r_mpc,n_plus,n_minus,rho_plus,rho_minus,v_circ_kms,m_enc_plus,m_enc_minus").unwrap();
        for j in 0..N_RADIAL_BINS {
            writeln!(f, "{:.4},{},{},{:.6e},{:.6e},{:.2},{:.4e},{:.4e}",
                     vp.r[j], vp.n_plus[j], vp.n_minus[j],
                     vp.rho_plus[j], vp.rho_minus[j],
                     vp.v_circ[j], vp.m_enc_plus[j], vp.m_enc_minus[j]).unwrap();
        }
        println!("  Saved: {}", vp_path);
    }

    // Save proto-star positions for visualization
    let viz_path = format!("{}/proto_stars.csv", output_dir);
    let mut f = BufWriter::new(File::create(&viz_path).unwrap());
    writeln!(f, "x,y,z,cluster_id").unwrap();
    for (i, &ps_idx) in proto_star_indices.iter().enumerate() {
        let cl_id = galaxy_labels[i];
        // Only save if in a valid cluster or unclustered
        if cl_id >= 0 || (cl_id < 0 && i % 10 == 0) { // subsample unclustered
            writeln!(f, "{:.3},{:.3},{:.3},{}",
                     pos_plus[ps_idx][0], pos_plus[ps_idx][1], pos_plus[ps_idx][2],
                     cl_id).unwrap();
        }
    }
    println!("  Saved: {}", viz_path);

    // Save all particle positions (subsampled) for visualization
    let all_path = format!("{}/all_particles.csv", output_dir);
    let mut f = BufWriter::new(File::create(&all_path).unwrap());
    writeln!(f, "x,y,z,sign").unwrap();
    // Subsample to ~100k particles
    let step_plus = (pos_plus.len() / 50000).max(1);
    let step_minus = (pos_minus.len() / 50000).max(1);
    for (i, p) in pos_plus.iter().enumerate().step_by(step_plus) {
        writeln!(f, "{:.3},{:.3},{:.3},1", p[0], p[1], p[2]).unwrap();
    }
    for (i, p) in pos_minus.iter().enumerate().step_by(step_minus) {
        writeln!(f, "{:.3},{:.3},{:.3},-1", p[0], p[1], p[2]).unwrap();
    }
    println!("  Saved: {}", all_path);

    // Summary
    println!("\n======================================================================");
    println!("SUMMARY");
    println!("======================================================================");
    println!("Redshift: z = {:.3}", z);
    println!("Proto-stars: {} ({:.1}% of m+)",
             proto_star_indices.len(),
             100.0 * proto_star_indices.len() as f64 / pos_plus.len() as f64);
    println!("Clusters (N >= {}): {}", MIN_CLUSTER_PARTICLES, clusters.len());
    println!("Total stellar mass: {:.2e} M_sun",
             clusters.iter().map(|c| c.m_stellar).sum::<f64>());
    println!("Most massive cluster: {:.2e} M_sun, R_half = {:.2} Mpc",
             clusters[0].m_stellar, clusters[0].r_half);

    // Verify v_circ is in expected range for clusters
    if !velocity_profiles.is_empty() {
        let max_v = velocity_profiles[0].v_circ.iter().cloned().fold(0.0f64, f64::max);
        println!("Peak v_circ (cluster #1): {:.0} km/s", max_v);
        if max_v > 500.0 && max_v < 2000.0 {
            println!("  → Consistent with cluster velocity dispersion ✓");
        } else if max_v > 100.0 && max_v < 500.0 {
            println!("  → Consistent with galaxy rotation ✓");
        } else {
            println!("  → WARNING: v_circ outside expected range!");
        }
    }
    println!("======================================================================");
}

#[derive(Debug)]
struct Cluster {
    id: i32,
    n_proto_stars: usize,
    center: [f32; 3],
    r_half: f32,  // Half-mass radius (Mpc)
    m_stellar: f64,  // Total proto-stellar mass (M_sun)
    proto_star_indices: Vec<usize>,
}

#[derive(Debug)]
struct VelocityProfile {
    r: Vec<f32>,
    n_plus: Vec<usize>,
    n_minus: Vec<usize>,
    rho_plus: Vec<f64>,
    rho_minus: Vec<f64>,
    v_circ: Vec<f64>,
    m_enc_plus: Vec<f64>,
    m_enc_minus: Vec<f64>,
}

fn identify_proto_stars(pos_plus: &[[f32; 3]], vel_plus: &[[f32; 3]], l_box: f32) -> Vec<usize> {
    let n_plus = pos_plus.len();

    // Build KD-tree with shifted coordinates
    let tree: KdTree<f32, 3> = pos_plus.iter()
        .enumerate()
        .map(|(i, p)| {
            let shifted = [
                p[0] + l_box/2.0,
                p[1] + l_box/2.0,
                p[2] + l_box/2.0,
            ];
            (shifted, i as u64)
        })
        .collect();

    // Mean density
    let volume = (l_box as f64).powi(3);
    let rho_mean = n_plus as f64 / volume;

    // Process in parallel
    let proto_stars: Vec<usize> = (0..n_plus).into_par_iter()
        .filter_map(|i| {
            let query = [
                pos_plus[i][0] + l_box/2.0,
                pos_plus[i][1] + l_box/2.0,
                pos_plus[i][2] + l_box/2.0,
            ];

            let neighbors = tree.nearest_n::<kiddo::SquaredEuclidean>(&query, K_NEIGHBORS);
            if neighbors.len() < K_NEIGHBORS {
                return None;
            }

            // Overdensity
            let r_k = neighbors.last().unwrap().distance.sqrt() as f64;
            if r_k < 1e-10 {
                return None;
            }
            let vol_k = 4.0 / 3.0 * std::f64::consts::PI * r_k.powi(3);
            let rho_local = K_NEIGHBORS as f64 / vol_k;
            let overdensity = rho_local / rho_mean;

            if overdensity <= OVERDENSITY_THRESHOLD {
                return None;
            }

            // Velocity divergence
            let v_i = &vel_plus[i];
            let mut div_v = 0.0f64;

            for nn in &neighbors[1..] {
                let j = nn.item as usize;
                let dx = [
                    pos_plus[j][0] - pos_plus[i][0],
                    pos_plus[j][1] - pos_plus[i][1],
                    pos_plus[j][2] - pos_plus[i][2],
                ];
                let dv = [
                    vel_plus[j][0] - v_i[0],
                    vel_plus[j][1] - v_i[1],
                    vel_plus[j][2] - v_i[2],
                ];

                let r2 = (dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]) as f64;
                if r2 > 1e-10 {
                    let dot = (dx[0]*dv[0] + dx[1]*dv[1] + dx[2]*dv[2]) as f64;
                    div_v += dot / r2;
                }
            }
            div_v /= (K_NEIGHBORS - 1) as f64;

            if div_v < 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    proto_stars
}

fn fof_clustering(positions: &[[f32; 3]], linking_length: f32, l_box: f32) -> (Vec<i32>, i32) {
    let n = positions.len();
    if n == 0 {
        return (vec![], 0);
    }

    // Build KD-tree
    let tree: KdTree<f32, 3> = positions.iter()
        .enumerate()
        .map(|(i, p)| {
            let shifted = [
                p[0] + l_box/2.0,
                p[1] + l_box/2.0,
                p[2] + l_box/2.0,
            ];
            (shifted, i as u64)
        })
        .collect();

    // Union-Find structure
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
        let ri = find(parent, i);
        let rj = find(parent, j);
        if ri != rj {
            if rank[ri] < rank[rj] {
                parent[ri] = rj;
            } else if rank[ri] > rank[rj] {
                parent[rj] = ri;
            } else {
                parent[rj] = ri;
                rank[ri] += 1;
            }
        }
    }

    // Link particles within linking_length
    let ll_sq = linking_length * linking_length;

    for i in 0..n {
        let query = [
            positions[i][0] + l_box/2.0,
            positions[i][1] + l_box/2.0,
            positions[i][2] + l_box/2.0,
        ];

        // Find all neighbors within linking length
        let neighbors = tree.within::<kiddo::SquaredEuclidean>(&query, ll_sq);

        for nn in neighbors {
            let j = nn.item as usize;
            if j != i {
                union(&mut parent, &mut rank, i, j);
            }
        }
    }

    // Assign group labels
    let mut labels = vec![-1i32; n];
    let mut label_map: HashMap<usize, i32> = HashMap::new();
    let mut next_label = 0i32;

    for i in 0..n {
        let root = find(&mut parent, i);
        let label = label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = *label;
    }

    (labels, next_label)
}

fn compute_velocity_profile(
    cluster: &Cluster,
    pos_plus: &[[f32; 3]],
    vel_plus: &[[f32; 3]],
    pos_minus: &[[f32; 3]],
    tree_plus: &KdTree<f32, 3>,
    tree_minus: &KdTree<f32, 3>,
    l_box: f32,
) -> VelocityProfile {
    let center = cluster.center;
    let center_shifted = [
        center[0] + l_box/2.0,
        center[1] + l_box/2.0,
        center[2] + l_box/2.0,
    ];

    // Radial bins
    let dr = R_MAX / N_RADIAL_BINS as f32;
    let mut r_bins: Vec<f32> = (0..N_RADIAL_BINS).map(|i| (i as f32 + 0.5) * dr).collect();

    let mut n_plus = vec![0usize; N_RADIAL_BINS];
    let mut n_minus = vec![0usize; N_RADIAL_BINS];

    // Query all particles within R_MAX
    let r_max_sq = R_MAX * R_MAX;

    // m+ particles
    let neighbors_plus = tree_plus.within::<kiddo::SquaredEuclidean>(&center_shifted, r_max_sq);
    for nn in neighbors_plus {
        let r = nn.distance.sqrt();
        let bin = (r / dr) as usize;
        if bin < N_RADIAL_BINS {
            n_plus[bin] += 1;
        }
    }

    // m- particles
    let neighbors_minus = tree_minus.within::<kiddo::SquaredEuclidean>(&center_shifted, r_max_sq);
    for nn in neighbors_minus {
        let r = nn.distance.sqrt();
        let bin = (r / dr) as usize;
        if bin < N_RADIAL_BINS {
            n_minus[bin] += 1;
        }
    }

    // Compute densities and enclosed masses
    let mut rho_plus = vec![0.0f64; N_RADIAL_BINS];
    let mut rho_minus = vec![0.0f64; N_RADIAL_BINS];
    let mut m_enc_plus = vec![0.0f64; N_RADIAL_BINS];
    let mut m_enc_minus = vec![0.0f64; N_RADIAL_BINS];
    let mut v_circ = vec![0.0f64; N_RADIAL_BINS];

    // Particle mass: M_box / N = (Ω_m × ρ_crit × h² × V) / N
    // ρ_crit = 2.78e11 M_sun/Mpc³, h=0.7, Ω_m=0.3
    // V = 500³ = 1.25e8 Mpc³, N = 10M
    // m_particle = 0.3 × 2.78e11 × 0.49 × 1.25e8 / 1e7 = 5.1e11 M_sun
    let m_particle = 5.1e11; // M_sun per particle

    let mut cumulative_plus = 0.0f64;
    let mut cumulative_minus = 0.0f64;

    for i in 0..N_RADIAL_BINS {
        let r_inner = i as f64 * dr as f64;
        let r_outer = (i + 1) as f64 * dr as f64;
        let shell_vol = 4.0 / 3.0 * std::f64::consts::PI * (r_outer.powi(3) - r_inner.powi(3));

        rho_plus[i] = n_plus[i] as f64 * m_particle / shell_vol;
        rho_minus[i] = n_minus[i] as f64 * m_particle / shell_vol;

        cumulative_plus += n_plus[i] as f64 * m_particle;
        cumulative_minus += n_minus[i] as f64 * m_particle;

        m_enc_plus[i] = cumulative_plus;
        m_enc_minus[i] = cumulative_minus;

        // Circular velocity: v_circ = sqrt(G * M_eff / r)
        // In Janus: M_eff = M+ - M- (anti-gravity contribution)
        // G = 4.302e-6 kpc·(km/s)²/M_sun
        // Convert to Mpc: 1 Mpc = 1000 kpc → G = 4.302e-9 Mpc·(km/s)²/M_sun
        let g_const = 4.302e-9; // Mpc·(km/s)²/M_sun
        let r_mpc = r_bins[i] as f64;
        let m_eff = cumulative_plus - cumulative_minus; // Janus effective mass

        if r_mpc > 0.01 && m_eff > 0.0 {
            v_circ[i] = (g_const * m_eff / r_mpc).sqrt();
        } else {
            v_circ[i] = 0.0;
        }
    }

    VelocityProfile {
        r: r_bins,
        n_plus,
        n_minus,
        rho_plus,
        rho_minus,
        v_circ,
        m_enc_plus,
        m_enc_minus,
    }
}

fn read_snapshot(path: &PathBuf) -> std::io::Result<(usize, f64, f64, Vec<f32>, Vec<f32>, Vec<i8>)> {
    let mut file = File::open(path)?;

    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8)?;
    let a = f64::from_le_bytes(buf8);

    file.read_exact(&mut buf8)?;
    let t = f64::from_le_bytes(buf8);

    let mut pos_bytes = vec![0u8; n * 3 * 4];
    file.read_exact(&mut pos_bytes)?;
    let pos: Vec<f32> = pos_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut vel_bytes = vec![0u8; n * 3 * 4];
    file.read_exact(&mut vel_bytes)?;
    let vel: Vec<f32> = vel_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut signs = vec![0i8; n];
    file.read_exact(unsafe { std::slice::from_raw_parts_mut(signs.as_mut_ptr() as *mut u8, n) })?;

    Ok((n, a, t, pos, vel, signs))
}
