//! Stellar post-processing: identify proto-star candidates in snapshots
//!
//! Uses kiddo for fast k-NN queries and rayon for parallel processing.
//! Applies SF criteria: overdensity > 5.0 AND div(v) < 0

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use kiddo::KdTree;
use rayon::prelude::*;

const K_NEIGHBORS: usize = 32;
const OVERDENSITY_THRESHOLD: f64 = 5.0;
const SAMPLE_FRACTION: f64 = 0.1; // Sample 10% for speed

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let snap_dir = args.get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/mnt/T2/janus-sim/output/janus_baryonic_calibrated/snapshots".to_string());

    let output_csv = args.get(2)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/mnt/T2/janus-sim/output/janus_baryonic_calibrated/stellar_evolution.csv".to_string());

    println!("======================================================================");
    println!("STELLAR POST-PROCESSING — Proto-star identification (Rust)");
    println!("======================================================================");
    println!("Snapshot dir: {}", snap_dir);
    println!("Output: {}", output_csv);
    println!("k-NN neighbors: {}", K_NEIGHBORS);
    println!("Overdensity threshold: {:.1}", OVERDENSITY_THRESHOLD);
    println!("Sample fraction: {:.1}", SAMPLE_FRACTION);
    println!("======================================================================\n");

    // Find all snapshots
    let pattern = format!("{}/*.bin", snap_dir);
    let mut snap_files: Vec<PathBuf> = glob::glob(&pattern)
        .expect("Failed to read glob pattern")
        .filter_map(|e| e.ok())
        .collect();

    snap_files.sort();
    println!("Found {} snapshots\n", snap_files.len());

    if snap_files.is_empty() {
        eprintln!("No snapshots found!");
        return;
    }

    // Process all snapshots in parallel
    let processed = AtomicUsize::new(0);
    let total = snap_files.len();

    let results: Vec<SnapshotResult> = snap_files
        .par_iter()
        .filter_map(|path| {
            let result = process_snapshot(path);
            let n = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 10 == 0 || n == total {
                println!("Processed {}/{} snapshots", n, total);
            }
            result
        })
        .collect();

    // Sort by step
    let mut results = results;
    results.sort_by_key(|r| r.step);

    // Write CSV
    let file = File::create(&output_csv).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    writeln!(writer, "step,z,t_gyr,n_plus,n_sample,n_proto_stars,mean_overdensity,frac_collapsing").unwrap();

    for r in &results {
        writeln!(
            writer,
            "{},{:.4},{:.4},{},{},{},{:.4},{:.4}",
            r.step, r.z, r.t_gyr, r.n_plus, r.n_sample, r.n_proto_stars,
            r.mean_overdensity, r.frac_collapsing
        ).unwrap();
    }

    writer.flush().unwrap();
    println!("\nWrote {} rows to {}", results.len(), output_csv);

    // Summary statistics
    if let Some(last) = results.last() {
        println!("\n=== Final snapshot (step {}, z={:.2}) ===", last.step, last.z);
        println!("Proto-stars: {} / {} sampled ({:.2}%)",
                 last.n_proto_stars, last.n_sample,
                 100.0 * last.n_proto_stars as f64 / last.n_sample as f64);
        println!("Mean overdensity: {:.2}", last.mean_overdensity);
        println!("Fraction collapsing (div_v < 0): {:.2}%", 100.0 * last.frac_collapsing);
    }
}

#[derive(Debug)]
struct SnapshotResult {
    step: u32,
    z: f64,
    t_gyr: f64,
    n_plus: usize,
    n_sample: usize,
    n_proto_stars: usize,
    mean_overdensity: f64,
    frac_collapsing: f64,
}

fn process_snapshot(path: &PathBuf) -> Option<SnapshotResult> {
    // Extract step from filename
    let stem = path.file_stem()?.to_str()?;
    let step: u32 = stem.strip_prefix("snap_")?.parse().ok()?;

    // Read snapshot
    let (n, a, t, pos, vel, signs) = read_snapshot(path).ok()?;
    let z = 1.0 / a - 1.0;

    // Get box size from data range
    let l_box = 500.0; // Known from simulation config

    // Filter m+ particles
    let mut pos_plus = Vec::new();
    let mut vel_plus = Vec::new();

    for i in 0..n {
        if signs[i] > 0 {
            // Shift to [0, L_box] for KdTree (expects positive coordinates)
            pos_plus.push([
                (pos[i * 3] as f64 + l_box / 2.0) as f32,
                (pos[i * 3 + 1] as f64 + l_box / 2.0) as f32,
                (pos[i * 3 + 2] as f64 + l_box / 2.0) as f32,
            ]);
            vel_plus.push([vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]]);
        }
    }

    let n_plus = pos_plus.len();
    if n_plus < K_NEIGHBORS {
        return None;
    }

    // Build KdTree
    let tree: KdTree<f32, 3> = pos_plus.iter()
        .enumerate()
        .map(|(i, p)| (*p, i as u64))
        .collect();

    // Sample particles for analysis
    let n_sample = ((n_plus as f64 * SAMPLE_FRACTION) as usize).max(1000).min(n_plus);
    let step_size = n_plus / n_sample;

    // Mean density for normalization
    let volume = l_box.powi(3);
    let rho_mean = n_plus as f64 / volume;

    let mut n_proto_stars = 0usize;
    let mut sum_overdensity = 0.0f64;
    let mut n_collapsing = 0usize;

    for i in (0..n_plus).step_by(step_size).take(n_sample) {
        let query = pos_plus[i];

        // k-NN query
        let neighbors = tree.nearest_n::<kiddo::SquaredEuclidean>(&query, K_NEIGHBORS);

        if neighbors.len() < K_NEIGHBORS {
            continue;
        }

        // Estimate local density from k-th neighbor distance
        let r_k = neighbors.last().unwrap().distance.sqrt() as f64;
        if r_k < 1e-10 {
            continue;
        }

        // SPH density estimate: rho ~ k / (4/3 * pi * r_k^3)
        let vol_k = 4.0 / 3.0 * std::f64::consts::PI * r_k.powi(3);
        let rho_local = K_NEIGHBORS as f64 / vol_k;
        let overdensity = rho_local / rho_mean;

        sum_overdensity += overdensity;

        // Velocity divergence estimate
        let v_i = &vel_plus[i];
        let mut div_v = 0.0f64;

        for nn in &neighbors[1..] {  // Skip self (index 0)
            let j = nn.item as usize;
            let dx = [
                pos_plus[j][0] - query[0],
                pos_plus[j][1] - query[1],
                pos_plus[j][2] - query[2],
            ];
            let dv = [
                vel_plus[j][0] - v_i[0],
                vel_plus[j][1] - v_i[1],
                vel_plus[j][2] - v_i[2],
            ];

            let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if r2 > 1e-10 {
                // div(v) ~ sum(dv . dr / r^2)
                let dot = dx[0] * dv[0] + dx[1] * dv[1] + dx[2] * dv[2];
                div_v += dot as f64 / r2 as f64;
            }
        }
        div_v /= (K_NEIGHBORS - 1) as f64;

        let is_collapsing = div_v < 0.0;
        if is_collapsing {
            n_collapsing += 1;
        }

        // Proto-star criteria: high overdensity AND collapsing
        if overdensity > OVERDENSITY_THRESHOLD && is_collapsing {
            n_proto_stars += 1;
        }
    }

    let actual_sample = (0..n_plus).step_by(step_size).take(n_sample).count();

    Some(SnapshotResult {
        step,
        z,
        t_gyr: t,
        n_plus,
        n_sample: actual_sample,
        n_proto_stars,
        mean_overdensity: sum_overdensity / actual_sample as f64,
        frac_collapsing: n_collapsing as f64 / actual_sample as f64,
    })
}

fn read_snapshot(path: &PathBuf) -> std::io::Result<(usize, f64, f64, Vec<f32>, Vec<f32>, Vec<i8>)> {
    let mut file = File::open(path)?;

    // Header
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8)?;
    let a = f64::from_le_bytes(buf8);

    file.read_exact(&mut buf8)?;
    let t = f64::from_le_bytes(buf8);

    // Positions (n * 3 * f32)
    let mut pos_bytes = vec![0u8; n * 3 * 4];
    file.read_exact(&mut pos_bytes)?;
    let pos: Vec<f32> = pos_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Velocities (n * 3 * f32)
    let mut vel_bytes = vec![0u8; n * 3 * 4];
    file.read_exact(&mut vel_bytes)?;
    let vel: Vec<f32> = vel_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Signs (n * i8)
    let mut signs = vec![0i8; n];
    file.read_exact(unsafe { std::slice::from_raw_parts_mut(signs.as_mut_ptr() as *mut u8, n) })?;

    Ok((n, a, t, pos, vel, signs))
}
