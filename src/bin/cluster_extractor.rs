//! cluster_extractor — Extract particles around a cluster for visualization
//!
//! Writes GBIN binary format for cluster_lightup.py renderer.
//! Uses kiddo for k-NN density estimation, rayon for parallelization.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use kiddo::KdTree;
use rayon::prelude::*;

const MAGIC: u32 = 0x4742494E; // "GBIN"

#[derive(Clone)]
struct Config {
    snap_dir: PathBuf,
    out_dir: PathBuf,
    center: [f64; 3],
    radius: f64,
    box_size: f64,
    sf_threshold: f64,
    n_neighbors: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let snap_dir = get_arg(&args, "--snap-dir")
        .unwrap_or_else(|| "output/janus_baryonic_calibrated/snapshots".to_string());
    let cluster_csv = get_arg(&args, "--cluster-csv")
        .unwrap_or_else(|| "output/janus_baryonic_calibrated/cluster_analysis/cluster_catalog.csv".to_string());
    let cluster_id: usize = get_arg(&args, "--cluster-id")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let out_dir = get_arg(&args, "--out-dir")
        .unwrap_or_else(|| "output/cluster_lightup/gbin".to_string());
    let radius: f64 = get_arg(&args, "--radius")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25.0);
    let sf_threshold: f64 = get_arg(&args, "--sf-threshold")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let n_neighbors: usize = get_arg(&args, "--n-neighbors")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let box_size: f64 = get_arg(&args, "--box-size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500.0);

    println!("======================================================================");
    println!("CLUSTER EXTRACTOR — GBIN v2 Output for Lightup Renderer");
    println!("======================================================================");

    // Read cluster center from CSV
    let center = read_cluster_center(&cluster_csv, cluster_id)
        .expect("Failed to read cluster center");

    println!("Cluster #{} center: ({:.2}, {:.2}, {:.2}) Mpc",
             cluster_id, center[0], center[1], center[2]);
    println!("Extraction radius: {:.1} Mpc", radius);
    println!("Box size: {:.1} Mpc", box_size);
    println!("SF threshold: {:.1}", sf_threshold);
    println!("k-NN neighbors: {}", n_neighbors);
    println!("Snapshot dir: {}", snap_dir);
    println!("Output dir: {}", out_dir);
    println!("======================================================================\n");

    // Create output directory
    fs::create_dir_all(&out_dir).expect("Failed to create output directory");

    let config = Config {
        snap_dir: PathBuf::from(&snap_dir),
        out_dir: PathBuf::from(&out_dir),
        center,
        radius,
        box_size,
        sf_threshold,
        n_neighbors,
    };

    // List snapshots
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

    // Process snapshots in parallel
    let processed = AtomicUsize::new(0);
    let total = snap_files.len();

    snap_files.par_iter().for_each(|path| {
        if let Err(e) = process_snapshot(path, &config) {
            eprintln!("Error processing {:?}: {}", path, e);
        }
        let n = processed.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 20 == 0 || n == total {
            println!("Extracted {}/{} snapshots", n, total);
        }
    });

    println!("\n=== Phase 2: Marking first appearances ===");
    mark_first_appearances(&config.out_dir);

    println!("\nDone! {} GBIN files written to {}", total, out_dir);
}

fn get_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone())
}

fn read_cluster_center(csv_path: &str, cluster_id: usize) -> Option<[f64; 3]> {
    let file = File::open(csv_path).ok()?;
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        if i - 1 != cluster_id { continue; }

        let line = line.ok()?;
        let parts: Vec<&str> = line.split(',').collect();
        // Format: id,n_proto_stars,x,y,z,r_half_mpc,m_stellar_msun
        if parts.len() >= 5 {
            let x: f64 = parts[2].parse().ok()?;
            let y: f64 = parts[3].parse().ok()?;
            let z: f64 = parts[4].parse().ok()?;
            return Some([x, y, z]);
        }
    }
    None
}

fn periodic_dist(d: f64, l: f64) -> f64 {
    let mut dd = d;
    while dd > l / 2.0 { dd -= l; }
    while dd < -l / 2.0 { dd += l; }
    dd
}

fn process_snapshot(path: &PathBuf, cfg: &Config) -> std::io::Result<()> {
    // Extract step from filename
    let stem = path.file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid filename"))?;
    let step: u32 = stem.strip_prefix("snap_")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Cannot parse step"))?;

    // Read snapshot
    let (n, a, t, pos, vel, signs) = read_snapshot(path)?;
    let z = 1.0 / a - 1.0;

    // Filter particles in sphere (with periodic boundary conditions)
    let mut plus_in_sphere: Vec<(usize, [f32; 3], [f32; 3])> = Vec::new();
    let mut minus_in_sphere: Vec<[f32; 3]> = Vec::new();

    let r2_max = cfg.radius * cfg.radius;

    for i in 0..n {
        let px = pos[i * 3] as f64;
        let py = pos[i * 3 + 1] as f64;
        let pz = pos[i * 3 + 2] as f64;

        let dx = periodic_dist(px - cfg.center[0], cfg.box_size);
        let dy = periodic_dist(py - cfg.center[1], cfg.box_size);
        let dz = periodic_dist(pz - cfg.center[2], cfg.box_size);

        let r2 = dx * dx + dy * dy + dz * dz;

        if r2 < r2_max {
            // Store relative to center
            let rel_pos = [dx as f32, dy as f32, dz as f32];

            if signs[i] > 0 {
                plus_in_sphere.push((
                    i,
                    rel_pos,
                    [vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]]
                ));
            } else {
                minus_in_sphere.push(rel_pos);
            }
        }
    }

    let n_plus = plus_in_sphere.len();
    let n_minus = minus_in_sphere.len();

    if n_plus < cfg.n_neighbors {
        // Not enough particles, write empty file
        write_gbin_empty(cfg, step, z as f32, t as f32)?;
        return Ok(());
    }

    // Calculate global mean density (all m+ in box)
    let n_plus_global = signs.iter().filter(|&&s| s > 0).count();
    let rho_mean = n_plus_global as f64 / cfg.box_size.powi(3);

    // Build KdTree on m+ in sphere
    let tree: KdTree<f32, 3> = plus_in_sphere.iter()
        .enumerate()
        .map(|(i, (_, pos, _))| (*pos, i as u64))
        .collect();

    // Process each m+ particle
    let particles_plus: Vec<ParticlePlus> = plus_in_sphere
        .par_iter()
        .map(|(_, pos, vel)| {
            // k-NN query
            let neighbors = tree.nearest_n::<kiddo::SquaredEuclidean>(pos, cfg.n_neighbors);

            // Density from k-th neighbor
            let r_k = neighbors.last()
                .map(|n| n.distance.sqrt() as f64)
                .unwrap_or(1.0);

            let overdensity = if r_k > 1e-10 {
                let vol_k = 4.0 / 3.0 * std::f64::consts::PI * r_k.powi(3);
                let rho_local = cfg.n_neighbors as f64 / vol_k;
                (rho_local / rho_mean) as f32
            } else {
                1.0
            };

            // Velocity divergence
            let mut div_v = 0.0f64;
            for nn in neighbors.iter().skip(1) {
                let j = nn.item as usize;
                let (_, pos_j, vel_j) = &plus_in_sphere[j];

                let dx = [
                    pos_j[0] - pos[0],
                    pos_j[1] - pos[1],
                    pos_j[2] - pos[2],
                ];
                let dv = [
                    vel_j[0] - vel[0],
                    vel_j[1] - vel[1],
                    vel_j[2] - vel[2],
                ];

                let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if r2 > 1e-10 {
                    let dot = dx[0] * dv[0] + dx[1] * dv[1] + dx[2] * dv[2];
                    div_v += dot as f64 / r2 as f64;
                }
            }
            div_v /= (cfg.n_neighbors - 1).max(1) as f64;

            let is_protostar = overdensity > cfg.sf_threshold as f32 && div_v < 0.0;

            ParticlePlus {
                x: pos[0],
                y: pos[1],
                z: pos[2],
                overdensity,
                is_protostar: is_protostar as u8,
                is_new_star: 0, // Filled in post-processing
                _pad: [0; 2],
            }
        })
        .collect();

    let n_proto = particles_plus.iter().filter(|p| p.is_protostar == 1).count();

    // Write GBIN file
    let out_name = format!("cluster_z{:.3}_s{:05}.gbin", z, step);
    let out_path = cfg.out_dir.join(&out_name);

    write_gbin(
        &out_path,
        z as f32,
        t as f32,
        step,
        &particles_plus,
        &minus_in_sphere,
        &cfg.center,
        cfg.radius,
        n_proto as u32,
        0, // n_new filled in post-processing
    )?;

    Ok(())
}

#[repr(C, packed)]
struct ParticlePlus {
    x: f32,
    y: f32,
    z: f32,
    overdensity: f32,
    is_protostar: u8,
    is_new_star: u8,
    _pad: [u8; 2],
}

fn write_gbin(
    path: &PathBuf,
    z: f32,
    t: f32,
    step: u32,
    plus: &[ParticlePlus],
    minus: &[[f32; 3]],
    center: &[f64; 3],
    radius: f64,
    n_proto: u32,
    n_new: u32,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header (48 bytes)
    writer.write_all(&MAGIC.to_le_bytes())?;           // [0..4]
    writer.write_all(&z.to_le_bytes())?;               // [4..8]
    writer.write_all(&t.to_le_bytes())?;               // [8..12]
    writer.write_all(&step.to_le_bytes())?;            // [12..16]
    writer.write_all(&(plus.len() as u32).to_le_bytes())?;   // [16..20]
    writer.write_all(&(minus.len() as u32).to_le_bytes())?;  // [20..24]
    writer.write_all(&n_proto.to_le_bytes())?;         // [24..28]
    writer.write_all(&n_new.to_le_bytes())?;           // [28..32]
    writer.write_all(&(center[0] as f32).to_le_bytes())?;  // [32..36]
    writer.write_all(&(center[1] as f32).to_le_bytes())?;  // [36..40]
    writer.write_all(&(center[2] as f32).to_le_bytes())?;  // [40..44]
    writer.write_all(&(radius as f32).to_le_bytes())?;     // [44..48]

    // m+ particles (20 bytes each)
    for p in plus {
        writer.write_all(&p.x.to_le_bytes())?;
        writer.write_all(&p.y.to_le_bytes())?;
        writer.write_all(&p.z.to_le_bytes())?;
        writer.write_all(&p.overdensity.to_le_bytes())?;
        writer.write_all(&[p.is_protostar, p.is_new_star, 0, 0])?;
    }

    // m- particles (12 bytes each)
    for m in minus {
        writer.write_all(&m[0].to_le_bytes())?;
        writer.write_all(&m[1].to_le_bytes())?;
        writer.write_all(&m[2].to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

fn write_gbin_empty(cfg: &Config, step: u32, z: f32, t: f32) -> std::io::Result<()> {
    let out_name = format!("cluster_z{:.3}_s{:05}.gbin", z, step);
    let out_path = cfg.out_dir.join(&out_name);
    write_gbin(&out_path, z, t, step, &[], &[], &cfg.center, cfg.radius, 0, 0)
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

/// Post-processing: mark is_new_star by comparing consecutive snapshots
fn mark_first_appearances(out_dir: &PathBuf) {
    // List all GBIN files sorted by step
    let pattern = format!("{}/*.gbin", out_dir.display());
    let mut gbin_files: Vec<PathBuf> = glob::glob(&pattern)
        .expect("Failed to read glob pattern")
        .filter_map(|e| e.ok())
        .collect();

    gbin_files.sort();

    if gbin_files.len() < 2 {
        println!("Not enough files for first appearance marking");
        return;
    }

    println!("Marking first appearances in {} files...", gbin_files.len());

    let mut prev_protostars: HashSet<[u32; 3]> = HashSet::new();

    for (i, path) in gbin_files.iter().enumerate() {
        // Read GBIN file
        let (header, mut plus, minus) = read_gbin(path).expect("Failed to read GBIN");

        // Current protostars (quantized positions as keys)
        let mut current_protostars: HashSet<[u32; 3]> = HashSet::new();
        let mut n_new = 0u32;

        for p in &mut plus {
            if p.is_protostar == 1 {
                // Quantize position to ~0.01 Mpc resolution
                let key = [
                    ((p.x + 300.0) * 100.0) as u32,
                    ((p.y + 300.0) * 100.0) as u32,
                    ((p.z + 300.0) * 100.0) as u32,
                ];

                current_protostars.insert(key);

                // New if not in previous step
                if !prev_protostars.contains(&key) {
                    p.is_new_star = 1;
                    n_new += 1;
                }
            }
        }

        // Rewrite file with updated is_new_star and n_new
        let mut new_header = header;
        new_header.n_new = n_new;

        rewrite_gbin(path, &new_header, &plus, &minus)
            .expect("Failed to rewrite GBIN");

        prev_protostars = current_protostars;

        if (i + 1) % 50 == 0 || i + 1 == gbin_files.len() {
            println!("  Processed {}/{} files", i + 1, gbin_files.len());
        }
    }
}

#[derive(Clone)]
struct GbinHeader {
    z: f32,
    t: f32,
    step: u32,
    n_plus: u32,
    n_minus: u32,
    n_proto: u32,
    n_new: u32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
}

fn read_gbin(path: &PathBuf) -> std::io::Result<(GbinHeader, Vec<ParticlePlus>, Vec<[f32; 3]>)> {
    let mut file = File::open(path)?;

    // Read header
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Bad magic"));
    }

    file.read_exact(&mut buf4)?;
    let z = f32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let t = f32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let step = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let n_plus = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let n_minus = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let n_proto = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let n_new = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let cx = f32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let cy = f32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let cz = f32::from_le_bytes(buf4);
    file.read_exact(&mut buf4)?;
    let radius = f32::from_le_bytes(buf4);

    let header = GbinHeader {
        z, t, step, n_plus, n_minus, n_proto, n_new, cx, cy, cz, radius,
    };

    // Read m+ particles
    let mut plus = Vec::with_capacity(n_plus as usize);
    for _ in 0..n_plus {
        let mut buf20 = [0u8; 20];
        file.read_exact(&mut buf20)?;
        plus.push(ParticlePlus {
            x: f32::from_le_bytes([buf20[0], buf20[1], buf20[2], buf20[3]]),
            y: f32::from_le_bytes([buf20[4], buf20[5], buf20[6], buf20[7]]),
            z: f32::from_le_bytes([buf20[8], buf20[9], buf20[10], buf20[11]]),
            overdensity: f32::from_le_bytes([buf20[12], buf20[13], buf20[14], buf20[15]]),
            is_protostar: buf20[16],
            is_new_star: buf20[17],
            _pad: [buf20[18], buf20[19]],
        });
    }

    // Read m- particles
    let mut minus = Vec::with_capacity(n_minus as usize);
    for _ in 0..n_minus {
        let mut buf12 = [0u8; 12];
        file.read_exact(&mut buf12)?;
        minus.push([
            f32::from_le_bytes([buf12[0], buf12[1], buf12[2], buf12[3]]),
            f32::from_le_bytes([buf12[4], buf12[5], buf12[6], buf12[7]]),
            f32::from_le_bytes([buf12[8], buf12[9], buf12[10], buf12[11]]),
        ]);
    }

    Ok((header, plus, minus))
}

fn rewrite_gbin(
    path: &PathBuf,
    header: &GbinHeader,
    plus: &[ParticlePlus],
    minus: &[[f32; 3]],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header
    writer.write_all(&MAGIC.to_le_bytes())?;
    writer.write_all(&header.z.to_le_bytes())?;
    writer.write_all(&header.t.to_le_bytes())?;
    writer.write_all(&header.step.to_le_bytes())?;
    writer.write_all(&header.n_plus.to_le_bytes())?;
    writer.write_all(&header.n_minus.to_le_bytes())?;
    writer.write_all(&header.n_proto.to_le_bytes())?;
    writer.write_all(&header.n_new.to_le_bytes())?;
    writer.write_all(&header.cx.to_le_bytes())?;
    writer.write_all(&header.cy.to_le_bytes())?;
    writer.write_all(&header.cz.to_le_bytes())?;
    writer.write_all(&header.radius.to_le_bytes())?;

    // m+ particles
    for p in plus {
        writer.write_all(&p.x.to_le_bytes())?;
        writer.write_all(&p.y.to_le_bytes())?;
        writer.write_all(&p.z.to_le_bytes())?;
        writer.write_all(&p.overdensity.to_le_bytes())?;
        writer.write_all(&[p.is_protostar, p.is_new_star, 0, 0])?;
    }

    // m- particles
    for m in minus {
        writer.write_all(&m[0].to_le_bytes())?;
        writer.write_all(&m[1].to_le_bytes())?;
        writer.write_all(&m[2].to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}
