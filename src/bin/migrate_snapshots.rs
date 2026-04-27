//! Migrate JSNP v1 snapshots to v3 format
//!
//! Usage:
//!   cargo run --bin migrate_snapshots -- \
//!     --input output/janus_baryonic_calibrated/snapshots/snap_00000.bin \
//!     --output output/test_v3.bin \
//!     --l-box 50.0 --h0 69.9 --mu 19.0 --omega-b 0.05 \
//!     --m-plus 8.08e9 --m-minus 1.54e11 \
//!     --eps-plus 0.10 --eps-minus 0.25 \
//!     --z-init 5.0 --run-label "janus_baryonic_calibrated"

use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use janus::snapshot_v3::{
    SnapshotHeaderV3, ParticleV3, write_snapshot_v3, snapshot_info, HEADER_SIZE_V3,
};

#[derive(Parser, Debug)]
#[command(name = "migrate_snapshots")]
#[command(about = "Convert JSNP v1 snapshots to v3 format")]
struct Args {
    /// Input snapshot path (JSNP v1 format)
    #[arg(long)]
    input: String,

    /// Output snapshot path (v3 format)
    #[arg(long)]
    output: String,

    /// Box size in Mpc
    #[arg(long, default_value = "50.0")]
    l_box: f64,

    /// Hubble constant (km/s/Mpc)
    #[arg(long, default_value = "69.9")]
    h0: f64,

    /// Mass ratio m-/m+
    #[arg(long, default_value = "19.0")]
    mu: f64,

    /// Baryon density parameter
    #[arg(long, default_value = "0.05")]
    omega_b: f64,

    /// Mass of m+ particles (M_sun)
    #[arg(long)]
    m_plus: f64,

    /// Mass of m- particles (M_sun)
    #[arg(long)]
    m_minus: f64,

    /// Softening length for m+ (Mpc)
    #[arg(long, default_value = "0.50")]
    eps_plus: f64,

    /// Softening length for m- (Mpc)
    #[arg(long, default_value = "0.25")]
    eps_minus: f64,

    /// Initial redshift of the ICs (z at which ICs were generated, NOT current snapshot z)
    #[arg(long, default_value = "10.0")]
    z_init: f64,

    /// Seed for ICs (if known)
    #[arg(long, default_value = "42")]
    seed: u32,

    /// Run label
    #[arg(long, default_value = "migrated")]
    run_label: String,
}

/// JSNP v1 snapshot data
struct JsnpV1 {
    n: u64,
    a: f64,
    t_gyr: f64,
    positions: Vec<f32>,  // n*3 values
    velocities: Vec<f32>, // n*3 values
    signs: Vec<i8>,       // n values
}

/// Read JSNP v1 format snapshot
fn read_jsnp_v1(path: &Path) -> std::io::Result<JsnpV1> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Header: n (u64), a (f64), t (f64)
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let a = f64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let t_gyr = f64::from_le_bytes(buf8);

    eprintln!("  JSNP v1 header: N={}, a={:.6}, t={:.3} Gyr", n, a, t_gyr);

    // Positions: n*3 * f32
    let mut positions = vec![0f32; (n * 3) as usize];
    let mut buf4 = [0u8; 4];
    for i in 0..(n * 3) as usize {
        reader.read_exact(&mut buf4)?;
        positions[i] = f32::from_le_bytes(buf4);
    }

    // Velocities: n*3 * f32
    let mut velocities = vec![0f32; (n * 3) as usize];
    for i in 0..(n * 3) as usize {
        reader.read_exact(&mut buf4)?;
        velocities[i] = f32::from_le_bytes(buf4);
    }

    // Signs: n * i8
    let mut signs = vec![0i8; n as usize];
    let mut buf1 = [0u8; 1];
    for i in 0..n as usize {
        reader.read_exact(&mut buf1)?;
        signs[i] = buf1[0] as i8;
    }

    Ok(JsnpV1 {
        n,
        a,
        t_gyr,
        positions,
        velocities,
        signs,
    })
}

fn main() {
    let args = Args::parse();

    eprintln!("=== Migrate Snapshots: JSNP v1 -> v3 ===");
    eprintln!("Input:  {}", args.input);
    eprintln!("Output: {}", args.output);

    // 1. Read JSNP v1
    eprintln!("\n[1/4] Reading JSNP v1 snapshot...");
    let jsnp = match read_jsnp_v1(Path::new(&args.input)) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("ERROR: Failed to read input: {}", e);
            std::process::exit(1);
        }
    };

    let z = 1.0 / jsnp.a - 1.0;
    eprintln!("  z = {:.4}", z);

    // Count m+ and m-
    let n_plus = jsnp.signs.iter().filter(|&&s| s > 0).count();
    let n_minus = jsnp.signs.iter().filter(|&&s| s < 0).count();
    eprintln!("  N+ = {}, N- = {}", n_plus, n_minus);

    // 2. Build ParticleV3 vector
    eprintln!("\n[2/4] Converting to ParticleV3...");
    let mut particles = Vec::with_capacity(jsnp.n as usize);

    for i in 0..jsnp.n as usize {
        let pos = [
            jsnp.positions[i * 3],
            jsnp.positions[i * 3 + 1],
            jsnp.positions[i * 3 + 2],
        ];
        let vel = [
            jsnp.velocities[i * 3],
            jsnp.velocities[i * 3 + 1],
            jsnp.velocities[i * 3 + 2],
        ];

        let is_positive = jsnp.signs[i] > 0;
        let (mass, epsilon, sign) = if is_positive {
            (args.m_plus as f32, args.eps_plus as f32, 1u8)
        } else {
            (args.m_minus as f32, args.eps_minus as f32, 255u8)
        };

        particles.push(ParticleV3 {
            pos,
            vel,
            mass,
            epsilon,
            sign,
            split_level: 0,
            is_star: 0,
            flags: 0,
        });
    }

    eprintln!("  Converted {} particles", particles.len());

    // 3. Build SnapshotHeaderV3
    eprintln!("\n[3/4] Building header...");
    let mut header = SnapshotHeaderV3::new(&args.run_label);

    // Values read from v1 snapshot (current state)
    header.n_total = jsnp.n;
    header.a = jsnp.a;
    header.t_gyr = jsnp.t_gyr;
    header.l_box = args.l_box;
    header.h0 = args.h0;
    header.mu = args.mu;
    header.omega_b = args.omega_b;
    header.m_part_plus_base = args.m_plus;
    header.m_part_minus_base = args.m_minus;
    header.eps_plus_base = args.eps_plus;
    header.eps_minus_base = args.eps_minus;
    header.n_split_max = 0;
    header.seed_ic = args.seed;
    header.z_init = args.z_init;  // IC generation redshift (NOT current snapshot z)
    header.n_stars = 0;
    header.z_start_run = args.z_init;
    header.sfr = 0.0;
    header.rho_max = 0.0;

    // 4. Write v3 snapshot
    eprintln!("\n[4/4] Writing v3 snapshot...");
    match write_snapshot_v3(Path::new(&args.output), &header, &particles) {
        Ok(()) => eprintln!("  Written successfully."),
        Err(e) => {
            eprintln!("ERROR: Failed to write output: {}", e);
            std::process::exit(1);
        }
    }

    // Verify
    eprintln!("\n=== Verification ===");
    match snapshot_info(Path::new(&args.output)) {
        Ok(info) => println!("{}", info),
        Err(e) => eprintln!("WARNING: Could not verify output: {}", e),
    }

    // File size comparison
    let input_size = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
    let output_size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
    eprintln!("\nFile sizes:");
    eprintln!("  Input (v1):  {} bytes ({:.1} MB)", input_size, input_size as f64 / 1e6);
    eprintln!("  Output (v3): {} bytes ({:.1} MB)", output_size, output_size as f64 / 1e6);

    let expected_v3 = HEADER_SIZE_V3 as u64 + jsnp.n * 36;
    eprintln!("  Expected v3: {} bytes", expected_v3);

    if output_size == expected_v3 {
        eprintln!("\n[OK] Migration successful!");
    } else {
        eprintln!("\n[WARNING] File size mismatch - expected {} bytes", expected_v3);
    }
}
