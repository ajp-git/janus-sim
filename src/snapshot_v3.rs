//! Snapshot v3 Format — Auto-descriptive snapshot format for adaptive zoom
//!
//! Each snapshot contains complete metadata in the header, allowing
//! resumption and analysis without external parameters.
//!
//! Header: 408 bytes (fixed, extensible via header_size field)
//! Particle: 36 bytes each

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;

/// Magic number: "JANUSV3\n" in ASCII = 0x4A414E555356330A
pub const MAGIC_V3: u64 = 0x0A335653554E414A;  // Little-endian
pub const VERSION_V3: u32 = 3;
pub const HEADER_SIZE_V3: u32 = 408;
pub const PARTICLE_SIZE_V3: usize = 36;

/// Snapshot header v3 — 408 bytes, auto-descriptive
/// Note: Fields are naturally aligned (no padding needed for 408 bytes)
#[derive(Debug, Clone)]
#[repr(C)]
pub struct SnapshotHeaderV3 {
    pub magic: u64,              // 0x4A414E555356330A ("JANUSV3\n")
    pub version: u32,            // = 3
    pub header_size: u32,        // = 408 bytes (extensible)
    pub n_total: u64,            // Total particle count
    pub a: f64,                  // Scale factor (= 1/(1+z))
    pub t_gyr: f64,              // Cosmic time in Gyr
    pub l_box: f64,              // Box size in Mpc
    pub h0: f64,                 // Hubble constant (km/s/Mpc)
    pub mu: f64,                 // Mass ratio m-/m+
    pub omega_b: f64,            // Baryon density parameter
    pub m_part_plus_base: f64,   // Base m+ mass before splits (M_sun)
    pub m_part_minus_base: f64,  // Base m- mass before splits (M_sun)
    pub eps_plus_base: f64,      // Base m+ softening (Mpc)
    pub eps_minus_base: f64,     // Base m- softening (Mpc)
    pub n_split_max: u32,        // Maximum split level used
    pub seed_ic: u32,            // IC seed
    pub z_init: f64,             // Initial redshift of ICs
    pub n_stars: u64,            // Number of stars formed
    pub z_start_run: f64,        // Start redshift of this run
    pub sfr: f64,                // Star formation rate (M_sun/Gyr)
    pub rho_max: f64,            // Maximum density (M_sun/Mpc^3)
    pub run_label: [u8; 256],    // Run label (UTF-8, null-padded)
}

impl Default for SnapshotHeaderV3 {
    fn default() -> Self {
        Self {
            magic: MAGIC_V3,
            version: VERSION_V3,
            header_size: HEADER_SIZE_V3,
            n_total: 0,
            a: 1.0,
            t_gyr: 0.0,
            l_box: 500.0,
            h0: 69.9,
            mu: 19.0,
            omega_b: 0.05,
            m_part_plus_base: 1e10,
            m_part_minus_base: 1e10,
            eps_plus_base: 0.5,
            eps_minus_base: 2.5,
            n_split_max: 0,
            seed_ic: 42,
            z_init: 10.0,
            n_stars: 0,
            z_start_run: 10.0,
            sfr: 0.0,
            rho_max: 0.0,
            run_label: [0u8; 256],
        }
    }
}

impl SnapshotHeaderV3 {
    /// Create a new header with the given run label
    pub fn new(label: &str) -> Self {
        let mut header = Self::default();
        header.set_run_label(label);
        header
    }

    /// Set the run label (truncated to 255 bytes + null terminator)
    pub fn set_run_label(&mut self, label: &str) {
        self.run_label = [0u8; 256];
        let bytes = label.as_bytes();
        let len = bytes.len().min(255);
        self.run_label[..len].copy_from_slice(&bytes[..len]);
    }

    /// Get the run label as a string
    pub fn get_run_label(&self) -> String {
        // Copy array to avoid unaligned reference
        let label: [u8; 256] = { self.run_label };
        let null_pos = label.iter().position(|&b| b == 0).unwrap_or(256);
        String::from_utf8_lossy(&label[..null_pos]).to_string()
    }

    /// Get redshift from scale factor
    pub fn z(&self) -> f64 {
        let a: f64 = { self.a };
        1.0 / a - 1.0
    }

    /// Write header to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Copy all fields to avoid unaligned references from packed struct
        let magic: u64 = { self.magic };
        let version: u32 = { self.version };
        let header_size: u32 = { self.header_size };
        let n_total: u64 = { self.n_total };
        let a: f64 = { self.a };
        let t_gyr: f64 = { self.t_gyr };
        let l_box: f64 = { self.l_box };
        let h0: f64 = { self.h0 };
        let mu: f64 = { self.mu };
        let omega_b: f64 = { self.omega_b };
        let m_part_plus_base: f64 = { self.m_part_plus_base };
        let m_part_minus_base: f64 = { self.m_part_minus_base };
        let eps_plus_base: f64 = { self.eps_plus_base };
        let eps_minus_base: f64 = { self.eps_minus_base };
        let n_split_max: u32 = { self.n_split_max };
        let seed_ic: u32 = { self.seed_ic };
        let z_init: f64 = { self.z_init };
        let n_stars: u64 = { self.n_stars };
        let z_start_run: f64 = { self.z_start_run };
        let sfr: f64 = { self.sfr };
        let rho_max: f64 = { self.rho_max };
        let run_label: [u8; 256] = { self.run_label };

        writer.write_all(&magic.to_le_bytes())?;
        writer.write_all(&version.to_le_bytes())?;
        writer.write_all(&header_size.to_le_bytes())?;
        writer.write_all(&n_total.to_le_bytes())?;
        writer.write_all(&a.to_le_bytes())?;
        writer.write_all(&t_gyr.to_le_bytes())?;
        writer.write_all(&l_box.to_le_bytes())?;
        writer.write_all(&h0.to_le_bytes())?;
        writer.write_all(&mu.to_le_bytes())?;
        writer.write_all(&omega_b.to_le_bytes())?;
        writer.write_all(&m_part_plus_base.to_le_bytes())?;
        writer.write_all(&m_part_minus_base.to_le_bytes())?;
        writer.write_all(&eps_plus_base.to_le_bytes())?;
        writer.write_all(&eps_minus_base.to_le_bytes())?;
        writer.write_all(&n_split_max.to_le_bytes())?;
        writer.write_all(&seed_ic.to_le_bytes())?;
        writer.write_all(&z_init.to_le_bytes())?;
        writer.write_all(&n_stars.to_le_bytes())?;
        writer.write_all(&z_start_run.to_le_bytes())?;
        writer.write_all(&sfr.to_le_bytes())?;
        writer.write_all(&rho_max.to_le_bytes())?;
        writer.write_all(&run_label)?;
        Ok(())
    }

    /// Read header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        reader.read_exact(&mut buf8)?;
        let magic = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let header_size = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let n_total = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let a = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let t_gyr = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let l_box = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let h0 = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let mu = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let omega_b = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let m_part_plus_base = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let m_part_minus_base = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let eps_plus_base = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let eps_minus_base = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let n_split_max = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let seed_ic = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let z_init = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let n_stars = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let z_start_run = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let sfr = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let rho_max = f64::from_le_bytes(buf8);

        let mut run_label = [0u8; 256];
        reader.read_exact(&mut run_label)?;

        Ok(Self {
            magic,
            version,
            header_size,
            n_total,
            a,
            t_gyr,
            l_box,
            h0,
            mu,
            omega_b,
            m_part_plus_base,
            m_part_minus_base,
            eps_plus_base,
            eps_minus_base,
            n_split_max,
            seed_ic,
            z_init,
            n_stars,
            z_start_run,
            sfr,
            rho_max,
            run_label,
        })
    }
}

/// Particle data v3 — 36 bytes per particle
/// Note: Fields are naturally aligned (12+12+4+4+4 = 36 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ParticleV3 {
    pub pos: [f32; 3],       // Position (Mpc)
    pub vel: [f32; 3],       // Velocity (km/s)
    pub mass: f32,           // Mass (M_sun) — varies with split_level
    pub epsilon: f32,        // Individual softening (Mpc)
    pub sign: u8,            // +1 = m+, 255 = m-
    pub split_level: u8,     // Refinement level (0 = original)
    pub is_star: u8,         // 0 = gas/DM, 1 = star
    pub flags: u8,           // Reserved (bit 0: is_HR, bit 1: is_active)
}

impl ParticleV3 {
    /// Create a new m+ particle
    pub fn new_plus(pos: [f32; 3], vel: [f32; 3], mass: f32, epsilon: f32) -> Self {
        Self {
            pos,
            vel,
            mass,
            epsilon,
            sign: 1,
            split_level: 0,
            is_star: 0,
            flags: 0,
        }
    }

    /// Create a new m- particle
    pub fn new_minus(pos: [f32; 3], vel: [f32; 3], mass: f32, epsilon: f32) -> Self {
        Self {
            pos,
            vel,
            mass,
            epsilon,
            sign: 255,
            split_level: 0,
            is_star: 0,
            flags: 0,
        }
    }

    /// Check if particle is m+
    pub fn is_positive(&self) -> bool {
        let sign: u8 = { self.sign };
        sign == 1
    }

    /// Check if particle is m-
    pub fn is_negative(&self) -> bool {
        let sign: u8 = { self.sign };
        sign == 255
    }

    /// Write particle to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Copy all fields to avoid unaligned references from packed struct
        let pos: [f32; 3] = { self.pos };
        let vel: [f32; 3] = { self.vel };
        let mass: f32 = { self.mass };
        let epsilon: f32 = { self.epsilon };
        let sign: u8 = { self.sign };
        let split_level: u8 = { self.split_level };
        let is_star: u8 = { self.is_star };
        let flags: u8 = { self.flags };

        writer.write_all(&pos[0].to_le_bytes())?;
        writer.write_all(&pos[1].to_le_bytes())?;
        writer.write_all(&pos[2].to_le_bytes())?;
        writer.write_all(&vel[0].to_le_bytes())?;
        writer.write_all(&vel[1].to_le_bytes())?;
        writer.write_all(&vel[2].to_le_bytes())?;
        writer.write_all(&mass.to_le_bytes())?;
        writer.write_all(&epsilon.to_le_bytes())?;
        writer.write_all(&[sign, split_level, is_star, flags])?;
        Ok(())
    }

    /// Read particle from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf1 = [0u8; 4];

        let mut pos = [0f32; 3];
        let mut vel = [0f32; 3];

        for i in 0..3 {
            reader.read_exact(&mut buf4)?;
            pos[i] = f32::from_le_bytes(buf4);
        }

        for i in 0..3 {
            reader.read_exact(&mut buf4)?;
            vel[i] = f32::from_le_bytes(buf4);
        }

        reader.read_exact(&mut buf4)?;
        let mass = f32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let epsilon = f32::from_le_bytes(buf4);

        reader.read_exact(&mut buf1)?;

        Ok(Self {
            pos,
            vel,
            mass,
            epsilon,
            sign: buf1[0],
            split_level: buf1[1],
            is_star: buf1[2],
            flags: buf1[3],
        })
    }
}

/// Write a complete snapshot v3 to file
pub fn write_snapshot_v3(
    path: &Path,
    header: &SnapshotHeaderV3,
    particles: &[ParticleV3],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    let mut header_copy = header.clone();
    header_copy.magic = MAGIC_V3;
    header_copy.version = VERSION_V3;
    header_copy.header_size = HEADER_SIZE_V3;
    header_copy.n_total = particles.len() as u64;
    header_copy.write_to(&mut writer)?;

    // Write particles
    for p in particles {
        p.write_to(&mut writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read a complete snapshot v3 from file
pub fn read_snapshot_v3(path: &Path) -> std::io::Result<(SnapshotHeaderV3, Vec<ParticleV3>)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let header = SnapshotHeaderV3::read_from(&mut reader)?;

    // Validate magic and version
    if header.magic != MAGIC_V3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid magic number: expected 0x{:016X}, got 0x{:016X}", MAGIC_V3, header.magic),
        ));
    }
    if header.version != VERSION_V3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid version: expected {}, got {}", VERSION_V3, header.version),
        ));
    }

    // Skip to particle data (in case header_size > 408 in future versions)
    // Header struct fields total 408 bytes, but we already read them
    // So we need to seek if header_size > 408
    if header.header_size > HEADER_SIZE_V3 {
        let skip = (header.header_size - HEADER_SIZE_V3) as i64;
        reader.seek(SeekFrom::Current(skip))?;
    }

    // Read particles
    let n = header.n_total as usize;
    let mut particles = Vec::with_capacity(n);
    for _ in 0..n {
        particles.push(ParticleV3::read_from(&mut reader)?);
    }

    Ok((header, particles))
}

/// Read only the header from a snapshot (fast, doesn't load particles)
pub fn read_header_only(path: &Path) -> std::io::Result<SnapshotHeaderV3> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let header = SnapshotHeaderV3::read_from(&mut reader)?;

    // Validate magic and version
    if header.magic != MAGIC_V3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid magic number: expected 0x{:016X}, got 0x{:016X}", MAGIC_V3, header.magic),
        ));
    }
    if header.version != VERSION_V3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid version: expected {}, got {}", VERSION_V3, header.version),
        ));
    }

    Ok(header)
}

/// Generate a human-readable info string for a snapshot
pub fn snapshot_info(path: &Path) -> std::io::Result<String> {
    let header = read_header_only(path)?;

    // Copy all values to avoid unaligned references from packed struct
    let z = header.z();
    let label = header.get_run_label();
    let n_total = { header.n_total };
    let a = { header.a };
    let t_gyr = { header.t_gyr };
    let l_box = { header.l_box };
    let h0 = { header.h0 };
    let mu = { header.mu };
    let omega_b = { header.omega_b };
    let m_part_plus_base = { header.m_part_plus_base };
    let m_part_minus_base = { header.m_part_minus_base };
    let eps_plus_base = { header.eps_plus_base };
    let eps_minus_base = { header.eps_minus_base };
    let n_split_max = { header.n_split_max };
    let seed_ic = { header.seed_ic };
    let z_init = { header.z_init };
    let n_stars = { header.n_stars };
    let z_start_run = { header.z_start_run };
    let sfr = { header.sfr };
    let rho_max = { header.rho_max };

    Ok(format!(
        "=== Snapshot v3: {} ===
  N_total     : {:>12}
  z           : {:>12.4}  (a = {:.6})
  t           : {:>12.3} Gyr
  L_box       : {:>12.1} Mpc
  H0          : {:>12.1} km/s/Mpc
  mu          : {:>12.1}
  Omega_b     : {:>12.4}
  m+_base     : {:>12.3e} M_sun
  m-_base     : {:>12.3e} M_sun
  eps+_base   : {:>12.4} Mpc
  eps-_base   : {:>12.4} Mpc
  split_max   : {:>12}
  seed        : {:>12}
  z_init      : {:>12.2}
  N_stars     : {:>12}
  z_start_run : {:>12.4}
  SFR         : {:>12.2} M_sun/Gyr
  rho_max     : {:>12.3e} M_sun/Mpc^3
  run_label   : {}",
        path.display(),
        n_total,
        z, a,
        t_gyr,
        l_box,
        h0,
        mu,
        omega_b,
        m_part_plus_base,
        m_part_minus_base,
        eps_plus_base,
        eps_minus_base,
        n_split_max,
        seed_ic,
        z_init,
        n_stars,
        z_start_run,
        sfr,
        rho_max,
        label,
    ))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_roundtrip_header() {
        // Create header with specific values
        let mut header = SnapshotHeaderV3::default();
        header.n_total = 1000;
        header.a = 0.5;
        header.t_gyr = 6.5;
        header.l_box = 100.0;
        header.h0 = 70.0;
        header.mu = 19.0;
        header.omega_b = 0.048;
        header.m_part_plus_base = 1.5e10;
        header.m_part_minus_base = 2.85e11;
        header.eps_plus_base = 0.3;
        header.eps_minus_base = 1.5;
        header.n_split_max = 3;
        header.seed_ic = 12345;
        header.z_init = 10.0;
        header.n_stars = 42000;
        header.z_start_run = 5.0;
        header.sfr = 1.5;
        header.rho_max = 1e8;
        header.set_run_label("test_run_roundtrip");

        // Create 100 particles
        let mut particles = Vec::new();
        for i in 0..100 {
            let pos = [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
            let vel = [10.0, 20.0, 30.0];
            if i % 2 == 0 {
                particles.push(ParticleV3::new_plus(pos, vel, 1e9, 0.1));
            } else {
                let mut p = ParticleV3::new_minus(pos, vel, 1.9e10, 0.5);
                p.split_level = (i % 4) as u8;
                particles.push(p);
            }
        }

        // Write to temp file
        let temp = NamedTempFile::new().unwrap();
        write_snapshot_v3(temp.path(), &header, &particles).unwrap();

        // Read back
        let (header2, particles2) = read_snapshot_v3(temp.path()).unwrap();

        // Verify header fields are bit-identical
        assert_eq!(header2.magic, MAGIC_V3, "magic mismatch");
        assert_eq!(header2.version, VERSION_V3, "version mismatch");
        assert_eq!(header2.header_size, HEADER_SIZE_V3, "header_size mismatch");
        assert_eq!(header2.n_total, 100, "n_total mismatch");
        assert_eq!(header2.a, header.a, "a mismatch");
        assert_eq!(header2.t_gyr, header.t_gyr, "t_gyr mismatch");
        assert_eq!(header2.l_box, header.l_box, "l_box mismatch");
        assert_eq!(header2.h0, header.h0, "h0 mismatch");
        assert_eq!(header2.mu, header.mu, "mu mismatch");
        assert_eq!(header2.omega_b, header.omega_b, "omega_b mismatch");
        assert_eq!(header2.m_part_plus_base, header.m_part_plus_base, "m_part_plus_base mismatch");
        assert_eq!(header2.m_part_minus_base, header.m_part_minus_base, "m_part_minus_base mismatch");
        assert_eq!(header2.eps_plus_base, header.eps_plus_base, "eps_plus_base mismatch");
        assert_eq!(header2.eps_minus_base, header.eps_minus_base, "eps_minus_base mismatch");
        assert_eq!(header2.n_split_max, header.n_split_max, "n_split_max mismatch");
        assert_eq!(header2.seed_ic, header.seed_ic, "seed_ic mismatch");
        assert_eq!(header2.z_init, header.z_init, "z_init mismatch");
        assert_eq!(header2.n_stars, header.n_stars, "n_stars mismatch");
        assert_eq!(header2.z_start_run, header.z_start_run, "z_start_run mismatch");
        assert_eq!(header2.sfr, header.sfr, "sfr mismatch");
        assert_eq!(header2.rho_max, header.rho_max, "rho_max mismatch");
        assert_eq!(header2.get_run_label(), "test_run_roundtrip", "run_label mismatch");

        // Verify particles
        assert_eq!(particles2.len(), 100, "particle count mismatch");
        for (p1, p2) in particles.iter().zip(particles2.iter()) {
            assert_eq!(p1.pos, p2.pos, "pos mismatch");
            assert_eq!(p1.vel, p2.vel, "vel mismatch");
            assert_eq!(p1.mass, p2.mass, "mass mismatch");
            assert_eq!(p1.epsilon, p2.epsilon, "epsilon mismatch");
            assert_eq!(p1.sign, p2.sign, "sign mismatch");
            assert_eq!(p1.split_level, p2.split_level, "split_level mismatch");
            assert_eq!(p1.is_star, p2.is_star, "is_star mismatch");
            assert_eq!(p1.flags, p2.flags, "flags mismatch");
        }
    }

    #[test]
    fn test_split_level_consistency() {
        // Verify mass × 8^split_level = m_part_base
        // Verify epsilon × 2^split_level = eps_base

        let m_base_plus = 1e10f32;
        let m_base_minus = 1.9e11f32;
        let eps_base_plus = 0.5f32;
        let eps_base_minus = 2.5f32;

        for level in 0..8 {
            let factor_mass = 8f32.powi(level as i32);
            let factor_eps = 2f32.powi(level as i32);

            let m_plus = m_base_plus / factor_mass;
            let m_minus = m_base_minus / factor_mass;
            let eps_plus = eps_base_plus / factor_eps;
            let eps_minus = eps_base_minus / factor_eps;

            // Verify consistency
            let m_plus_check = m_plus * factor_mass;
            let m_minus_check = m_minus * factor_mass;
            let eps_plus_check = eps_plus * factor_eps;
            let eps_minus_check = eps_minus * factor_eps;

            assert!((m_plus_check - m_base_plus).abs() < 1e-3 * m_base_plus,
                "m+ consistency failed at level {}: {} vs {}", level, m_plus_check, m_base_plus);
            assert!((m_minus_check - m_base_minus).abs() < 1e-3 * m_base_minus,
                "m- consistency failed at level {}: {} vs {}", level, m_minus_check, m_base_minus);
            assert!((eps_plus_check - eps_base_plus).abs() < 1e-6,
                "eps+ consistency failed at level {}: {} vs {}", level, eps_plus_check, eps_base_plus);
            assert!((eps_minus_check - eps_base_minus).abs() < 1e-6,
                "eps- consistency failed at level {}: {} vs {}", level, eps_minus_check, eps_base_minus);
        }
    }

    #[test]
    fn test_header_size_field() {
        // Verify header_size = 408
        let header = SnapshotHeaderV3::default();
        assert_eq!(header.header_size, 408, "header_size should be 408");
        assert_eq!(HEADER_SIZE_V3, 408, "HEADER_SIZE_V3 constant should be 408");

        // Verify reader can skip header_size bytes
        let mut particles = Vec::new();
        for i in 0..10 {
            particles.push(ParticleV3::new_plus(
                [i as f32, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                1e9,
                0.1,
            ));
        }

        let temp = NamedTempFile::new().unwrap();
        write_snapshot_v3(temp.path(), &header, &particles).unwrap();

        // Read file directly and verify we can skip to particles
        let file = File::open(temp.path()).unwrap();
        let mut reader = BufReader::new(file);

        // Read just magic, version, header_size
        let mut buf = [0u8; 16];
        reader.read_exact(&mut buf).unwrap();
        let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let header_size = u32::from_le_bytes(buf[12..16].try_into().unwrap());

        assert_eq!(magic, MAGIC_V3);
        assert_eq!(version, VERSION_V3);
        assert_eq!(header_size, 408);

        // Skip to particle data using header_size
        reader.seek(SeekFrom::Start(header_size as u64)).unwrap();

        // Read first particle
        let p = ParticleV3::read_from(&mut reader).unwrap();
        assert_eq!(p.pos[0], 0.0);
        assert_eq!(p.sign, 1);
    }

    #[test]
    fn test_read_header_only_fast() {
        // Create a snapshot with 1M particles
        let header = SnapshotHeaderV3::new("large_snapshot_test");
        let particles: Vec<ParticleV3> = (0..10_000)
            .map(|i| ParticleV3::new_plus(
                [i as f32, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                1e9,
                0.1,
            ))
            .collect();

        let temp = NamedTempFile::new().unwrap();
        write_snapshot_v3(temp.path(), &header, &particles).unwrap();

        // Time the header-only read
        let start = std::time::Instant::now();
        let header2 = read_header_only(temp.path()).unwrap();
        let elapsed = start.elapsed();

        // Should complete in < 10ms (generous margin)
        assert!(elapsed.as_millis() < 10,
            "read_header_only took {}ms, should be < 10ms", elapsed.as_millis());

        // Verify header is correct
        assert_eq!(header2.n_total, 10_000);
        assert_eq!(header2.get_run_label(), "large_snapshot_test");
    }
}
