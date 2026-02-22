//! Binary snapshot storage for Janus PM simulations
//!
//! Simple binary format, compatible with numpy.fromfile()

use crate::janus_pm::JanusParticle;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write, Result as IoResult};
use std::path::Path;

/// Snapshot metadata
#[derive(Clone)]
pub struct SnapshotMeta {
    pub step: usize,
    pub time: f64,
    pub tau: f64,
    pub scale_factor: f64,
    pub segregation: f64,
    pub ke_ratio: f64,
    pub n_positive: usize,
    pub n_negative: usize,
}

/// Save full snapshot (binary: header + positions f64 + velocities f32 + signs i8)
///
/// Format:
///   - Header (64 bytes): step(u64), n(u64), time(f64), tau(f64), a(f64), seg(f64), ke_ratio(f64), padding
///   - pos_x[n] (f64)
///   - pos_y[n] (f64)
///   - pos_z[n] (f64)
///   - vel_x[n] (f32)
///   - vel_y[n] (f32)
///   - vel_z[n] (f32)
///   - sign[n] (i8)
pub fn save_snapshot_full(
    path: &Path,
    particles: &[JanusParticle],
    meta: &SnapshotMeta,
) -> IoResult<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let n = particles.len();

    // Header (64 bytes)
    w.write_u64::<LittleEndian>(meta.step as u64)?;
    w.write_u64::<LittleEndian>(n as u64)?;
    w.write_f64::<LittleEndian>(meta.time)?;
    w.write_f64::<LittleEndian>(meta.tau)?;
    w.write_f64::<LittleEndian>(meta.scale_factor)?;
    w.write_f64::<LittleEndian>(meta.segregation)?;
    w.write_f64::<LittleEndian>(meta.ke_ratio)?;
    w.write_u64::<LittleEndian>(0)?; // padding

    // Positions (f64)
    for p in particles {
        w.write_f64::<LittleEndian>(p.pos.0)?;
    }
    for p in particles {
        w.write_f64::<LittleEndian>(p.pos.1)?;
    }
    for p in particles {
        w.write_f64::<LittleEndian>(p.pos.2)?;
    }

    // Velocities (f32)
    for p in particles {
        w.write_f32::<LittleEndian>(p.vel.0)?;
    }
    for p in particles {
        w.write_f32::<LittleEndian>(p.vel.1)?;
    }
    for p in particles {
        w.write_f32::<LittleEndian>(p.vel.2)?;
    }

    // Signs (i8)
    for p in particles {
        w.write_i8(p.sign)?;
    }

    w.flush()?;
    Ok(())
}

/// Save light snapshot (f32 positions + i8 signs only)
pub fn save_snapshot_light(
    path: &Path,
    particles: &[JanusParticle],
    meta: &SnapshotMeta,
) -> IoResult<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let n = particles.len();

    // Header (64 bytes)
    w.write_u64::<LittleEndian>(meta.step as u64)?;
    w.write_u64::<LittleEndian>(n as u64)?;
    w.write_f64::<LittleEndian>(meta.time)?;
    w.write_f64::<LittleEndian>(meta.tau)?;
    w.write_f64::<LittleEndian>(meta.scale_factor)?;
    w.write_f64::<LittleEndian>(meta.segregation)?;
    w.write_f64::<LittleEndian>(meta.ke_ratio)?;
    w.write_u64::<LittleEndian>(1)?; // light format marker

    // Positions (f32)
    for p in particles {
        w.write_f32::<LittleEndian>(p.pos.0 as f32)?;
    }
    for p in particles {
        w.write_f32::<LittleEndian>(p.pos.1 as f32)?;
    }
    for p in particles {
        w.write_f32::<LittleEndian>(p.pos.2 as f32)?;
    }

    // Signs (i8)
    for p in particles {
        w.write_i8(p.sign)?;
    }

    w.flush()?;
    Ok(())
}

/// Time series data writer (CSV format)
pub struct TimeSeriesWriter {
    file: File,
}

impl TimeSeriesWriter {
    pub fn new(path: &Path) -> IoResult<Self> {
        let mut file = File::create(path)?;
        writeln!(file, "step,time,tau,scale_factor,segregation,ke_ratio,n_positive,n_negative")?;
        Ok(Self { file })
    }

    pub fn write(&mut self, meta: &SnapshotMeta) -> IoResult<()> {
        writeln!(
            self.file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            meta.step, meta.time, meta.tau, meta.scale_factor,
            meta.segregation, meta.ke_ratio, meta.n_positive, meta.n_negative
        )
    }
}
