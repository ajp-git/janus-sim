/// GPU-accelerated Barnes-Hut N-body simulation using CUDA
///
/// Based on Bedorf et al. 2012: "A sparse octree gravitational N-body code
/// that runs entirely on the GPU processor"
///
/// Key optimizations:
/// - Linear tree array representation (no pointers)
/// - Separate positive/negative mass COM for Janus interactions
/// - Parallel tree traversal on GPU

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::nbody::{Vec3, BoundingBox};
use std::sync::Arc;

/// Simulation parameters passed to GPU
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SimParams {
    pub n_particles: i32,
    pub n_nodes: i32,
    pub theta: f32,
    pub softening: f32,
    pub box_half: f32,
    pub dt: f32,
    pub half_dt: f32,
    pub _pad: f32,
}

/// Linear tree node for GPU (Bedorf-style) - f64 precision
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct LinearTreeNode {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub half_size: f64,
    pub com_plus_x: f64,
    pub com_plus_y: f64,
    pub com_plus_z: f64,
    pub mass_plus: f64,
    pub com_minus_x: f64,
    pub com_minus_y: f64,
    pub com_minus_z: f64,
    pub mass_minus: f64,
    pub children: [u32; 8],
    pub node_type: u32,
    pub particle_idx: u32,
    pub _pad: [u32; 2],
}

/// Particle data in GPU-friendly format (f64 for precision)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct GpuParticle {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub mass: f64,
    pub sign: i32,
}

/// CUDA kernels using double precision (f64) for accuracy
const CUDA_KERNEL_SRC: &str = r#"
extern "C" __global__ void compute_forces_simple(
    const double* __restrict__ pos,      // x,y,z interleaved
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ node_children,
    const int* __restrict__ node_types,
    double* __restrict__ acc,            // ax,ay,az interleaved
    int n_particles,
    int n_nodes,
    double theta,
    double softening
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    double eps2 = softening * softening;

    // Stack for tree traversal
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0 || node_idx >= n_nodes) continue;

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int base = node_idx * 12;
        double cx = node_data[base + 0];
        double cy = node_data[base + 1];
        double cz = node_data[base + 2];
        double half_size = node_data[base + 3];

        double dx = cx - px;
        double dy = cy - py;
        double dz = cz - pz;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        if (node_type == 1 || s_over_r < theta) {
            double com_plus_x = node_data[base + 4];
            double com_plus_y = node_data[base + 5];
            double com_plus_z = node_data[base + 6];
            double mass_plus = node_data[base + 7];

            double com_minus_x = node_data[base + 8];
            double com_minus_y = node_data[base + 9];
            double com_minus_z = node_data[base + 10];
            double mass_minus = node_data[base + 11];

            if (mass_plus > 0.0) {
                double dpx = com_plus_x - px;
                double dpy = com_plus_y - py;
                double dpz = com_plus_z - pz;
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));  // Fixed: was rsqrt(rp2)/rp2
                double interaction = (my_sign > 0) ? 1.0 : -1.0;
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            if (mass_minus > 0.0) {
                double dmx = com_minus_x - px;
                double dmy = com_minus_y - py;
                double dmz = com_minus_z - pz;
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));  // Fixed: was rsqrt(rm2)/rm2
                double interaction = (my_sign < 0) ? 1.0 : -1.0;
                double f = interaction * mass_minus * inv_rm3;
                ax += f * dmx;
                ay += f * dmy;
                az += f * dmz;
            }
        } else {
            int child_base = node_idx * 8;
            for (int i = 0; i < 8; i++) {
                int child = node_children[child_base + i];
                if (child > 0 && stack_ptr < 32) {
                    stack[stack_ptr++] = child - 1;
                }
            }
        }
    }

    acc[tid * 3] = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

extern "C" __global__ void leapfrog_kick_drift(
    double* __restrict__ pos,
    double* __restrict__ vel,
    const double* __restrict__ acc,
    double half_dt,
    double dt,
    double box_half,
    int n,
    int do_drift,
    double scale_factor,    // a(t) - facteur d'echelle cosmologique
    double hubble_param     // H(t) = adot/a - parametre de Hubble
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 3;

    // Kick avec Hubble friction (coordonnees comobiles)
    // dv/dt = g_Janus / a^3 - 2*H*v
    double a3 = scale_factor * scale_factor * scale_factor;

    for (int d = 0; d < 3; d++) {
        // 1. Acceleration Janus en coordonnees comobiles
        double accel_comoving = acc[base + d] / a3;

        // 2. Terme de Hubble friction: -2*H*v
        double friction = -2.0 * hubble_param * vel[base + d];

        // 3. Mise a jour de la vitesse (Kick)
        vel[base + d] += (accel_comoving + friction) * half_dt;
    }

    // Drift (only if do_drift)
    if (do_drift) {
        pos[base]     += vel[base]     * dt;
        pos[base + 1] += vel[base + 1] * dt;
        pos[base + 2] += vel[base + 2] * dt;

        // Periodic BC
        for (int i = 0; i < 3; i++) {
            if (pos[base + i] > box_half) pos[base + i] -= 2.0 * box_half;
            if (pos[base + i] < -box_half) pos[base + i] += 2.0 * box_half;
        }
    }
}
"#;

/// Linear octree for GPU Barnes-Hut
pub struct LinearOctree {
    pub nodes: Vec<LinearTreeNode>,
    pub node_data: Vec<f64>,
    pub node_children: Vec<i32>,
    pub node_types: Vec<i32>,
    pub bounds: BoundingBox,
}

impl LinearOctree {
    pub fn build(particles: &[GpuParticle], box_size: f64) -> Self {
        let half = box_size / 2.0;
        let bounds = BoundingBox::new(
            Vec3::new(-half, -half, -half),
            Vec3::new(half, half, half),
        );

        let max_nodes = particles.len() * 2 + 1000;
        let mut nodes = Vec::with_capacity(max_nodes);

        nodes.push(LinearTreeNode {
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            half_size: half,
            ..Default::default()
        });

        for (idx, p) in particles.iter().enumerate() {
            Self::insert(&mut nodes, 0, idx, p, 0.0, 0.0, 0.0, half, particles);
        }

        Self::compute_com(&mut nodes, 0, particles);

        let n = nodes.len();
        let mut node_data = vec![0.0f64; n * 12];
        let mut node_children = vec![0i32; n * 8];
        let mut node_types = vec![0i32; n];

        for (i, node) in nodes.iter().enumerate() {
            let base = i * 12;
            node_data[base + 0] = node.center_x;
            node_data[base + 1] = node.center_y;
            node_data[base + 2] = node.center_z;
            node_data[base + 3] = node.half_size;
            node_data[base + 4] = node.com_plus_x;
            node_data[base + 5] = node.com_plus_y;
            node_data[base + 6] = node.com_plus_z;
            node_data[base + 7] = node.mass_plus;
            node_data[base + 8] = node.com_minus_x;
            node_data[base + 9] = node.com_minus_y;
            node_data[base + 10] = node.com_minus_z;
            node_data[base + 11] = node.mass_minus;

            let child_base = i * 8;
            for j in 0..8 {
                node_children[child_base + j] = node.children[j] as i32;
            }
            node_types[i] = node.node_type as i32;
        }

        Self { nodes, node_data, node_children, node_types, bounds }
    }

    fn insert(nodes: &mut Vec<LinearTreeNode>, node_idx: usize, particle_idx: usize,
              particle: &GpuParticle, cx: f64, cy: f64, cz: f64, half_size: f64,
              particles: &[GpuParticle]) {

        while nodes.len() <= node_idx {
            nodes.push(LinearTreeNode::default());
        }

        let node_type = nodes[node_idx].node_type;

        match node_type {
            0 => {
                nodes[node_idx].node_type = 1;
                nodes[node_idx].particle_idx = particle_idx as u32;
                nodes[node_idx].center_x = cx;
                nodes[node_idx].center_y = cy;
                nodes[node_idx].center_z = cz;
                nodes[node_idx].half_size = half_size;
            }
            1 => {
                // Converting leaf to internal node - must re-insert existing particle
                let existing_idx = nodes[node_idx].particle_idx as usize;
                let existing_p = &particles[existing_idx];

                nodes[node_idx].node_type = 2;
                nodes[node_idx].particle_idx = 0;

                let base_child = nodes.len();
                for i in 0..8 {
                    nodes.push(LinearTreeNode {
                        center_x: cx + if i & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        center_y: cy + if i & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        center_z: cz + if i & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        half_size: half_size / 2.0,
                        ..Default::default()
                    });
                    nodes[node_idx].children[i] = (base_child + i + 1) as u32;
                }

                // Re-insert the existing particle into appropriate child
                let oct_exist = Self::get_octant(existing_p.x, existing_p.y, existing_p.z, cx, cy, cz);
                let child_exist = (nodes[node_idx].children[oct_exist] - 1) as usize;
                let cx_e = cx + if oct_exist & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let cy_e = cy + if oct_exist & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let cz_e = cz + if oct_exist & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                Self::insert(nodes, child_exist, existing_idx, existing_p, cx_e, cy_e, cz_e, half_size / 2.0, particles);

                // Insert the new particle into appropriate child
                let octant = Self::get_octant(particle.x, particle.y, particle.z, cx, cy, cz);
                let child_idx = (nodes[node_idx].children[octant] - 1) as usize;
                let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                Self::insert(nodes, child_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
            }
            2 => {
                let octant = Self::get_octant(particle.x, particle.y, particle.z, cx, cy, cz);
                let child = nodes[node_idx].children[octant];

                if child == 0 {
                    let new_idx = nodes.len();
                    let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };

                    nodes.push(LinearTreeNode {
                        center_x: new_cx,
                        center_y: new_cy,
                        center_z: new_cz,
                        half_size: half_size / 2.0,
                        ..Default::default()
                    });
                    nodes[node_idx].children[octant] = (new_idx + 1) as u32;
                    Self::insert(nodes, new_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
                } else {
                    let child_idx = (child - 1) as usize;
                    let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    Self::insert(nodes, child_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
                }
            }
            _ => {}
        }
    }

    fn get_octant(x: f64, y: f64, z: f64, cx: f64, cy: f64, cz: f64) -> usize {
        let mut oct = 0;
        if x > cx { oct |= 1; }
        if y > cy { oct |= 2; }
        if z > cz { oct |= 4; }
        oct
    }

    fn compute_com(nodes: &mut Vec<LinearTreeNode>, node_idx: usize, particles: &[GpuParticle])
        -> (f64, f64, f64, f64, f64, f64, f64, f64)
    {
        if node_idx >= nodes.len() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let node_type = nodes[node_idx].node_type;

        match node_type {
            0 => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1 => {
                let p_idx = nodes[node_idx].particle_idx as usize;
                if p_idx < particles.len() {
                    let p = &particles[p_idx];
                    if p.sign > 0 {
                        nodes[node_idx].com_plus_x = p.x;
                        nodes[node_idx].com_plus_y = p.y;
                        nodes[node_idx].com_plus_z = p.z;
                        nodes[node_idx].mass_plus = p.mass;
                        (p.x * p.mass, p.y * p.mass, p.z * p.mass, p.mass, 0.0, 0.0, 0.0, 0.0)
                    } else {
                        nodes[node_idx].com_minus_x = p.x;
                        nodes[node_idx].com_minus_y = p.y;
                        nodes[node_idx].com_minus_z = p.z;
                        nodes[node_idx].mass_minus = p.mass;
                        (0.0, 0.0, 0.0, 0.0, p.x * p.mass, p.y * p.mass, p.z * p.mass, p.mass)
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                }
            }
            2 => {
                let mut sum_px = 0.0f64;
                let mut sum_py = 0.0f64;
                let mut sum_pz = 0.0f64;
                let mut sum_mp = 0.0f64;
                let mut sum_mx = 0.0f64;
                let mut sum_my = 0.0f64;
                let mut sum_mz = 0.0f64;
                let mut sum_mm = 0.0f64;

                for i in 0..8 {
                    let child = nodes[node_idx].children[i];
                    if child > 0 {
                        let (cpx, cpy, cpz, cmp, cmx, cmy, cmz, cmm) =
                            Self::compute_com(nodes, (child - 1) as usize, particles);
                        sum_px += cpx;
                        sum_py += cpy;
                        sum_pz += cpz;
                        sum_mp += cmp;
                        sum_mx += cmx;
                        sum_my += cmy;
                        sum_mz += cmz;
                        sum_mm += cmm;
                    }
                }

                if sum_mp > 0.0 {
                    nodes[node_idx].com_plus_x = sum_px / sum_mp;
                    nodes[node_idx].com_plus_y = sum_py / sum_mp;
                    nodes[node_idx].com_plus_z = sum_pz / sum_mp;
                    nodes[node_idx].mass_plus = sum_mp;
                }
                if sum_mm > 0.0 {
                    nodes[node_idx].com_minus_x = sum_mx / sum_mm;
                    nodes[node_idx].com_minus_y = sum_my / sum_mm;
                    nodes[node_idx].com_minus_z = sum_mz / sum_mm;
                    nodes[node_idx].mass_minus = sum_mm;
                }

                (sum_px, sum_py, sum_pz, sum_mp, sum_mx, sum_my, sum_mz, sum_mm)
            }
            _ => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Compute gravitational potential at position (x,y,z) from same-sign particles only.
    /// Uses Barnes-Hut approximation with opening angle theta.
    /// Returns potential (negative for attractive interactions).
    pub fn compute_potential_same_sign(
        &self,
        x: f64, y: f64, z: f64,
        particle_sign: i32,
        particle_idx: usize,
        softening: f64,
        theta: f64,
    ) -> f64 {
        self.potential_tree_walk(0, x, y, z, particle_sign, particle_idx, softening, theta)
    }

    fn potential_tree_walk(
        &self,
        node_idx: usize,
        x: f64, y: f64, z: f64,
        particle_sign: i32,
        exclude_idx: usize,
        softening: f64,
        theta: f64,
    ) -> f64 {
        if node_idx >= self.nodes.len() {
            return 0.0;
        }

        let node = &self.nodes[node_idx];

        match node.node_type {
            0 => 0.0, // Empty node
            1 => {
                // Leaf node - single particle
                let p_idx = node.particle_idx as usize;
                if p_idx == exclude_idx {
                    return 0.0; // Skip self
                }

                // Get position and sign from node's COM
                let (com_x, com_y, com_z, mass) = if particle_sign > 0 {
                    // Looking for positive mass contribution
                    if node.mass_plus > 0.0 {
                        (node.com_plus_x, node.com_plus_y, node.com_plus_z, node.mass_plus)
                    } else {
                        return 0.0; // This leaf is a negative particle
                    }
                } else {
                    // Looking for negative mass contribution
                    if node.mass_minus > 0.0 {
                        (node.com_minus_x, node.com_minus_y, node.com_minus_z, node.mass_minus)
                    } else {
                        return 0.0; // This leaf is a positive particle
                    }
                };

                let dx = com_x - x;
                let dy = com_y - y;
                let dz = com_z - z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = (r2 + softening * softening).sqrt();

                -mass / r  // Plummer potential (negative = attractive)
            }
            2 => {
                // Internal node
                // Get same-sign mass and COM
                let (com_x, com_y, com_z, mass) = if particle_sign > 0 {
                    (node.com_plus_x, node.com_plus_y, node.com_plus_z, node.mass_plus)
                } else {
                    (node.com_minus_x, node.com_minus_y, node.com_minus_z, node.mass_minus)
                };

                if mass <= 0.0 {
                    return 0.0; // No same-sign particles in this branch
                }

                let dx = com_x - x;
                let dy = com_y - y;
                let dz = com_z - z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();

                // Barnes-Hut criterion: s/d < theta
                let s = 2.0 * node.half_size;
                if s / r < theta {
                    // Use monopole approximation
                    let r_soft = (r2 + softening * softening).sqrt();
                    -mass / r_soft
                } else {
                    // Descend into children
                    let mut potential = 0.0;
                    for i in 0..8 {
                        let child = node.children[i];
                        if child > 0 {
                            potential += self.potential_tree_walk(
                                (child - 1) as usize,
                                x, y, z,
                                particle_sign,
                                exclude_idx,
                                softening,
                                theta,
                            );
                        }
                    }
                    potential
                }
            }
            _ => 0.0,
        }
    }
}

/// GPU Barnes-Hut simulation (f64 precision)
#[cfg(feature = "cuda")]
pub struct GpuNBodySimulation {
    device: Arc<CudaDevice>,
    pos: CudaSlice<f64>,        // Interleaved x,y,z
    vel: CudaSlice<f64>,        // Interleaved vx,vy,vz
    signs: CudaSlice<i32>,
    acc: CudaSlice<f64>,        // Interleaved ax,ay,az
    node_data: CudaSlice<f64>,
    node_children: CudaSlice<i32>,
    node_types: CudaSlice<i32>,
    n_particles: usize,
    n_nodes: usize,
    theta: f64,
    softening: f64,
    box_size: f64,
    time: f64,
    particles_cpu: Vec<GpuParticle>,
}

#[cfg(feature = "cuda")]
impl GpuNBodySimulation {
    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "nbody", &["compute_forces_simple", "leapfrog_kick_drift"])?;

        let n_total = n_positive + n_negative;

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let mut particles_cpu = Vec::with_capacity(n_total);

        let virial_velocity = ((n_total as f64) / box_size).sqrt() * 0.3;

        // f64 precision throughout
        for _ in 0..n_positive {
            particles_cpu.push(GpuParticle {
                x: (rng.random::<f64>() - 0.5) * box_size,
                y: (rng.random::<f64>() - 0.5) * box_size,
                z: (rng.random::<f64>() - 0.5) * box_size,
                vx: (rng.random::<f64>() - 0.5) * virial_velocity,
                vy: (rng.random::<f64>() - 0.5) * virial_velocity,
                vz: (rng.random::<f64>() - 0.5) * virial_velocity,
                mass: 1.0,
                sign: 1,
            });
        }

        for _ in 0..n_negative {
            particles_cpu.push(GpuParticle {
                x: (rng.random::<f64>() - 0.5) * box_size,
                y: (rng.random::<f64>() - 0.5) * box_size,
                z: (rng.random::<f64>() - 0.5) * box_size,
                vx: (rng.random::<f64>() - 0.5) * virial_velocity,
                vy: (rng.random::<f64>() - 0.5) * virial_velocity,
                vz: (rng.random::<f64>() - 0.5) * virial_velocity,
                mass: 1.0,
                sign: -1,
            });
        }

        // Interleaved position/velocity arrays (f64)
        let mut pos_data = Vec::with_capacity(n_total * 3);
        let mut vel_data = Vec::with_capacity(n_total * 3);
        let mut signs_data = Vec::with_capacity(n_total);

        for p in &particles_cpu {
            pos_data.push(p.x);
            pos_data.push(p.y);
            pos_data.push(p.z);
            vel_data.push(p.vx);
            vel_data.push(p.vy);
            vel_data.push(p.vz);
            signs_data.push(p.sign);
        }

        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        let tree = LinearOctree::build(&particles_cpu, box_size);
        let n_nodes = tree.nodes.len();
        let node_data = device.htod_sync_copy(&tree.node_data)?;
        let node_children = device.htod_sync_copy(&tree.node_children)?;
        let node_types = device.htod_sync_copy(&tree.node_types)?;

        let mean_sep = box_size / (n_total as f64).powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Ok(Self {
            device,
            pos, vel, signs, acc,
            node_data, node_children, node_types,
            n_particles: n_total,
            n_nodes,
            theta: 0.7,
            softening,
            box_size,
            time: 0.0,
            particles_cpu,
        })
    }

    /// Step sans expansion cosmologique (a=1, H=0)
    pub fn step(&mut self, dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.step_with_expansion(dt, 1.0, 0.0)
    }

    /// Step avec expansion cosmologique
    /// scale_factor: a(t) facteur d'echelle
    /// hubble: H(t) = adot/a parametre de Hubble
    pub fn step_with_expansion(&mut self, dt: f64, scale_factor: f64, hubble: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;

        // Download positions for tree rebuild
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;

        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Rebuild tree
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        let blocks = (self.n_particles + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel functions
        let compute_forces = self.device.get_func("nbody", "compute_forces_simple")
            .ok_or("Failed to get compute_forces_simple kernel")?;
        let leapfrog = self.device.get_func("nbody", "leapfrog_kick_drift")
            .ok_or("Failed to get leapfrog_kick_drift kernel")?;

        // Compute forces (10 args - within tuple limit)
        unsafe {
            compute_forces.clone().launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32, self.n_nodes as i32,
                self.theta, self.softening,
            ))?;
        }

        // Kick + Drift avec parametres cosmologiques
        unsafe {
            leapfrog.clone().launch(cfg, (
                &mut self.pos, &mut self.vel, &self.acc,
                half_dt, dt, box_half,
                self.n_particles as i32, 1i32, // do_drift = 1
                scale_factor, hubble,
            ))?;
        }

        // Download new positions for second force calc
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Rebuild tree
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        // Compute forces again
        unsafe {
            compute_forces.clone().launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32, self.n_nodes as i32,
                self.theta, self.softening,
            ))?;
        }

        // Kick only (no drift) avec parametres cosmologiques
        unsafe {
            leapfrog.launch(cfg, (
                &mut self.pos, &mut self.vel, &self.acc,
                half_dt, 0.0f64, box_half,
                self.n_particles as i32, 0i32, // do_drift = 0
                scale_factor, hubble,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    pub fn kinetic_energy(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let vel = self.device.dtoh_sync_copy(&self.vel)?;

        let ke: f64 = (0..self.n_particles)
            .map(|i| {
                let vx = vel[i * 3];
                let vy = vel[i * 3 + 1];
                let vz = vel[i * 3 + 2];
                0.5 * (vx * vx + vy * vy + vz * vz)
            })
            .sum();

        Ok(ke)
    }

    /// Compute binding potential energy (same-sign pairs only, always negative)
    /// Uses Barnes-Hut tree for O(N log N) complexity.
    pub fn potential_energy_binding(&self) -> Result<f64, Box<dyn std::error::Error>> {
        use rayon::prelude::*;

        // Build tree from CPU particle data
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);

        // Compute potential for each particle using tree traversal
        // Use theta = 0.5 for better accuracy in PE calculation
        let theta_pe = 0.5;

        let pe_bind: f64 = (0..self.n_particles)
            .into_par_iter()
            .map(|i| {
                let p = &self.particles_cpu[i];
                tree.compute_potential_same_sign(
                    p.x, p.y, p.z,
                    p.sign,
                    i,
                    self.softening,
                    theta_pe,
                )
            })
            .sum();

        // Divide by 2 to avoid double counting (each pair counted twice)
        Ok(pe_bind / 2.0)
    }

    /// Virialize velocities to satisfy 2KE + PE_binding = 0
    /// Call this once at t=0 after initialization.
    pub fn virialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ke = self.kinetic_energy()?;
        let pe_bind = self.potential_energy_binding()?;

        println!("Virialization (GPU, Janus mode):");
        println!("  KE initial    = {:.4e}", ke);
        println!("  PE binding    = {:.4e}", pe_bind);

        if ke < 1e-20 {
            println!("  WARNING: KE too small, skipping virialization");
            return Ok(());
        }

        if pe_bind >= 0.0 {
            println!("  WARNING: PE_binding >= 0, no bound system to virialize");
            return Ok(());
        }

        // Virial condition: 2KE + PE_bind = 0 → KE_target = |PE_bind|/2
        let ke_target = pe_bind.abs() / 2.0;
        let alpha = (ke_target / ke).sqrt();

        println!("  KE target     = {:.4e}", ke_target);
        println!("  Alpha scale   = {:.6}", alpha);

        // Scale velocities on GPU
        let mut vel = self.device.dtoh_sync_copy(&self.vel)?;
        for v in vel.iter_mut() {
            *v *= alpha;
        }
        self.vel = self.device.htod_sync_copy(&vel)?;

        // Also update CPU copy
        for p in self.particles_cpu.iter_mut() {
            p.vx *= alpha;
            p.vy *= alpha;
            p.vz *= alpha;
        }

        // Verify
        let ke_after = self.kinetic_energy()?;
        let virial_ratio = 2.0 * ke_after + pe_bind;
        let virial_error = virial_ratio.abs() / pe_bind.abs();

        println!("  KE after      = {:.4e}", ke_after);
        println!("  2KE + PE_bind = {:.4e}", virial_ratio);
        println!("  Virial error  = {:.4}%", virial_error * 100.0);

        Ok(())
    }

    /// Compute segregation distance using minimum image convention
    /// and normalized by box_size for comparability across scales.
    ///
    /// Uses a SINGLE reference point for both populations to avoid bias.
    /// Computes relative positions using minimum image convention.
    pub fn segregation_distance(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let pos = self.device.dtoh_sync_copy(&self.pos)?;
        let signs = self.device.dtoh_sync_copy(&self.signs)?;
        let l = self.box_size;

        if self.n_particles == 0 {
            return Ok(0.0);
        }

        // Use first particle as COMMON reference for both populations
        // This is critical: using different references biases the result
        let ref_pos = (pos[0], pos[1], pos[2]);

        let mut sum_plus = (0.0f64, 0.0f64, 0.0f64);
        let mut n_plus = 0usize;
        let mut sum_minus = (0.0f64, 0.0f64, 0.0f64);
        let mut n_minus = 0usize;

        for i in 0..self.n_particles {
            let x = pos[i * 3];
            let y = pos[i * 3 + 1];
            let z = pos[i * 3 + 2];

            // Unwrap relative to common reference
            let ux = ref_pos.0 + minimum_image_gpu(x - ref_pos.0, l);
            let uy = ref_pos.1 + minimum_image_gpu(y - ref_pos.1, l);
            let uz = ref_pos.2 + minimum_image_gpu(z - ref_pos.2, l);

            if signs[i] > 0 {
                sum_plus.0 += ux;
                sum_plus.1 += uy;
                sum_plus.2 += uz;
                n_plus += 1;
            } else {
                sum_minus.0 += ux;
                sum_minus.1 += uy;
                sum_minus.2 += uz;
                n_minus += 1;
            }
        }

        // Compute COMs
        let com_plus = if n_plus > 0 {
            let n = n_plus as f64;
            (sum_plus.0 / n, sum_plus.1 / n, sum_plus.2 / n)
        } else {
            (0.0, 0.0, 0.0)
        };

        let com_minus = if n_minus > 0 {
            let n = n_minus as f64;
            (sum_minus.0 / n, sum_minus.1 / n, sum_minus.2 / n)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Minimum image distance between COMs
        let dx = minimum_image_gpu(com_plus.0 - com_minus.0, l);
        let dy = minimum_image_gpu(com_plus.1 - com_minus.1, l);
        let dz = minimum_image_gpu(com_plus.2 - com_minus.2, l);

        // Normalize by box_size for comparability
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        Ok(distance / l)
    }

    pub fn get_positions(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let pos = self.device.dtoh_sync_copy(&self.pos)?;
        Ok(pos)
    }

    pub fn get_velocities(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let vel = self.device.dtoh_sync_copy(&self.vel)?;
        Ok(vel)
    }

    pub fn get_signs(&self) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let signs = self.device.dtoh_sync_copy(&self.signs)?;
        Ok(signs)
    }
}

/// Minimum image convention for periodic boundaries
fn minimum_image_gpu(dx: f64, box_size: f64) -> f64 {
    let mut d = dx;
    while d > box_size / 2.0 { d -= box_size; }
    while d < -box_size / 2.0 { d += box_size; }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_octree_build() {
        let particles = vec![
            GpuParticle { x: 10.0, y: 10.0, z: 10.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0, sign: 1 },
            GpuParticle { x: -10.0, y: -10.0, z: -10.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0, sign: -1 },
        ];

        let tree = LinearOctree::build(&particles, 100.0);
        assert!(tree.nodes.len() >= 2);
    }

    #[test]
    fn test_octant_calculation() {
        assert_eq!(LinearOctree::get_octant(1.0, 1.0, 1.0, 0.0, 0.0, 0.0), 7);
        assert_eq!(LinearOctree::get_octant(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0), 0);
    }
}
