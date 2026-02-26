# Janus N-Body: Metal Port for Mac M4

**Target**: 85-100M particles on Mac M4 (48GB unified memory)
**API**: Metal 3 + metal-rs
**Performance target**: <5s/step for 100M particles

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY (48GB)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Positions   │  │  Velocities  │  │    Tree      │         │
│  │   (2.4 GB)   │  │   (2.4 GB)   │  │   (5 GB)     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│    ┌────┴─────────────────┴─────────────────┴────┐            │
│    │              ZERO-COPY ACCESS               │            │
│    └────┬─────────────────┬─────────────────┬────┘            │
│         │                 │                 │                  │
│    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐            │
│    │   CPU   │       │   GPU   │       │   NPU   │            │
│    │ (Rayon) │       │ (Metal) │       │ (future)│            │
│    └─────────┘       └─────────┘       └─────────┘            │
└────────────────────────────────────────────────────────────────┘
```

**Key advantage**: No PCIe transfers. CPU and GPU read/write the same memory.

---

## 2. Project Structure

```
janus-metal/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── simulation.rs      # Main simulation struct
│   ├── particle.rs        # SoA particle data
│   ├── tree.rs            # Octree (CPU build)
│   ├── metal_backend.rs   # Metal integration
│   └── shaders/
│       ├── mod.rs
│       ├── force.metal    # Tree traversal kernel
│       ├── tree.metal     # Tree build kernels (optional)
│       └── reduce.metal   # Reductions (KE, segregation)
├── build.rs               # Compile .metal to .metallib
└── tests/
    └── validation.rs      # Compare with CUDA results
```

---

## 3. Cargo.toml

```toml
[package]
name = "janus-metal"
version = "0.1.0"
edition = "2021"

[dependencies]
# Metal bindings
metal = "0.28"
objc = "0.2"
block = "0.1"

# Parallelism
rayon = "1.10"

# Math
nalgebra = "0.33"

# Random
rand = "0.8"
rand_distr = "0.4"

# Serialization
serde = { version = "1.0", features = ["derive"] }
csv = "1.3"

[build-dependencies]
cc = "1.0"

[profile.release]
opt-level = 3
lto = true
```

---

## 4. Metal Shader: Force Computation

```metal
// shaders/force.metal
// Barnes-Hut tree traversal on GPU

#include <metal_stdlib>
using namespace metal;

// Tree node structure (must match Rust layout exactly)
struct TreeNode {
    float3 center_of_mass;  // 12 bytes
    float mass_positive;    // 4 bytes
    float mass_negative;    // 4 bytes
    float size;             // 4 bytes
    uint children[8];       // 32 bytes (0 = no child)
    uint particle_start;    // 4 bytes (for leaves)
    uint particle_count;    // 4 bytes
    // Total: 64 bytes (cache-aligned)
};

// Simulation parameters
struct Params {
    float eta;              // Mass ratio (1.045)
    float theta;            // Opening angle (0.5)
    float softening;        // Plummer softening (0.1)
    float G;                // Gravitational constant
    uint n_particles;
    uint n_nodes;
};

// Force computation kernel
kernel void compute_forces(
    device const float3* positions [[buffer(0)]],
    device const int8_t* signs [[buffer(1)]],
    device const TreeNode* tree [[buffer(2)]],
    device float3* accelerations [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (id >= params.n_particles) return;

    float3 pos = positions[id];
    int8_t sign = signs[id];
    float3 acc = float3(0.0);

    float theta2 = params.theta * params.theta;
    float soft2 = params.softening * params.softening;

    // Stack-based tree traversal (no recursion in Metal)
    uint stack[64];  // Max depth ~20 for 100M particles
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Root node

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];
        TreeNode node = tree[node_idx];

        // Skip empty nodes
        float total_mass = node.mass_positive + node.mass_negative;
        if (total_mass < 1e-10) continue;

        // Distance to node center of mass
        float3 d = node.center_of_mass - pos;
        float r2 = dot(d, d);

        // Opening criterion: (size/r)^2 < theta^2
        float size2 = node.size * node.size;
        bool is_leaf = (node.children[0] == 0);  // All children zero = leaf

        if (size2 < theta2 * r2 || is_leaf) {
            // Use this node (monopole approximation)
            float r2_soft = r2 + soft2;
            float r = sqrt(r2_soft);
            float r3 = r2_soft * r;

            // Janus effective mass
            float mass_eff;
            if (sign > 0) {
                // Positive particle: attracted by +, repelled by -
                mass_eff = node.mass_positive - params.eta * node.mass_negative;
            } else {
                // Negative particle: attracted by - (scaled), repelled by +
                mass_eff = params.eta * node.mass_negative - node.mass_positive;
            }

            acc += params.G * mass_eff * d / r3;
        } else {
            // Open node: push children onto stack
            for (int c = 7; c >= 0; c--) {  // Reverse order for stack
                uint child = node.children[c];
                if (child != 0) {
                    stack[stack_ptr++] = child;
                }
            }
        }
    }

    accelerations[id] = acc;
}

// Kinetic energy reduction (parallel sum)
kernel void compute_kinetic_energy(
    device const float3* velocities [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& n_particles [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];

    float ke = 0.0;
    if (id < n_particles) {
        float3 v = velocities[id];
        ke = 0.5 * dot(v, v);  // mass = 1
    }

    shared_data[local_id] = ke;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] += shared_data[local_id + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        partial_sums[group_id] = shared_data[0];
    }
}

// Segregation distance computation
kernel void compute_segregation(
    device const float3* positions [[buffer(0)]],
    device const int8_t* signs [[buffer(1)]],
    device float4* partial_sums [[buffer(2)]],  // (sum_pos, count_pos, sum_neg, count_neg)
    constant uint& n_particles [[buffer(3)]],
    constant float& box_size [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float3 shared_pos[256];
    threadgroup float3 shared_neg[256];
    threadgroup uint shared_count_pos[256];
    threadgroup uint shared_count_neg[256];

    float3 pos_sum = float3(0.0);
    float3 neg_sum = float3(0.0);
    uint count_pos = 0;
    uint count_neg = 0;

    if (id < n_particles) {
        float3 p = positions[id];
        if (signs[id] > 0) {
            pos_sum = p;
            count_pos = 1;
        } else {
            neg_sum = p;
            count_neg = 1;
        }
    }

    shared_pos[local_id] = pos_sum;
    shared_neg[local_id] = neg_sum;
    shared_count_pos[local_id] = count_pos;
    shared_count_neg[local_id] = count_neg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_pos[local_id] += shared_pos[local_id + s];
            shared_neg[local_id] += shared_neg[local_id + s];
            shared_count_pos[local_id] += shared_count_pos[local_id + s];
            shared_count_neg[local_id] += shared_count_neg[local_id + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        // Pack results: use partial_sums as (pos.x, pos.y, pos.z, count_pos), etc.
        // This is a simplification; real implementation needs more buffers
        partial_sums[group_id * 2] = float4(shared_pos[0], float(shared_count_pos[0]));
        partial_sums[group_id * 2 + 1] = float4(shared_neg[0], float(shared_count_neg[0]));
    }
}
```

---

## 5. Rust Metal Integration

```rust
// src/metal_backend.rs

use metal::*;
use std::mem;
use std::path::Path;

/// Metal compute context for Janus simulation
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,

    // Compute pipelines
    force_pipeline: ComputePipelineState,
    ke_pipeline: ComputePipelineState,
    seg_pipeline: ComputePipelineState,

    // Shared buffers (CPU + GPU access)
    positions: Buffer,
    velocities: Buffer,
    accelerations: Buffer,
    signs: Buffer,
    tree: Buffer,
    params: Buffer,

    // Work buffers
    partial_sums: Buffer,

    n_particles: usize,
    n_nodes: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    eta: f32,
    theta: f32,
    softening: f32,
    g: f32,
    n_particles: u32,
    n_nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TreeNode {
    pub center_of_mass: [f32; 3],
    pub mass_positive: f32,
    pub mass_negative: f32,
    pub size: f32,
    pub children: [u32; 8],
    pub particle_start: u32,
    pub particle_count: u32,
}

impl MetalContext {
    pub fn new(n_particles: usize) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        println!("Metal device: {}", device.name());
        println!("Unified memory: {}", device.has_unified_memory());
        println!("Max buffer size: {} GB", device.max_buffer_length() / 1_000_000_000);

        let queue = device.new_command_queue();

        // Load compiled shader library
        let library_path = Path::new("shaders/janus.metallib");
        let library = device.new_library_with_file(library_path)
            .map_err(|e| format!("Failed to load shader library: {:?}", e))?;

        // Create compute pipelines
        let force_fn = library.get_function("compute_forces", None)
            .map_err(|e| format!("Failed to get compute_forces: {:?}", e))?;
        let force_pipeline = device.new_compute_pipeline_state_with_function(&force_fn)
            .map_err(|e| format!("Failed to create force pipeline: {:?}", e))?;

        let ke_fn = library.get_function("compute_kinetic_energy", None)
            .map_err(|e| format!("Failed to get compute_kinetic_energy: {:?}", e))?;
        let ke_pipeline = device.new_compute_pipeline_state_with_function(&ke_fn)
            .map_err(|e| format!("Failed to create KE pipeline: {:?}", e))?;

        let seg_fn = library.get_function("compute_segregation", None)
            .map_err(|e| format!("Failed to get compute_segregation: {:?}", e))?;
        let seg_pipeline = device.new_compute_pipeline_state_with_function(&seg_fn)
            .map_err(|e| format!("Failed to create segregation pipeline: {:?}", e))?;

        // Allocate shared buffers (StorageModeShared = unified memory)
        let pos_size = (n_particles * 3 * mem::size_of::<f32>()) as u64;
        let vel_size = pos_size;
        let acc_size = pos_size;
        let sign_size = n_particles as u64;

        // Estimate tree size: ~2N nodes for octree
        let n_nodes = n_particles * 2;
        let tree_size = (n_nodes * mem::size_of::<TreeNode>()) as u64;

        let options = MTLResourceOptions::StorageModeShared;

        let positions = device.new_buffer(pos_size, options);
        let velocities = device.new_buffer(vel_size, options);
        let accelerations = device.new_buffer(acc_size, options);
        let signs = device.new_buffer(sign_size, options);
        let tree = device.new_buffer(tree_size, options);
        let params = device.new_buffer(mem::size_of::<Params>() as u64, options);

        // Work buffer for reductions
        let n_groups = (n_particles + 255) / 256;
        let partial_size = (n_groups * mem::size_of::<f32>()) as u64;
        let partial_sums = device.new_buffer(partial_size * 4, options);  // Extra for segregation

        Ok(Self {
            device,
            queue,
            force_pipeline,
            ke_pipeline,
            seg_pipeline,
            positions,
            velocities,
            accelerations,
            signs,
            tree,
            params,
            partial_sums,
            n_particles,
            n_nodes,
        })
    }

    /// Get mutable slice of positions (zero-copy)
    pub fn positions_mut(&self) -> &mut [[f32; 3]] {
        unsafe {
            let ptr = self.positions.contents() as *mut [f32; 3];
            std::slice::from_raw_parts_mut(ptr, self.n_particles)
        }
    }

    /// Get mutable slice of velocities (zero-copy)
    pub fn velocities_mut(&self) -> &mut [[f32; 3]] {
        unsafe {
            let ptr = self.velocities.contents() as *mut [f32; 3];
            std::slice::from_raw_parts_mut(ptr, self.n_particles)
        }
    }

    /// Get mutable slice of signs (zero-copy)
    pub fn signs_mut(&self) -> &mut [i8] {
        unsafe {
            let ptr = self.signs.contents() as *mut i8;
            std::slice::from_raw_parts_mut(ptr, self.n_particles)
        }
    }

    /// Upload tree nodes (built on CPU)
    pub fn upload_tree(&self, nodes: &[TreeNode]) {
        assert!(nodes.len() <= self.n_nodes);
        unsafe {
            let ptr = self.tree.contents() as *mut TreeNode;
            std::ptr::copy_nonoverlapping(nodes.as_ptr(), ptr, nodes.len());
        }
    }

    /// Update simulation parameters
    pub fn set_params(&self, eta: f32, theta: f32, softening: f32, n_nodes: u32) {
        let params = Params {
            eta,
            theta,
            softening,
            g: 1.0,
            n_particles: self.n_particles as u32,
            n_nodes,
        };
        unsafe {
            let ptr = self.params.contents() as *mut Params;
            *ptr = params;
        }
    }

    /// Compute forces on GPU
    pub fn compute_forces(&self) {
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.force_pipeline);
        encoder.set_buffer(0, Some(&self.positions), 0);
        encoder.set_buffer(1, Some(&self.signs), 0);
        encoder.set_buffer(2, Some(&self.tree), 0);
        encoder.set_buffer(3, Some(&self.accelerations), 0);
        encoder.set_buffer(4, Some(&self.params), 0);

        let threads_per_group = self.force_pipeline.thread_execution_width();
        let n_groups = (self.n_particles as u64 + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(n_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Compute kinetic energy on GPU
    pub fn kinetic_energy(&self) -> f64 {
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ke_pipeline);
        encoder.set_buffer(0, Some(&self.velocities), 0);
        encoder.set_buffer(1, Some(&self.partial_sums), 0);

        let n_bytes = mem::size_of::<u32>() as u64;
        encoder.set_bytes(2, n_bytes, &(self.n_particles as u32) as *const u32 as *const _);

        let threads_per_group = 256u64;
        let n_groups = (self.n_particles as u64 + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(n_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Sum partial results on CPU
        unsafe {
            let ptr = self.partial_sums.contents() as *const f32;
            let partials = std::slice::from_raw_parts(ptr, n_groups as usize);
            partials.iter().map(|&x| x as f64).sum()
        }
    }

    /// Read accelerations (zero-copy, already in unified memory)
    pub fn accelerations(&self) -> &[[f32; 3]] {
        unsafe {
            let ptr = self.accelerations.contents() as *const [f32; 3];
            std::slice::from_raw_parts(ptr, self.n_particles)
        }
    }
}
```

---

## 6. CPU Octree Builder (Parallel with Rayon)

```rust
// src/tree.rs

use rayon::prelude::*;
use crate::metal_backend::TreeNode;

/// Octree for Barnes-Hut force computation
pub struct Octree {
    nodes: Vec<TreeNode>,
    root_size: f32,
}

impl Octree {
    pub fn new(max_particles: usize) -> Self {
        // Pre-allocate for ~2N nodes
        Self {
            nodes: Vec::with_capacity(max_particles * 2),
            root_size: 0.0,
        }
    }

    /// Build octree from particle positions (parallel)
    pub fn build(
        &mut self,
        positions: &[[f32; 3]],
        signs: &[i8],
        box_size: f32,
    ) {
        self.nodes.clear();
        self.root_size = box_size;

        // Create root node
        self.nodes.push(TreeNode {
            center_of_mass: [0.0; 3],
            mass_positive: 0.0,
            mass_negative: 0.0,
            size: box_size,
            children: [0; 8],
            particle_start: 0,
            particle_count: 0,
        });

        // Insert particles sequentially (tree structure requires this)
        for (i, pos) in positions.iter().enumerate() {
            self.insert(0, i, *pos, signs[i], [0.0, 0.0, 0.0], box_size);
        }

        // Update centers of mass (can be parallelized bottom-up)
        self.update_com_parallel(positions, signs);
    }

    fn insert(
        &mut self,
        node_idx: usize,
        particle_idx: usize,
        pos: [f32; 3],
        sign: i8,
        center: [f32; 3],
        size: f32,
    ) {
        let node = &mut self.nodes[node_idx];

        // Update mass
        if sign > 0 {
            node.mass_positive += 1.0;
        } else {
            node.mass_negative += 1.0;
        }

        // Determine octant
        let octant = Self::octant(pos, center);

        if node.children[octant] == 0 {
            // No child in this octant yet
            if node.particle_count == 0 {
                // Empty leaf: store particle here
                node.particle_start = particle_idx as u32;
                node.particle_count = 1;
            } else if node.particle_count == 1 {
                // Split: create child for existing particle
                let existing_idx = node.particle_start as usize;
                node.particle_start = 0;
                node.particle_count = 0;

                // Re-insert existing particle
                // (simplified: in real impl, need positions array)

                // Create new child node
                let child_idx = self.nodes.len();
                let child_center = Self::child_center(center, size, octant);
                self.nodes.push(TreeNode {
                    center_of_mass: [0.0; 3],
                    mass_positive: 0.0,
                    mass_negative: 0.0,
                    size: size / 2.0,
                    children: [0; 8],
                    particle_start: particle_idx as u32,
                    particle_count: 1,
                });

                self.nodes[node_idx].children[octant] = child_idx as u32;
            }
        } else {
            // Child exists: recurse
            let child_idx = node.children[octant] as usize;
            let child_center = Self::child_center(center, size, octant);
            self.insert(child_idx, particle_idx, pos, sign, child_center, size / 2.0);
        }
    }

    fn octant(pos: [f32; 3], center: [f32; 3]) -> usize {
        let mut idx = 0;
        if pos[0] > center[0] { idx |= 1; }
        if pos[1] > center[1] { idx |= 2; }
        if pos[2] > center[2] { idx |= 4; }
        idx
    }

    fn child_center(center: [f32; 3], size: f32, octant: usize) -> [f32; 3] {
        let q = size / 4.0;
        [
            center[0] + if octant & 1 != 0 { q } else { -q },
            center[1] + if octant & 2 != 0 { q } else { -q },
            center[2] + if octant & 4 != 0 { q } else { -q },
        ]
    }

    fn update_com_parallel(&mut self, positions: &[[f32; 3]], signs: &[i8]) {
        // Bottom-up update of centers of mass
        // This can be parallelized by level

        // For simplicity, do sequential post-order traversal
        self.update_com_recursive(0, positions, signs);
    }

    fn update_com_recursive(
        &mut self,
        node_idx: usize,
        positions: &[[f32; 3]],
        signs: &[i8],
    ) -> ([f32; 3], f32, f32) {
        let node = &self.nodes[node_idx];

        if node.particle_count == 1 {
            // Leaf: COM is particle position
            let p = positions[node.particle_start as usize];
            let (m_pos, m_neg) = if signs[node.particle_start as usize] > 0 {
                (1.0f32, 0.0f32)
            } else {
                (0.0f32, 1.0f32)
            };
            self.nodes[node_idx].center_of_mass = p;
            return (p, m_pos, m_neg);
        }

        let mut com = [0.0f32; 3];
        let mut total_mass = 0.0f32;
        let mut m_pos = 0.0f32;
        let mut m_neg = 0.0f32;

        for c in 0..8 {
            let child_idx = self.nodes[node_idx].children[c];
            if child_idx != 0 {
                let (child_com, child_m_pos, child_m_neg) =
                    self.update_com_recursive(child_idx as usize, positions, signs);

                let child_mass = child_m_pos + child_m_neg;
                com[0] += child_com[0] * child_mass;
                com[1] += child_com[1] * child_mass;
                com[2] += child_com[2] * child_mass;
                total_mass += child_mass;
                m_pos += child_m_pos;
                m_neg += child_m_neg;
            }
        }

        if total_mass > 0.0 {
            com[0] /= total_mass;
            com[1] /= total_mass;
            com[2] /= total_mass;
        }

        self.nodes[node_idx].center_of_mass = com;
        self.nodes[node_idx].mass_positive = m_pos;
        self.nodes[node_idx].mass_negative = m_neg;

        (com, m_pos, m_neg)
    }

    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}
```

---

## 7. Main Simulation Loop

```rust
// src/simulation.rs

use crate::metal_backend::MetalContext;
use crate::tree::Octree;
use std::time::Instant;

pub struct Simulation {
    ctx: MetalContext,
    tree: Octree,

    n_particles: usize,
    n_positive: usize,
    box_size: f32,
    eta: f32,
    theta: f32,
    dt: f32,
}

impl Simulation {
    pub fn new(n_particles: usize, eta: f32, theta: f32, dt: f32) -> Result<Self, String> {
        let n_positive = (n_particles as f64 / (1.0 + eta as f64)) as usize;
        let box_size = 100.0 * (n_particles as f32 / 100_000.0).powf(1.0/3.0);

        let ctx = MetalContext::new(n_particles)?;
        let tree = Octree::new(n_particles);

        Ok(Self {
            ctx,
            tree,
            n_particles,
            n_positive,
            box_size,
            eta,
            theta,
            dt,
        })
    }

    /// Initialize with uniform random positions, zero velocity
    pub fn initialize_cold(&mut self, seed: u64) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let positions = self.ctx.positions_mut();
        let signs = self.ctx.signs_mut();
        let velocities = self.ctx.velocities_mut();

        let half = self.box_size / 2.0;

        for i in 0..self.n_particles {
            positions[i] = [
                (rng.gen::<f32>() - 0.5) * self.box_size,
                (rng.gen::<f32>() - 0.5) * self.box_size,
                (rng.gen::<f32>() - 0.5) * self.box_size,
            ];
            velocities[i] = [0.0, 0.0, 0.0];
            signs[i] = if i < self.n_positive { 1 } else { -1 };
        }
    }

    /// One simulation step (DKD leapfrog)
    pub fn step(&mut self) -> StepStats {
        let t_total = Instant::now();

        // Drift (half step)
        let t_drift = Instant::now();
        self.drift(self.dt / 2.0);
        let drift_time = t_drift.elapsed().as_secs_f64();

        // Build tree (CPU)
        let t_tree = Instant::now();
        let positions = unsafe {
            std::slice::from_raw_parts(
                self.ctx.positions_mut().as_ptr(),
                self.n_particles
            )
        };
        let signs = unsafe {
            std::slice::from_raw_parts(
                self.ctx.signs_mut().as_ptr(),
                self.n_particles
            )
        };
        self.tree.build(positions, signs, self.box_size);
        self.ctx.upload_tree(self.tree.nodes());
        self.ctx.set_params(self.eta, self.theta, 0.1, self.tree.num_nodes() as u32);
        let tree_time = t_tree.elapsed().as_secs_f64();

        // Compute forces (GPU)
        let t_force = Instant::now();
        self.ctx.compute_forces();
        let force_time = t_force.elapsed().as_secs_f64();

        // Kick (full step)
        let t_kick = Instant::now();
        self.kick(self.dt);
        let kick_time = t_kick.elapsed().as_secs_f64();

        // Drift (half step)
        self.drift(self.dt / 2.0);

        let total_time = t_total.elapsed().as_secs_f64();

        StepStats {
            total_ms: total_time * 1000.0,
            tree_ms: tree_time * 1000.0,
            force_ms: force_time * 1000.0,
            drift_ms: drift_time * 2000.0,  // Two half-drifts
            kick_ms: kick_time * 1000.0,
        }
    }

    fn drift(&mut self, dt: f32) {
        let positions = self.ctx.positions_mut();
        let velocities = self.ctx.velocities_mut();
        let half = self.box_size / 2.0;

        // Parallel drift with Rayon
        use rayon::prelude::*;
        positions.par_iter_mut()
            .zip(velocities.par_iter())
            .for_each(|(pos, vel)| {
                pos[0] += vel[0] * dt;
                pos[1] += vel[1] * dt;
                pos[2] += vel[2] * dt;

                // Periodic boundary conditions
                for i in 0..3 {
                    if pos[i] > half { pos[i] -= self.box_size; }
                    if pos[i] < -half { pos[i] += self.box_size; }
                }
            });
    }

    fn kick(&mut self, dt: f32) {
        let velocities = self.ctx.velocities_mut();
        let accelerations = self.ctx.accelerations();

        // Parallel kick with Rayon
        use rayon::prelude::*;
        velocities.par_iter_mut()
            .zip(accelerations.par_iter())
            .for_each(|(vel, acc)| {
                vel[0] += acc[0] * dt;
                vel[1] += acc[1] * dt;
                vel[2] += acc[2] * dt;
            });
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.ctx.kinetic_energy()
    }
}

pub struct StepStats {
    pub total_ms: f64,
    pub tree_ms: f64,
    pub force_ms: f64,
    pub drift_ms: f64,
    pub kick_ms: f64,
}
```

---

## 8. Build Script (Compile Metal Shaders)

```rust
// build.rs

use std::process::Command;
use std::path::Path;

fn main() {
    // Only compile shaders on macOS
    #[cfg(target_os = "macos")]
    {
        let shader_dir = Path::new("src/shaders");
        let output_dir = Path::new("shaders");

        std::fs::create_dir_all(output_dir).unwrap();

        // Compile .metal to .air
        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metal",
                "-c",
                shader_dir.join("force.metal").to_str().unwrap(),
                "-o",
                output_dir.join("force.air").to_str().unwrap(),
            ])
            .status()
            .expect("Failed to compile Metal shader");

        assert!(status.success(), "Metal shader compilation failed");

        // Link .air to .metallib
        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metallib",
                output_dir.join("force.air").to_str().unwrap(),
                "-o",
                output_dir.join("janus.metallib").to_str().unwrap(),
            ])
            .status()
            .expect("Failed to link Metal library");

        assert!(status.success(), "Metal library linking failed");

        println!("cargo:rerun-if-changed=src/shaders/force.metal");
    }
}
```

---

## 9. Main Binary

```rust
// src/main.rs

use janus_metal::simulation::Simulation;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_particles: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let eta = 1.045f32;
    let theta = 0.5f32;
    let dt = 0.01f32;
    let max_steps = 12000;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   Janus N-Body Simulation (Metal)                          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Initializing {} particles...", n_particles);
    let mut sim = Simulation::new(n_particles, eta, theta, dt)?;
    sim.initialize_cold(42);

    let mut ts_file = BufWriter::new(File::create("time_series.csv")?);
    writeln!(ts_file, "step,ke,step_time_ms,tree_ms,force_ms")?;

    let ke_init = sim.kinetic_energy();
    println!("Initial KE: {:.4e}", ke_init);
    println!("\nStarting simulation...\n");

    for step in 1..=max_steps {
        let stats = sim.step();
        let ke = sim.kinetic_energy();

        writeln!(ts_file, "{},{:.6e},{:.1},{:.1},{:.1}",
            step, ke, stats.total_ms, stats.tree_ms, stats.force_ms)?;

        if step % 100 == 0 || step <= 10 {
            println!("step {:05} | KE={:.4e} | total={:.0}ms | tree={:.0}ms | force={:.0}ms",
                step, ke, stats.total_ms, stats.tree_ms, stats.force_ms);
        }

        if step % 100 == 0 {
            ts_file.flush()?;
        }
    }

    println!("\nSimulation complete!");
    Ok(())
}
```

---

## 10. Performance Expectations

### Theoretical Analysis

```
Force kernel: N × ~100 tree nodes × ~50 FLOPs = 5×10⁹ FLOPs for 100M
M4 GPU FP32:  ~3.5 TFLOPS
→ Theoretical: 5000/3500 = ~1.5 seconds

Tree build (CPU): O(N log N)
M4 CPU (10 cores): ~50 GFLOPS effective
→ Estimated: ~5-10 seconds for 100M

Total per step: ~10 seconds (optimistic)
             or ~20 seconds (realistic with memory bandwidth)
```

### Optimization Priorities

1. **Tree build**: Consider GPU radix sort + parallel construction
2. **Memory layout**: Ensure coalesced access in force kernel
3. **Thread divergence**: Minimize stack depth variation
4. **Occupancy**: Tune threadgroup size for M4 GPU

---

## 11. Validation Checklist

- [ ] Energy conservation: dE/E < 1% over 1000 steps
- [ ] Compare with CUDA results for 1M particles
- [ ] Verify Janus force law: +/+ attract, +/- repel (×η)
- [ ] Benchmark: tree build, force, total times
- [ ] Memory usage stays < 40GB for 100M particles

---

## 12. Quick Start

```bash
# On Mac M4:
cd janus-metal
cargo build --release

# Test with 1M particles
./target/release/janus-metal 1000000

# Production run (100M)
./target/release/janus-metal 100000000 > run.log 2>&1 &
```

---

## Appendix: Memory Budget (100M Particles)

```
Positions (f32):    100M × 12B = 1.2 GB
Velocities (f32):   100M × 12B = 1.2 GB
Accelerations (f32): 100M × 12B = 1.2 GB
Signs (i8):         100M × 1B  = 0.1 GB
Tree nodes:         200M × 64B = 12.8 GB
Work buffers:                  = 1.0 GB
──────────────────────────────────────────
TOTAL:                         = 17.5 GB  ✅ Fits in 48GB
```
