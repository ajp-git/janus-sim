# Scaling Janus to 85-100M Particles

**Objective**: Run N-body simulation with 85-100 million particles
**Current baseline**: 8M particles, 7.7s/step, ~10.8GB VRAM on RTX 3060
**Target hardware**: RTX 3060 (12GB) and/or Mac M4 (48GB unified memory)

---

## 1. Memory Analysis

### Current Memory Usage (8M particles)

```
Component                    Formula              8M Usage
─────────────────────────────────────────────────────────────
Positions (f64)              N × 3 × 8           192 MB
Velocities (f64)             N × 3 × 8           192 MB
Accelerations (f64)          N × 3 × 8           192 MB
Signs (i32)                  N × 4                32 MB
Morton codes (u64)           N × 8                64 MB
Sorted indices (u32)         N × 4                32 MB
BVH internal nodes           N × 48              384 MB
Radix sort buffers           N × 16              128 MB
CUDA context overhead        fixed               ~500 MB
─────────────────────────────────────────────────────────────
TOTAL (theoretical)                              ~1.7 GB
ACTUAL (observed)                                ~10.8 GB
```

**Gap analysis**: 6× overhead from:
- cudarc buffer management (double buffering)
- Tree rebuild allocations
- Force computation workspace
- Kernel launch overhead

### Projected Memory for 100M

```
Linear scaling:     10.8 GB × (100/8) = 135 GB  ❌ Impossible on 12GB
With optimization:  ~50-60 GB                    ❌ Still too large for RTX
Mac M4 48GB:        Feasible with streaming      ✅ Possible
```

---

## 2. Hardware Comparison

| Aspect | RTX 3060 12GB | Mac M4 48GB |
|--------|---------------|-------------|
| Memory | 12 GB VRAM (dedicated) | 48 GB unified |
| Bandwidth | 360 GB/s | 273 GB/s |
| FP64 | 0.2 TFLOPS | ~0.8 TFLOPS (estimate) |
| FP32 | 12.7 TFLOPS | ~3.5 TFLOPS |
| CPU cores | Separate (Ryzen) | 10-core integrated |
| Data transfer | PCIe bottleneck | Zero-copy |
| API | CUDA | Metal |

**Key insight**: Mac M4's unified memory eliminates the GPU↔CPU transfer bottleneck. The entire 100M particle dataset can live in shared memory, with both CPU and GPU accessing it directly.

---

## 3. Strategy A: RTX 3060 (12GB) — Streaming Approach

### Concept
Keep particles in CPU RAM (32GB), stream batches to GPU for force computation.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CPU RAM    │────▶│  GPU VRAM   │────▶│  Forces     │
│  100M parts │     │  10M batch  │     │  computed   │
│  (14 GB)    │◀────│  (1.4 GB)   │◀────│  streamed   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Implementation

```rust
// Streaming force computation for 100M particles
pub struct StreamingSimulation {
    // CPU storage (full dataset)
    positions_cpu: Vec<f64>,      // 100M × 3
    velocities_cpu: Vec<f64>,     // 100M × 3
    accelerations_cpu: Vec<f64>, // 100M × 3
    signs_cpu: Vec<i8>,           // 100M (use i8, not i32!)

    // GPU buffers (batch size)
    batch_size: usize,            // e.g., 8M
    positions_gpu: CudaSlice<f64>,
    tree_gpu: BVHTree,            // Built from ALL particles
    forces_gpu: CudaSlice<f64>,
}

impl StreamingSimulation {
    pub fn compute_forces(&mut self) -> Result<(), Error> {
        let n_batches = (self.n_particles + self.batch_size - 1) / self.batch_size;

        // Step 1: Build tree from ALL particles (streaming upload)
        self.build_tree_streaming()?;

        // Step 2: Compute forces in batches
        for batch in 0..n_batches {
            let start = batch * self.batch_size;
            let end = (start + self.batch_size).min(self.n_particles);

            // Upload batch positions
            self.positions_gpu.copy_from_host(
                &self.positions_cpu[start*3..end*3]
            )?;

            // Compute forces using full tree
            self.kernel_compute_forces(start, end)?;

            // Download forces
            self.forces_gpu.copy_to_host(
                &mut self.accelerations_cpu[start*3..end*3]
            )?;
        }

        Ok(())
    }

    fn build_tree_streaming(&mut self) -> Result<(), Error> {
        // Option 1: Build on CPU, upload structure
        // Option 2: Stream particles, build incrementally
        // Option 3: Use spatial hashing (O(1) per particle)

        // CPU tree build with Rayon (parallel)
        let tree = self.build_cpu_octree_parallel();
        self.upload_tree_to_gpu(tree)?;
        Ok(())
    }
}
```

### Memory Budget (RTX 3060)

```
Component                    Size
─────────────────────────────────────
Tree structure (100M)        4.0 GB   // Compressed BVH
Batch positions (8M)         0.2 GB
Batch forces (8M)            0.2 GB
Work buffers                 0.5 GB
CUDA overhead                0.5 GB
─────────────────────────────────────
TOTAL                        5.4 GB   ✅ Fits in 12GB
```

### Performance Estimate

```
Tree build (CPU, parallel):     ~30s for 100M
Force computation (10 batches): 10 × 8s = 80s
Data transfer overhead:         ~20s
─────────────────────────────────────────────────
Total per step:                 ~130s (vs 7.7s for 8M native)
Slowdown factor:                ~17×
Full run (12000 steps):         ~18 days
```

**Verdict**: Feasible but very slow. Better suited for short test runs.

---

## 4. Strategy B: Mac M4 (48GB) — Unified Memory

### Concept
Use Metal or CPU-parallel approach with unified memory. No data transfers needed.

### Option B1: Pure CPU (Rayon + SIMD)

```rust
// CPU Barnes-Hut with Rayon parallelism
use rayon::prelude::*;

pub struct CpuSimulation {
    positions: Vec<[f64; 3]>,     // 100M × 24B = 2.4 GB
    velocities: Vec<[f64; 3]>,    // 100M × 24B = 2.4 GB
    accelerations: Vec<[f64; 3]>, // 100M × 24B = 2.4 GB
    signs: Vec<i8>,               // 100M × 1B = 0.1 GB
    tree: Octree,                 // ~3 GB compressed
}

impl CpuSimulation {
    pub fn compute_forces_parallel(&mut self) {
        // Build octree (single-threaded for simplicity, could parallelize)
        self.tree.build(&self.positions, &self.signs);

        // Parallel force computation with Rayon
        self.accelerations
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc = self.tree.compute_force(
                    self.positions[i],
                    self.signs[i],
                    THETA,
                    SOFTENING,
                );
            });
    }
}

// Optimized octree node (cache-friendly)
#[repr(C)]
pub struct OctreeNode {
    center_of_mass: [f32; 3],    // 12 bytes (f32 for tree walk)
    total_mass_pos: f32,         // 4 bytes
    total_mass_neg: f32,         // 4 bytes
    size: f32,                   // 4 bytes
    children: [u32; 8],          // 32 bytes (indices, 0 = empty)
    // Total: 56 bytes per node
}
```

### Memory Budget (Mac M4 48GB)

```
Component                    Size
─────────────────────────────────────
Positions (100M)             2.4 GB
Velocities (100M)            2.4 GB
Accelerations (100M)         2.4 GB
Signs (100M, i8)             0.1 GB
Octree nodes (~100M)         5.6 GB
Work buffers                 2.0 GB
OS + headroom                5.0 GB
─────────────────────────────────────
TOTAL                        19.9 GB  ✅ Fits easily
```

### Performance Estimate (CPU-only)

```
M4 has ~10 high-performance cores
Barnes-Hut is O(N log N)
Each particle: ~100 tree nodes visited × ~50 FLOPs = 5000 FLOPs

100M particles × 5000 FLOPs = 500 GFLOP per step
M4 CPU FP64: ~100 GFLOPS (estimate)
Time per step: ~5 seconds (optimistic)

Reality with memory bandwidth: ~20-30 seconds per step
```

### Option B2: Metal Compute Shaders

```metal
// Metal kernel for Barnes-Hut force computation
kernel void compute_forces(
    device const float3* positions [[buffer(0)]],
    device const int8_t* signs [[buffer(1)]],
    device const TreeNode* tree [[buffer(2)]],
    device float3* accelerations [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float3 pos = positions[id];
    int8_t sign = signs[id];
    float3 acc = float3(0.0);

    // Stack-based tree traversal
    uint stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // root

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];
        TreeNode node = tree[node_idx];

        float3 d = node.center_of_mass - pos;
        float r2 = dot(d, d) + params.softening2;
        float r = sqrt(r2);

        // Opening criterion
        if (node.size / r < params.theta || node.is_leaf) {
            // Compute force from this node
            float mass_eff = sign > 0 ?
                (node.mass_pos - params.eta * node.mass_neg) :
                (params.eta * node.mass_neg - node.mass_pos);
            acc += mass_eff * d / (r2 * r);
        } else {
            // Open node, push children
            for (int c = 0; c < 8; c++) {
                if (node.children[c] != 0) {
                    stack[stack_ptr++] = node.children[c];
                }
            }
        }
    }

    accelerations[id] = acc * params.G;
}
```

### Rust + Metal Integration

```rust
// Using metal-rs crate
use metal::*;

pub struct MetalSimulation {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,

    // Shared buffers (CPU + GPU access, zero-copy!)
    positions: Buffer,
    velocities: Buffer,
    accelerations: Buffer,
    signs: Buffer,
    tree: Buffer,
}

impl MetalSimulation {
    pub fn new(n_particles: usize) -> Self {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();

        // Compile shader
        let library = device.new_library_with_source(METAL_SHADER, &CompileOptions::new()).unwrap();
        let kernel = library.get_function("compute_forces", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel).unwrap();

        // Allocate shared buffers (StorageModeShared = unified memory)
        let pos_buf = device.new_buffer(
            (n_particles * 3 * 8) as u64,
            MTLResourceOptions::StorageModeShared
        );

        // ... similar for other buffers

        Self { device, command_queue: queue, pipeline, positions: pos_buf, ... }
    }

    pub fn compute_forces(&mut self) {
        let cmd_buffer = self.command_queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.positions), 0);
        encoder.set_buffer(1, Some(&self.signs), 0);
        encoder.set_buffer(2, Some(&self.tree), 0);
        encoder.set_buffer(3, Some(&self.accelerations), 0);

        let threads_per_group = 256;
        let n_groups = (self.n_particles + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(n_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1)
        );

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
}
```

### Performance Estimate (Metal)

```
M4 GPU: ~3.5 TFLOPS FP32, ~0.8 TFLOPS FP64 (estimate)
If using FP32 for tree walk: ~3.5 TFLOPS effective

100M × 5000 FLOPs = 500 GFLOP per step
Time: 500 / 3500 = ~0.15s (theoretical)

With memory bandwidth limits: ~2-5 seconds per step
Full run (12000 steps): ~7-17 hours
```

**Verdict**: Mac M4 with Metal is the most promising approach.

---

## 5. Strategy C: Hybrid Precision

### Concept
Use FP32 for tree traversal (speed), FP64 for integration (accuracy).

```rust
pub struct HybridPrecisionSimulation {
    // Integration state (f64 for accuracy)
    positions_f64: Vec<[f64; 3]>,
    velocities_f64: Vec<[f64; 3]>,

    // Tree walk (f32 for speed and memory)
    positions_f32: Vec<[f32; 3]>,  // Converted each step
    tree_f32: OctreeF32,
    accelerations_f32: Vec<[f32; 3]>,
}

impl HybridPrecisionSimulation {
    pub fn step(&mut self, dt: f64) {
        // Convert positions to f32 for tree build
        self.positions_f32.par_iter_mut()
            .zip(self.positions_f64.par_iter())
            .for_each(|(f32_pos, f64_pos)| {
                f32_pos[0] = f64_pos[0] as f32;
                f32_pos[1] = f64_pos[1] as f32;
                f32_pos[2] = f64_pos[2] as f32;
            });

        // Build tree and compute forces in f32
        self.tree_f32.build(&self.positions_f32);
        self.compute_forces_f32();

        // Integrate in f64
        self.positions_f64.par_iter_mut()
            .zip(self.velocities_f64.par_iter_mut())
            .zip(self.accelerations_f32.par_iter())
            .for_each(|((pos, vel), acc)| {
                // Leapfrog kick
                vel[0] += acc[0] as f64 * dt;
                vel[1] += acc[1] as f64 * dt;
                vel[2] += acc[2] as f64 * dt;
                // Leapfrog drift
                pos[0] += vel[0] * dt;
                pos[1] += vel[1] * dt;
                pos[2] += vel[2] * dt;
            });
    }
}
```

### Memory Savings

```
                        f64 only    Hybrid (f32 tree)
─────────────────────────────────────────────────────
Positions               2.4 GB      2.4 + 1.2 = 3.6 GB
Velocities              2.4 GB      2.4 GB
Accelerations           2.4 GB      1.2 GB (f32)
Tree nodes              5.6 GB      2.8 GB
─────────────────────────────────────────────────────
TOTAL                   12.8 GB     10.0 GB
```

Not huge savings, but 2× faster tree traversal due to cache efficiency.

---

## 6. Strategy D: Algorithmic Optimization

### Dual-Tree Traversal

Instead of computing forces particle-by-particle, use node-node interactions.

```rust
// Dual-tree traversal: O(N) instead of O(N log N) in favorable cases
fn compute_forces_dual_tree(
    source_node: &TreeNode,
    target_node: &TreeNode,
    forces: &mut [Vec3],
    theta: f64,
) {
    let d = target_node.center - source_node.center;
    let r = d.norm();

    // MAC: can we approximate entire source → entire target?
    let mac = (source_node.size + target_node.size) / r;

    if mac < theta {
        // Node-node interaction
        let mass_eff = compute_effective_mass(source_node, target_node);
        let force = mass_eff * d / r.powi(3);

        // Apply to all particles in target node
        for &idx in &target_node.particles {
            forces[idx] += force;
        }
    } else if source_node.is_leaf() && target_node.is_leaf() {
        // Direct particle-particle
        for &i in &target_node.particles {
            for &j in &source_node.particles {
                if i != j {
                    forces[i] += particle_force(i, j);
                }
            }
        }
    } else {
        // Open larger node
        if source_node.size > target_node.size {
            for child in source_node.children() {
                compute_forces_dual_tree(child, target_node, forces, theta);
            }
        } else {
            for child in target_node.children() {
                compute_forces_dual_tree(source_node, child, forces, theta);
            }
        }
    }
}
```

### Fast Multipole Method (FMM)

True O(N) complexity, but complex to implement.

```
BH:  O(N log N) — ~17 seconds for 100M
FMM: O(N)       — ~1-2 seconds for 100M (theoretical)
```

FMM would require:
1. Multipole expansion coefficients (p=4 gives ~1% error)
2. Local expansion coefficients
3. M2M, M2L, L2L translation operators

### Spatial Hashing for Neighbors

For close interactions, use O(1) neighbor lookup:

```rust
use std::collections::HashMap;

pub struct SpatialHash {
    cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    pub fn build(&mut self, positions: &[[f64; 3]]) {
        self.cells.clear();
        for (i, pos) in positions.iter().enumerate() {
            let key = (
                (pos[0] / self.cell_size).floor() as i32,
                (pos[1] / self.cell_size).floor() as i32,
                (pos[2] / self.cell_size).floor() as i32,
            );
            self.cells.entry(key).or_default().push(i);
        }
    }

    pub fn neighbors(&self, pos: [f64; 3]) -> impl Iterator<Item = usize> + '_ {
        let key = (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
            (pos[2] / self.cell_size).floor() as i32,
        );

        // Check 27 neighboring cells
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1).flat_map(move |dz| {
                    self.cells
                        .get(&(key.0 + dx, key.1 + dy, key.2 + dz))
                        .into_iter()
                        .flatten()
                        .copied()
                })
            })
        })
    }
}
```

---

## 7. Recommended Approach

### For Mac M4 48GB (PRIMARY)

```
Phase 1: Pure Rust CPU implementation
├── Use existing nbody.rs as base
├── Optimize with Rayon parallelism
├── SoA layout for cache efficiency
├── Target: ~20s/step for 100M
└── Full run: ~67 hours (3 days)

Phase 2: Metal acceleration
├── Port tree traversal to Metal compute shader
├── Keep integration on CPU (f64)
├── Target: ~5s/step for 100M
└── Full run: ~17 hours
```

### For RTX 3060 (SECONDARY)

```
Use streaming approach for validation runs:
├── Short runs (1000 steps) for testing
├── Compare with Mac results
├── Target: ~130s/step for 100M
└── 1000 steps: ~36 hours
```

---

## 8. Implementation Roadmap

### Week 1: CPU Foundation

```bash
# Create cross-platform CPU implementation
cargo new janus-cpu --lib

# Structure:
src/
├── lib.rs
├── particle.rs      # SoA layout
├── octree.rs        # Cache-optimized tree
├── forces.rs        # Parallel force computation
└── integrator.rs    # Leapfrog with cosmology
```

### Week 2: Mac M4 Port

```bash
# Add Metal support
[dependencies]
metal = "0.27"
objc = "0.2"

# Conditional compilation
#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(feature = "cuda")]
mod cuda_backend;
```

### Week 3: Validation

```
1. Compare 1M particles: CUDA vs CPU vs Metal
2. Verify energy conservation
3. Verify segregation dynamics match
4. Profile and optimize bottlenecks
```

### Week 4: Production Run

```
Run 100M on Mac M4:
├── z = 5 → 0
├── dt = 0.01
├── θ = 0.5
├── Estimated time: 17-67 hours depending on optimization
└── Output: time_series.csv + snapshots every 100 steps
```

---

## 9. Quick Start Code

### Minimal CPU Barnes-Hut (copy-paste ready)

```rust
//! Minimal Barnes-Hut for 100M particles
//! Compile: cargo build --release
//! Run: cargo run --release -- --n 100000000

use rayon::prelude::*;
use std::time::Instant;

const THETA: f64 = 0.5;
const SOFTENING: f64 = 0.1;
const G: f64 = 1.0;

#[derive(Clone, Copy)]
struct Vec3 { x: f64, y: f64, z: f64 }

impl Vec3 {
    fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    fn norm2(&self) -> f64 { self.x*self.x + self.y*self.y + self.z*self.z }
    fn norm(&self) -> f64 { self.norm2().sqrt() }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z;
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

struct OctreeNode {
    center: Vec3,
    size: f64,
    mass_pos: f64,
    mass_neg: f64,
    com: Vec3,
    children: [Option<Box<OctreeNode>>; 8],
    particle_idx: Option<usize>,
}

impl OctreeNode {
    fn new(center: Vec3, size: f64) -> Self {
        Self {
            center, size,
            mass_pos: 0.0, mass_neg: 0.0,
            com: Vec3::zero(),
            children: Default::default(),
            particle_idx: None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }

    fn child_index(&self, pos: Vec3) -> usize {
        let mut idx = 0;
        if pos.x > self.center.x { idx |= 1; }
        if pos.y > self.center.y { idx |= 2; }
        if pos.z > self.center.z { idx |= 4; }
        idx
    }

    fn child_center(&self, idx: usize) -> Vec3 {
        let q = self.size / 4.0;
        Vec3 {
            x: self.center.x + if idx & 1 != 0 { q } else { -q },
            y: self.center.y + if idx & 2 != 0 { q } else { -q },
            z: self.center.z + if idx & 4 != 0 { q } else { -q },
        }
    }

    fn insert(&mut self, idx: usize, pos: Vec3, sign: i8, positions: &[Vec3]) {
        // Update mass and COM
        let mass = sign as f64;
        if sign > 0 { self.mass_pos += 1.0; } else { self.mass_neg += 1.0; }
        let total_mass = self.mass_pos + self.mass_neg;
        self.com = self.com * ((total_mass - 1.0) / total_mass) + pos * (1.0 / total_mass);

        if self.is_leaf() {
            if let Some(existing_idx) = self.particle_idx {
                // Split: move existing particle to child
                self.particle_idx = None;
                let existing_pos = positions[existing_idx];
                let existing_sign = if existing_idx < positions.len() / 2 { 1i8 } else { -1i8 };
                let child_idx = self.child_index(existing_pos);
                let child_center = self.child_center(child_idx);
                self.children[child_idx] = Some(Box::new(OctreeNode::new(child_center, self.size / 2.0)));
                self.children[child_idx].as_mut().unwrap().insert(existing_idx, existing_pos, existing_sign, positions);

                // Insert new particle
                let child_idx = self.child_index(pos);
                if self.children[child_idx].is_none() {
                    let child_center = self.child_center(child_idx);
                    self.children[child_idx] = Some(Box::new(OctreeNode::new(child_center, self.size / 2.0)));
                }
                self.children[child_idx].as_mut().unwrap().insert(idx, pos, sign, positions);
            } else {
                self.particle_idx = Some(idx);
            }
        } else {
            let child_idx = self.child_index(pos);
            if self.children[child_idx].is_none() {
                let child_center = self.child_center(child_idx);
                self.children[child_idx] = Some(Box::new(OctreeNode::new(child_center, self.size / 2.0)));
            }
            self.children[child_idx].as_mut().unwrap().insert(idx, pos, sign, positions);
        }
    }

    fn compute_force(&self, pos: Vec3, sign: i8, eta: f64) -> Vec3 {
        let d = self.com - pos;
        let r2 = d.norm2() + SOFTENING * SOFTENING;
        let r = r2.sqrt();

        // Opening criterion
        if self.size / r < THETA || self.is_leaf() {
            if self.mass_pos + self.mass_neg < 0.5 { return Vec3::zero(); }

            // Janus effective mass
            let mass_eff = if sign > 0 {
                self.mass_pos - eta * self.mass_neg
            } else {
                eta * self.mass_neg - self.mass_pos
            };

            d * (G * mass_eff / (r2 * r))
        } else {
            let mut acc = Vec3::zero();
            for child in &self.children {
                if let Some(c) = child {
                    acc += c.compute_force(pos, sign, eta);
                }
            }
            acc
        }
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let eta = 1.045;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    let n_pos = (n as f64 / (1.0 + eta)) as usize;

    println!("Initializing {} particles...", n);

    // Initialize positions (uniform random)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let positions: Vec<Vec3> = (0..n)
        .map(|_| Vec3 {
            x: (rng.gen::<f64>() - 0.5) * box_size,
            y: (rng.gen::<f64>() - 0.5) * box_size,
            z: (rng.gen::<f64>() - 0.5) * box_size,
        })
        .collect();
    let signs: Vec<i8> = (0..n).map(|i| if i < n_pos { 1 } else { -1 }).collect();

    println!("Building octree...");
    let t0 = Instant::now();

    let mut root = OctreeNode::new(Vec3::zero(), box_size);
    for (i, &pos) in positions.iter().enumerate() {
        root.insert(i, pos, signs[i], &positions);
    }

    println!("  Tree built in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Computing forces (parallel)...");
    let t0 = Instant::now();

    let accelerations: Vec<Vec3> = positions
        .par_iter()
        .zip(signs.par_iter())
        .map(|(&pos, &sign)| root.compute_force(pos, sign, eta))
        .collect();

    let elapsed = t0.elapsed().as_secs_f64();
    println!("  Forces computed in {:.2}s", elapsed);
    println!("  Rate: {:.0} particles/second", n as f64 / elapsed);

    // Sanity check
    let total_acc: f64 = accelerations.iter().map(|a| a.norm()).sum();
    println!("  Total |a|: {:.4e}", total_acc);
}
```

---

## 10. Conclusion

| Approach | Hardware | Time/step | Full run | Complexity |
|----------|----------|-----------|----------|------------|
| Streaming BH | RTX 3060 | ~130s | 18 days | Medium |
| CPU Rayon | Mac M4 | ~20s | 67 hours | Low |
| Metal BH | Mac M4 | ~5s | 17 hours | High |
| Metal FMM | Mac M4 | ~2s | 7 hours | Very High |

**Recommendation**: Start with CPU Rayon on Mac M4, then optimize with Metal if needed.

The Mac M4's unified memory is a game-changer for large N-body simulations — no PCIe bottleneck, no complex memory management, just allocate and compute.

---

## Appendix: Useful Commands

```bash
# Check Mac M4 memory
system_profiler SPMemoryDataType

# Monitor Metal GPU usage
sudo powermetrics --samplers gpu_power

# Cross-compile from Linux to Mac (if needed)
rustup target add aarch64-apple-darwin

# Profile on Mac
instruments -t "Time Profiler" ./target/release/janus-cpu
```
