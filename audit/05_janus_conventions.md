# Audit 05 — Conventions Janus existant (Barnes-Hut)

**Date** : 2026-04-29

## Encodage signe m+/m-

Snapshot V3 (src/snapshot_v3.rs) — sign en u1:
265:    pub sign: u8,            // +1 = m+, 255 = m-
302:        let sign: u8 = { self.sign };
308:        let sign: u8 = { self.sign };
319:        let sign: u8 = { self.sign };

GpuNBodySimulation runtime — sign en i32:
1663:    signs: CudaSlice<i32>,

Conversion u1 → i32 (snapshot read):
src/snapshot_v3.rs:706:        assert_eq!(p.sign, 1);

## Convention force Janus dans compute_forces_bvh

                double interaction = (my_sign > 0) ? 1.0 : -cross_minus_plus;
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            // Interaction with m- mass in this node
            if (mass_minus > 0.0) {
                double com_minus_x = node_data[base + 8];
                double com_minus_y = node_data[base + 9];
                double com_minus_z = node_data[base + 10];
                double dmx = minimum_image(com_minus_x - px, box_size, box_half);
                double dmy = minimum_image(com_minus_y - py, box_size, box_half);
                double dmz = minimum_image(com_minus_z - pz, box_size, box_half);
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));
                // Cross-species: m+ feels repulsion from m- (cross_plus_minus factor)
                double interaction = (my_sign < 0) ? 1.0 : -cross_plus_minus;
                double f = interaction * mass_minus * inv_rm3;
                ax += f * dmx;
                ay += f * dmy;
                az += f * dmz;
            }
        } else {
            // Descend into children
            int left = left_child[node_idx];
            int right = right_child[node_idx];
            if (left >= 0 && stack_ptr < 63) stack[stack_ptr++] = left;
            if (right >= 0 && stack_ptr < 63) stack[stack_ptr++] = right;
        }
    }

    acc[tid * 3] = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

// Compute forces with configurable cross-sign interaction
// cross_factor: 0.0 = no cross interaction (attraction only)
//              -1.0 = Janus (repulsion between opposite signs)
extern "C" __global__ void compute_forces_bvh_cross(
    const double* __restrict__ pos,
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ node_types,
    double* __restrict__ acc,
    int n_particles,
    double theta,
    double softening,
    double cross_factor,
    double box_size  // Box size for periodic boundary conditions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double box_half = 0.5 * box_size;
    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    // Asymmetric softening: m- uses larger value
    double eps = (my_sign > 0) ? softening : (softening * SOFTENING_MINUS_RATIO);
    double eps2 = eps * eps;

    int stack[64];
    int stack_ptr = 0;

## Cross-coupling factors (computed in step_with_expansion_dkd_gpu_cosmo)
        };

        let phi_inv = 1.0 / self.phi;
        let cross_minus_plus = self.c_ratio_sq * phi_inv * self.repulsion_scale;
        let cross_plus_minus = self.phi * self.repulsion_scale;

        let drift_kernel = self.device.get_func("nbody", "drift_only_cosmo")
            .ok_or("Failed to get drift_only_cosmo kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only_cosmo")
            .ok_or("Failed to get kick_only_cosmo kernel")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh")
            .ok_or("Failed to get compute_forces_bvh kernel")?;

        // D1: Drift(dt/2) at a_plus(t), a_minus(t)
        unsafe {
            drift_kernel.clone().launch(cfg, (

## Drift convention
extern "C" __global__ void drift_only_cosmo(
    double* __restrict__ pos,
    const double* __restrict__ vel,
    const int* __restrict__ signs,
    double dt,
    double box_half,
    double a_plus,
    double a_minus,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int base = tid * 3;
    double a_eff = (signs[tid] > 0) ? a_plus : a_minus;
    double inv_a = 1.0 / a_eff;
    pos[base]     += vel[base]     * dt * inv_a;
    pos[base + 1] += vel[base + 1] * dt * inv_a;
    pos[base + 2] += vel[base + 2] * dt * inv_a;
    // Robust periodic BC for arbitrary-size displacements
    double box = 2.0 * box_half;
    for (int i = 0; i < 3; i++) {
        double p = pos[base + i] + box_half;     // shift to [0, box]
        p = p - box * floor(p / box);            // wrap
        pos[base + i] = p - box_half;            // shift back to [-box_half, box_half]
    }
}

## Kick convention
// COSMOLOGICAL KICK (peculiar convention)
//   acc[]  : bare comoving accel = G*m / r_co^2   (output of compute_forces_bvh, no a-factor)
//   dv_pec/dt = -H * v_pec  -  G*m / (a^2 * r_co^2) =  acc / a^2  -  H * v_pec
// Per-particle a and H selected by sign.
extern "C" __global__ void kick_only_cosmo(
    double* __restrict__ vel,
    const double* __restrict__ acc,
    const int* __restrict__ signs,
    double dt,
    int n,
    double a_plus,
    double a_minus,
    double h_plus,
    double h_minus
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int base = tid * 3;
    double a_eff = (signs[tid] > 0) ? a_plus  : a_minus;
    double h_eff = (signs[tid] > 0) ? h_plus  : h_minus;
    double inv_a2 = 1.0 / (a_eff * a_eff);
    for (int d = 0; d < 3; d++) {
        double grav = acc[base + d] * inv_a2;
        double friction = -h_eff * vel[base + d];
        vel[base + d] += (grav + friction) * dt;
    }
}
