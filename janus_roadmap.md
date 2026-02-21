# Roadmap — Janus Project
# Document for Claude CLI
# Date: February 20, 2026
# Based on analysis from ChatGPT, Grok and Gemini

---

## CONTEXT AND CURRENT STATE

### What is validated
- Pantheon+ fit: eta=1.045, q0=-0.022, chi^2/dof=0.914 ✓
- Barnes-Hut GPU f64: 0% CPU/GPU deviation ✓
- Bugs fixed: rsqrt(), periodic COM, acceleration equations ✓
- Run A (eta=1.045) and Run B (eta=2.03) in progress on 10,000 steps ✓

### Central problem identified
Run A shows segregation that rises (+7.8% at step 500)
then falls back (-19.8% at step 2000).

Three probable causes identified by Gemini/ChatGPT/Grok:
1. Non-virialized initial conditions (Seg0 ~ 0.49 artificial)
2. Missing Hubble friction (static universe instead of expanding)
3. Quasi-symmetric regime eta~1 (positive/negative forces nearly equal)

---

## TASK 1 — VIRIALIZED INITIAL CONDITIONS
### Priority: CRITICAL — Do before any new simulation

### Why it's priority
Currently Seg0 ~ 0.49: the system starts from an already partially
segregated state and relaxes toward a mixed state. We're not testing natural
emergence of segregation, we're testing relaxation of a biased state.
Objective: Seg0 ~ 0.03, 2KE + PE = 0.

### Gemini Code — PE calculation with Janus logic

```rust
// Add in the Node or Tree struct of nbody_gpu.rs (CPU version)
// Calculate PE once at t=0 on CPU before launching GPU

impl Node {
    pub fn get_potential(&self, pos: [f64; 3], mass_sign: f64, theta: f64) -> f64 {
        let r_vec = [
            self.center_of_mass[0] - pos[0],
            self.center_of_mass[1] - pos[1],
            self.center_of_mass[2] - pos[2],
        ];
        let r = (r_vec[0].powi(2) + r_vec[1].powi(2) + r_vec[2].powi(2)).sqrt();

        if r < 1e-10 {
            return 0.0; // Avoid division by zero
        }

        if self.is_leaf() || (self.width / r < theta) {
            // JANUS LOGIC:
            // Same sign → Attraction (Negative Potential)
            // Opposite signs → Repulsion (Positive Potential)
            let interaction_sign = if self.mass * mass_sign > 0.0 { -1.0 } else { 1.0 };
            let softened_r = (r * r + EPSILON * EPSILON).sqrt();
            return interaction_sign * (G * self.mass.abs() / softened_r);
        }

        // Recursive call for internal nodes
        self.children
            .iter()
            .filter_map(|child| child.as_ref())
            .map(|child| child.get_potential(pos, mass_sign, theta))
            .sum()
    }
}

// Global PE calculation
pub fn compute_total_potential_energy(
    positions: &[[f64; 3]],
    masses: &[f64],
    signs: &[f64],
    tree: &OctreeNode,
    theta: f64,
) -> f64 {
    let pe: f64 = positions
        .iter()
        .zip(masses.iter())
        .zip(signs.iter())
        .map(|((pos, &mass), &sign)| {
            mass * tree.get_potential(*pos, sign, theta)
        })
        .sum();
    pe * 0.5 // Factor 1/2 to avoid double counting
}
```

### Code — Velocity virialization

```rust
pub fn virialize_velocities(
    velocities: &mut [[f64; 3]],
    kinetic_energy: f64,
    potential_energy: f64,
) {
    // Virial condition: 2KE + PE = 0
    // → KE_target = -PE/2
    // → alpha = sqrt(KE_target / KE_current) = sqrt(|PE| / (2*KE))

    let ke_target = potential_energy.abs() / 2.0;
    let alpha = (ke_target / kinetic_energy).sqrt();

    println!("Virialization:");
    println!("  KE initial    = {:.4e}", kinetic_energy);
    println!("  PE            = {:.4e}", potential_energy);
    println!("  KE target     = {:.4e}", ke_target);
    println!("  Alpha scale   = {:.6}", alpha);

    for vel in velocities.iter_mut() {
        vel[0] *= alpha;
        vel[1] *= alpha;
        vel[2] *= alpha;
    }

    // Verification
    let ke_after: f64 = velocities.iter()
        .map(|v| 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    let virial_ratio = 2.0 * ke_after + potential_energy;
    println!("  2KE + PE after = {:.4e} (should be ~ 0)", virial_ratio);

    assert!(
        virial_ratio.abs() / potential_energy.abs() < 0.01,
        "Virialization failed: ratio = {:.4}", virial_ratio
    );
}
```

### Mandatory validation test (VALIDATION_RULES.md)

```rust
#[test]
fn test_virialization() {
    // After virialization:
    // 1. Seg0 must be < 0.05
    // 2. KE/KE0 must remain < 5 over 100 steps
    // 3. Segregation must be increasing between step 0 and step 100

    let (ke, pe) = compute_energies(&sim);
    let virial = 2.0 * ke + pe;
    assert!(virial.abs() / pe.abs() < 0.01,
        "Virial not satisfied: {:.4e}", virial);
    assert!(sim.segregation() < 0.05,
        "Seg0 too high: {:.4}", sim.segregation());
}
```

### Instructions for Claude CLI

```
TASK 1: Implement virialization

1. Add compute_total_potential_energy() in nbody.rs (CPU)
   Use existing Barnes-Hut tree with Janus logic.

2. Add virialize_velocities() in the init module.

3. Modify initialize_particles() to call:
   a. compute_total_potential_energy()
   b. virialize_velocities()
   c. Verify 2KE + PE ~ 0

4. Mandatory validation test before any simulation:
   500K particles, eta=1.045, verify Seg0 < 0.05

5. Launch test 500K, eta=1.045, dt=0.01, 200 steps.
   Report Seg0, Seg100, Seg200, KE/KE0.
   Wait for instruction.

IMPORTANT: Calculate PE on CPU once at t=0.
Do not include in the GPU loop.
Read VALIDATION_RULES.md before any implementation.
```

---

## TASK 2 — HUBBLE FRICTION (COSMOLOGICAL EXPANSION)
### Priority: HIGH — After validation of Task 1

### Why it's important
Without cosmological expansion, KE/KE0 explodes (115 in Run A).
Expansion "damps" velocities (v ~ 1/a) and allows structures
to stabilize instead of bouncing after collapse.

### Physics — Equation of motion in comoving coordinates

```
Physical position: r = a(t) × x (x = comoving coordinate)
Equation of motion:
  x_ddot = g_Janus / a^3 - 2H·x_dot

where:
  g_Janus = acceleration calculated by Barnes-Hut
  H = a_dot/a = Hubble parameter (from friedmann.rs)
  -2H·x_dot = Hubble friction term
```

### Gemini Code — Modified CUDA kernel

```cuda
// Modify the leapfrog_kick kernel in nbody_gpu.rs
// Replace existing kick kernel with:

extern "C" __global__ void leapfrog_kick_kernel(
    double* v,           // Velocity vector [vx, vy, vz] × N
    const double* f,     // Force/Acceleration calculated [fx, fy, fz] × N
    double dt,           // Time step
    double a,            // Scale factor a(t) at current step
    double adot_over_a,  // H(t) = a_dot/a = Hubble parameter
    int n                // Number of particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = i * 3;

        for (int d = 0; d < 3; d++) {
            // 1. Janus acceleration in comoving coordinates
            double accel_janus = f[idx + d] / (a * a * a);

            // 2. Hubble Friction term
            double friction = -2.0 * adot_over_a * v[idx + d];

            // 3. Velocity update (Kick)
            v[idx + d] += (accel_janus + friction) * dt;
        }
    }
}
```

### Rust Integration — Pass a(t) and H(t) from friedmann.rs

```rust
// In the main loop of nbody_gpu.rs

pub struct CosmologicalParams {
    pub a: f64,        // Current scale factor
    pub h: f64,        // H(t) = a_dot/a Hubble parameter
}

impl NBodyGPU {
    pub fn step_with_expansion(
        &mut self,
        dt: f64,
        cosmo: &CosmologicalParams,
    ) -> Result<(), SimError> {
        // 1. Drift (positions)
        self.drift(dt / 2.0)?;

        // 2. Rebuild tree
        self.build_tree()?;

        // 3. Compute forces
        self.compute_forces(cosmo.a)?;

        // 4. Kick with Hubble friction
        // Pass a and H to CUDA kernel
        unsafe {
            launch_kernel!(
                self.kick_kernel,
                grid = (self.n + 255) / 256,
                block = 256,
                args = [
                    &self.vel_gpu,
                    &self.acc_gpu,
                    &dt,
                    &cosmo.a,        // ← new parameter
                    &cosmo.h,        // ← new parameter
                    &(self.n as i32),
                ]
            )?;
        }

        // 5. Drift (positions)
        self.drift(dt / 2.0)?;

        Ok(())
    }
}

// In main or simulation loop:
// Retrieve a(t) and H(t) from friedmann.rs at each step

fn get_cosmological_params(t: f64, eta: f64) -> CosmologicalParams {
    let history = integrate_friedmann(eta, t);
    let a = interpolate_a(&history, t);
    let adot = interpolate_adot(&history, t);
    CosmologicalParams {
        a,
        h: adot / a,
    }
}
```

### CRITICAL CORRECTION — Ghost functions (Gemini)

```
WARNING: The following functions mentioned in the roadmap
DO NOT EXIST in current friedmann.rs code:

  integrate_friedmann(eta, t)   ← DOES NOT EXIST
  interpolate_a(&history, t)    ← DOES NOT EXIST

Current code only does backward integration
(integrate_backward) from z=0.

Solution: Implement CosmoInterpolator below.
```

### Gemini Code — CosmoInterpolator (to add in friedmann.rs)

```rust
/// Cosmological interpolator for N-body simulation
/// Calculates universe history ONCE,
/// then provides a(t) and H(t) by fast interpolation.
pub struct CosmoInterpolator {
    history: Vec<JanusState>,
    pub tau_start: f64,  // tau at initial redshift (past)
    pub tau_end: f64,    // tau = 0 (today)
}

impl CosmoInterpolator {
    /// Prepares universe history from z_init to z=0
    /// z_init = 50.0 for standard cosmological simulations
    pub fn new(params: &JanusParams, z_init: f64) -> Self {
        // 1. Integrate backward (z=0 → z_init)
        let mut history = integrate_backward(params, z_init, 10000);

        // 2. Sort by INCREASING tau (from past to present)
        history.sort_by(|s1, s2| s1.tau.partial_cmp(&s2.tau).unwrap());

        let tau_start = history.first().unwrap().tau;
        let tau_end = history.last().unwrap().tau;

        Self { history, tau_start, tau_end }
    }

    /// Returns (a, H) for a given cosmological time tau
    /// Linear interpolation between history points
    pub fn get_params_at_tau(&self, tau_target: f64) -> (f64, f64) {
        // Edge safety
        if tau_target <= self.tau_start {
            let s = self.history.first().unwrap();
            return (s.a, s.hubble());
        }
        if tau_target >= self.tau_end {
            let s = self.history.last().unwrap();
            return (s.a, s.hubble());
        }

        // Binary search for interval
        let idx = self.history.partition_point(|s| s.tau < tau_target);
        let s0 = &self.history[idx - 1];
        let s1 = &self.history[idx];

        // Linear interpolation
        let fraction = (tau_target - s0.tau) / (s1.tau - s0.tau);
        let a_interp = s0.a + fraction * (s1.a - s0.a);
        let h_interp = s0.hubble() + fraction * (s1.hubble() - s0.hubble());

        (a_interp, h_interp)
    }
}
```

### Code — Usage in N-body loop (nbody_gpu.rs)

```rust
// BEFORE the loop: initialize ONCE
let params = JanusParams::from_eta(eta);
let cosmo = CosmoInterpolator::new(&params, 50.0); // z_init = 50
let total_nbody_steps = 10_000usize;

// Time unit conversion:
// The 10000 N-body steps cover exactly the cosmological age
// from z=50 to z=0
let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (total_nbody_steps as f64);

// IN the N-body loop
for step in 0..total_nbody_steps {
    let current_tau = cosmo.tau_start + (step as f64) * dtau_cosmo;
    let (a, h) = cosmo.get_params_at_tau(current_tau);

    // Launch CUDA kernel with (a, h)
    sim.step_with_expansion(dt, a, h)?;
}
```

### Mandatory unit test — test_cosmo_interpolator

```rust
// Add in tests module of friedmann.rs
#[test]
fn test_cosmo_interpolator() {
    // 1. Initialization with validated parameters (Pantheon+ fit)
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let z_init = 50.0;

    let cosmo = CosmoInterpolator::new(&params, z_init);

    // 2. Past verification (start of simulation at z_init)
    let (a_start, h_start) = cosmo.get_params_at_tau(cosmo.tau_start);
    let expected_a_start = 1.0 / (1.0 + z_init);

    assert!(
        (a_start - expected_a_start).abs() < 1e-4,
        "a_start should be 1/(1+z_init) = {:.6}, got: {:.6}",
        expected_a_start, a_start
    );
    assert!(
        h_start > 0.0,
        "H should be > 0 at start (expanding universe)"
    );

    // 3. Present verification (end of simulation z=0)
    let (a_end, h_end) = cosmo.get_params_at_tau(cosmo.tau_end);

    assert!(
        (a_end - 1.0).abs() < 1e-4,
        "a_end should be 1.0 (today), got: {:.6}", a_end
    );

    // Convention validated in friedmann.rs line 106:
    // a_dot_0 = sqrt(Omega_+)  →  H0 = a_dot_0/a0 = sqrt(Omega_+) (since a0=1)
    let expected_h_end = params.omega_plus.sqrt();
    assert!(
        (h_end - expected_h_end).abs() < 1e-4,
        "H_end should be sqrt(Omega_+) = {:.6}, got: {:.6}",
        expected_h_end, h_end
    );

    // 4. Monotonicity of a(t)
    let tau_mid = cosmo.tau_start + (cosmo.tau_end - cosmo.tau_start) / 2.0;
    let (a_mid, _) = cosmo.get_params_at_tau(tau_mid);

    assert!(
        a_mid > a_start && a_mid < a_end,
        "a(t) must be strictly increasing: {} < {} < {}",
        a_start, a_mid, a_end
    );

    println!("CosmoInterpolator validated:");
    println!("  a_start = {:.6} (expected {:.6})", a_start, expected_a_start);
    println!("  a_end   = {:.6} (expected 1.0)", a_end);
    println!("  H_end   = {:.6} (expected {:.6})", h_end, expected_h_end);
}
```

### Note on H0 convention — verified in friedmann.rs

```
CONFIRMED in friedmann.rs line 106:

    let a_dot = params.omega_plus.sqrt();
    // From Friedmann: (a_dot/a)^2 = Omega_+/a^3 → a_dot_0 = sqrt(Omega_+)

So H0 = a_dot_0/a0 = sqrt(Omega_+)/1 = sqrt(Omega_+)

For eta=1.045:
  Omega_+ = 1/(1+1.045) = 0.4890
  H0 = sqrt(0.4890) = 0.6993 (dimensionless units H0=1)

The test expected_h_end = params.omega_plus.sqrt() is CORRECT.
```

### Important note on unit consistency

```
MANDATORY VERIFICATION before activating this kernel:

Are distances in Barnes-Hut in physical coordinates
or comoving?

If physical → dividing by a^3 is correct (as in kernel)
If comoving → adjust conversion factor

Mandatory consistency test:
  Run 100 steps with a=1.0 constant, H=0.
  Results IDENTICAL to simulation without expansion.
  If different → bug in implementation.
```

### Instructions for Claude CLI

```
TASK 2: Implement Hubble friction
(ONLY after Task 1 validation)

STEP 2a — Add CosmoInterpolator in friedmann.rs
  Use code provided above.
  Add test_cosmo_interpolator in tests module.
  Test MUST pass before continuing.

STEP 2b — Modify leapfrog_kick_kernel
  Use Gemini code (CUDA kernel provided above).
  Accept a and adot_over_a as parameters.

STEP 2c — Modify main loop
  Initialize CosmoInterpolator ONCE before the loop.
  Calculate dtau_cosmo = (tau_end - tau_start) / total_steps.
  Pass (a, h) to kernel at each step.

STEP 2d — Consistency test
  Run 100 steps with a=1.0 constant, H=0.
  Results identical to simulation without expansion → OK.

STEP 2e — Physical validation
  Run 500K, eta=1.045, 500 steps with expansion enabled.
  Report: max KE/KE0, Seg500.
  Objective: KE/KE0 < 20 (vs 115 without expansion).

Wait for instruction before each step.
```

---

## TASK 3 — CONVERGENCE STUDY IN N
### Priority: HIGH — After validation of Tasks 1 and 2

### Why it's necessary
Without convergence in N, impossible to defend simulations
before a reviewer. This is the minimum criterion for publication.

### Protocol

```
Fixed parameters for all runs:
  eta = 1.045, dt = 0.01, f64
  Virialized initial conditions (Task 1)
  Cosmological expansion enabled (Task 2)
  Same random seed

N to test: 100K, 500K, 1M, 2M

For each N, measure:
  - Seg0, Seg100, Seg500, Seg_max
  - Step where Seg is maximum
  - Max KE/KE0
  - Cluster formation time (step where structure is visible)
```

### Convergence criterion

```
Result is converged if:
  |Seg_max(1M) - Seg_max(2M)| / Seg_max(2M) < 10%

If converged → physically robust result, defensible
If not converged → increase N or investigate
```

### Instructions for Claude CLI

```
TASK 3: Convergence study
(ONLY after validation of Tasks 1 and 2)

Run 4 sequential simulations:
  N = 100K : 500 steps
  N = 500K : 500 steps
  N = 1M   : 500 steps
  N = 2M   : 500 steps (if time available)

Same seed (42), same eta=1.045, same virialized ICs.

Produce comparison table:
  | N    | Seg0 | Seg_max | Step_max | KE/KE0_max |
  |------|------|---------|----------|------------|

Convergence criterion: Seg_max(N) and Seg_max(2N) < 10% difference.
Report results and wait for instruction.
```

---

## TASK 4 — CORRELATION FUNCTION xi(r)
### Priority: MEDIUM — After convergence in N

### What it provides
Standard measure of large-scale structure.
Qualitative comparison with SDSS DR7.
Strong argument for presentation to Petit.

### Python Code — xi(r) calculation with Corrfunc

```python
#!/usr/bin/env python3
"""
Two-point correlation function xi(r) calculation
for HDF5 snapshots from Janus Run A.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

# pip install Corrfunc --break-system-packages
import Corrfunc
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf

def compute_xi(snapshot_path: str, n_bins: int = 20) -> tuple:
    """
    Calculates xi(r) for positive masses from an HDF5 snapshot.

    Returns:
        r_centers : bin centers in simulation units
        xi        : xi(r) values
    """
    with h5py.File(snapshot_path, 'r') as f:
        pos = f['positions'][:]      # shape (N, 3)
        signs = f['signs'][:]        # shape (N,) : +1 or -1
        box_size = f.attrs['box_size']

    # Select only positive masses
    mask_plus = signs > 0
    pos_plus = pos[mask_plus]
    N_plus = len(pos_plus)

    print(f"Particles+: {N_plus}")

    # Logarithmic bins from r=1 to r=box_size/2
    r_min = 1.0
    r_max = box_size / 2.0
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)

    # Pair counting with Corrfunc (optimized multi-thread)
    X = pos_plus[:, 0].astype(np.float64)
    Y = pos_plus[:, 1].astype(np.float64)
    Z = pos_plus[:, 2].astype(np.float64)

    results = DD(
        autocorr=1,
        nthreads=8,
        binfile=bins,
        X1=X, Y1=Y, Z1=Z,
        periodic=True,
        boxsize=box_size,
        verbose=False,
    )

    # Normalization: Poisson random distribution
    # For periodic box with N particles:
    # DD_random(r) = N*(N-1)/2 × (4pi/3 × (r_max^3 - r_min^3)) / box_size^3

    dd = results['npairs'].astype(np.float64)

    # Volume of each shell
    r_low = bins[:-1]
    r_high = bins[1:]
    vol_shells = (4.0/3.0) * np.pi * (r_high**3 - r_low**3)
    vol_box = box_size**3

    # Expected number of random pairs
    rr = (N_plus * (N_plus - 1) / 2.0) * vol_shells / vol_box

    # xi(r) = DD/RR - 1
    xi = dd / rr - 1.0
    r_centers = np.sqrt(r_low * r_high)  # Geometric center of bins

    return r_centers, xi


def plot_xi_evolution(
    snapshot_dir: str,
    steps: list,
    output_path: str = "xi_evolution.png"
):
    """
    Plot xi(r) at different steps to see temporal evolution.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

    for step, color in zip(steps, colors):
        snap_path = f"{snapshot_dir}/snap_{step:05d}.h5"
        try:
            r, xi = compute_xi(snap_path)
            ax.loglog(r, np.maximum(xi, 1e-3),
                     color=color,
                     label=f"Step {step}",
                     linewidth=1.5)
        except FileNotFoundError:
            print(f"Snapshot {snap_path} not found, skipped.")

    # SDSS DR7 reference (power law approximation)
    # xi(r) ~ (r/r0)^(-gamma) with r0~5 Mpc/h, gamma~1.8
    # In dimensionless units: to be calibrated
    r_ref = np.logspace(0, 2, 50)
    xi_ref = (r_ref / 20.0) ** (-1.8)  # Adjust r0=20 according to box_size
    ax.loglog(r_ref, xi_ref, 'k--',
             linewidth=2, label='SDSS DR7 (ref. slope)')

    ax.set_xlabel("r [simulation units]", fontsize=12)
    ax.set_ylabel("xi(r)", fontsize=12)
    ax.set_title("Correlation function — Positive masses\nJanus eta=1.045",
                fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-2, 1e3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Calculate xi(r) every 500 steps on Run A
    steps_to_analyze = list(range(0, 10001, 500))

    plot_xi_evolution(
        snapshot_dir="/mnt/T2/janus-sim/output/run_A/snapshots",
        steps=steps_to_analyze,
        output_path="/mnt/T2/janus-sim/output/run_A/xi_evolution.png"
    )
```

### Instructions for Claude CLI

```
TASK 4: Calculate xi(r) in post-processing
(ONLY after Task 3)

1. Install Corrfunc:
   pip install Corrfunc --break-system-packages

2. Execute compute_xi.py script above
   on Run A snapshots (eta=1.045 with virialized ICs).

3. Calculate xi(r) at steps: 0, 500, 1000, 2000, 5000
   (or available steps).

4. Generate xi(r) vs r plot with temporal evolution.
   Visually compare with SDSS slope (r^-1.8).

5. Report: is the xi(r) slope compatible with
   SDSS slope up to a scale factor?

Do not interpret absolute units — only slopes.
```

---

## TASK 5 — TEST eta=1.0 (theoretical limit case)
### Priority: QUICK — Can be done in parallel

### Grok's recommendation
If eta=1.0 gives even more redispersion than eta=1.045,
it confirms the physical interpretation of the quasi-symmetric regime.
Simple test, 30 minutes, very informative.

### Instructions for Claude CLI

```
TASK 5: Test eta=1.0 (limit case)

500K particles, eta=1.0, dt=0.01, virialized ICs, 200 steps.
Report Seg0, Seg100, Seg200, KE/KE0.

Expected comparison:
  eta=1.0   → even slower/null segregation
  eta=1.045 → slow segregation
  eta=2.03  → fast and stable segregation

If eta=1.0 gives null or decreasing segregation →
  confirms that eta~1 is a physical quasi-symmetric regime,
  not a numerical bug.
```

---

## RECOMMENDED EXECUTION ORDER

```
WEEK 1:
  [ ] Task 1: Virialization (2 days)
  [ ] Task 5: Test eta=1.0 (in parallel, 1 day)

WEEK 2:
  [ ] Task 2: Hubble friction (3 days)
  [ ] Consistency test expansion vs static

WEEK 3:
  [ ] Task 3: Convergence in N (3-4 days)
  [ ] Decision: are simulations defensible or not?

WEEK 4:
  [ ] Task 4: xi(r) in post-processing (2 days)
  [ ] Draft email to Jean-Pierre Petit

CONTACT PETIT:
  [ ] Pantheon+ fit (already ready)
  [ ] 3-4 Run A images with virialized ICs
  [ ] Qualitative xi(r) curve
  [ ] Open question: "Is the redispersion at eta=1.045
      physical or numerical?"
```

---

## ABSOLUTE RULES FOR CLAUDE CLI

```
1. Read VALIDATION_RULES.md at the beginning of each session.

2. Every new physics function:
   mandatory trivial test before use.

3. Never launch > 10 steps without explicit instruction.

4. Tasks in numerical order — no skipping.
   Do not start Task 2 before Task 1 validation.

5. After each Task, report results and wait for instruction.

6. Do not relaunch Run A or Run B with old parameters.
   Old simulations are obsolete without virialized ICs.

7. Any ambiguous decision → ask the question, do not decide alone.
```

---

## REFERENCES

- Petit, Margnat & Zejli (2024), EPJC 84:1226
- D'Agostini & Petit (2018), Astrophys. Space Sci. 363:139
- Lane et al. (2024), MNRAS (arXiv:2311.01438) — calibration bias
- Gemini 2.0 Flash — virialization and Hubble friction code
- ChatGPT o3 — initial conditions analysis
- Grok 3 — quasi-symmetric regime eta~1 analysis
