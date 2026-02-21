# VALIDATION_RULES.md
# Mandatory Validation Rules — Janus Project
# Read at the beginning of each Claude CLI session

## FUNDAMENTAL RULE

**Every new physics function must be validated on a trivial case
with known result BEFORE being used in a simulation.**

Mandatory order:
1. Write the trivial test
2. Verify the test passes
3. Only then use the function
4. Never bypass this procedure even under time pressure

---

## 1. SEGREGATION

### Mandatory Test
```rust
#[test]
fn test_segregation_trivial() {
    // 4 positive particles on right, 4 negative particles on left
    // COM+ = (10, 0, 0), COM- = (-10, 0, 0)
    // Expected distance = 20.0 EXACTLY
    let positions_plus  = vec![(10.0, 0.0, 0.0); 4];
    let positions_minus = vec![(-10.0, 0.0, 0.0); 4];
    let distance = compute_segregation(&positions_plus, &positions_minus);
    assert!((distance - 20.0).abs() < 1e-10,
        "Expected COM distance 20.0, got {}", distance);
}
```

### Mandatory Normalization
```rust
// Segregation must ALWAYS be normalized by box_size
let segregation = distance_com / box_size;
// Typical values: 0.0 (mixed) to ~0.5 (segregated)
// Phase 1b validated: Seg₀ ≈ 0.029, Seg_final ≈ 0.163 (+294%)
```

### Known Pitfall — Periodic Boundary Conditions
```
CLASSIC ERROR: Computing COM with simple position average
when particles wrap around the box.

BUG EXAMPLE:
  Box: half_size = 50
  Particle A: x = +49
  Particle B: x = -49
  Naive average: (49 + (-49)) / 2 = 0  ← WRONG
  Real distance: 2                      ← CORRECT

SOLUTION: "minimum image convention" algorithm
```

```rust
fn minimum_image(dx: f64, box_size: f64) -> f64 {
    let mut d = dx % box_size;
    if d > box_size / 2.0  { d -= box_size; }
    if d < -box_size / 2.0 { d += box_size; }
    d
}

fn compute_com_periodic(positions: &[(f64,f64,f64)], box_size: f64) -> (f64,f64,f64) {
    // Use first particle position as reference
    let ref_pos = positions[0];
    let mut sum = (0.0f64, 0.0f64, 0.0f64);
    for &(x, y, z) in positions {
        sum.0 += ref_pos.0 + minimum_image(x - ref_pos.0, box_size);
        sum.1 += ref_pos.1 + minimum_image(y - ref_pos.1, box_size);
        sum.2 += ref_pos.2 + minimum_image(z - ref_pos.2, box_size);
    }
    let n = positions.len() as f64;
    (sum.0 / n, sum.1 / n, sum.2 / n)
}
```

### Mandatory Minimum Image Test
```rust
#[test]
fn test_minimum_image() {
    let box_size = 100.0;
    // Two particles "close" across periodic boundary
    assert!((minimum_image(49.0 - (-49.0), box_size) + 2.0).abs() < 1e-10,
        "Expected periodic distance -2.0");
    assert!((minimum_image(1.0, box_size) - 1.0).abs() < 1e-10,
        "Expected normal distance 1.0");
}
```

---

## 2. ENERGY CONSERVATION

### Mandatory Test
```rust
#[test]
fn test_energy_conservation_2body() {
    // 2 same-sign particles, far apart
    // Initial KE = final KE within 0.01% over 100 steps
    let ke_initial = simulate_2body(steps=100);
    let ke_final = ...;
    let drift = (ke_final - ke_initial).abs() / ke_initial;
    assert!(drift < 1e-4,
        "Energy drift {:.4}% > 0.01%", drift * 100.0);
}
```

### Validation Thresholds by Simulation
| Simulation | Max acceptable KE/KE₀ | Expected segregation |
|------------|----------------------|----------------------|
| 2-body     | 1.001 (0.1%)         | Not applicable       |
| 100K test  | < 50 at step 50      | Must increase        |
| 1M Phase1b | 113 at step 200      | +294% validated      |
| 5M Phase1c | < 200 at step 100    | To be determined     |

---

## 3. JANUS FORCES

### Mandatory Test — Interaction Laws
```rust
#[test]
fn test_janus_force_laws() {
    let origin = [0.0, 0.0, 0.0];
    let right   = [1.0, 0.0, 0.0];

    // Same sign → ATTRACTION (force toward +x)
    let f_pp = janus_force(origin, right, sign_i=+1, sign_j=+1);
    assert!(f_pp[0] > 0.0, "Mass+ attracts Mass+ toward +x");

    let f_mm = janus_force(origin, right, sign_i=-1, sign_j=-1);
    assert!(f_mm[0] > 0.0, "Mass- attracts Mass- toward +x");

    // Opposite signs → REPULSION (force toward -x)
    let f_pm = janus_force(origin, right, sign_i=+1, sign_j=-1);
    assert!(f_pm[0] < 0.0, "Mass+ repels Mass- toward -x");

    let f_mp = janus_force(origin, right, sign_i=-1, sign_j=+1);
    assert!(f_mp[0] < 0.0, "Mass- repels Mass+ toward -x");

    // Magnitude: G*m/r² (softening negligible at r=1)
    let expected = G * 1.0 / 1.0;
    assert!((f_pp[0] - expected).abs() / expected < 0.01,
        "Incorrect magnitude");
}
```

---

## 4. PERIODIC BOUNDARY CONDITIONS

### Mandatory Test
```rust
#[test]
fn test_periodic_bc() {
    let box_half = 50.0;

    // Particle exiting right
    let mut x = 51.0;
    apply_periodic(&mut x, box_half);
    assert!((x - (-49.0)).abs() < 1e-10, "Wrap right: {}", x);

    // Particle exiting left
    let mut x = -51.0;
    apply_periodic(&mut x, box_half);
    assert!((x - 49.0).abs() < 1e-10, "Wrap left: {}", x);

    // Particle inside box — should not move
    let mut x = 10.0;
    apply_periodic(&mut x, box_half);
    assert!((x - 10.0).abs() < 1e-10, "No wrap: {}", x);
}
```

---

## 5. FRIEDMANN INTEGRATION

### Mandatory Test — Exact Parametric Solution
```rust
#[test]
fn test_friedmann_vs_parametric() {
    // Compare RK4 integrator with exact solution
    // a(u) = α²·ch²(u) for η = 1.045
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let history = integrate_backward(&params, 2.0, 5000);

    // At z=1 (a=0.5), compare with parametric solution
    // Acceptable error: < 1%
    let a_numerical = interpolate_a(&history, z=1.0);
    let a_parametric = compute_parametric(eta, z=1.0);
    let error = (a_numerical - a_parametric).abs() / a_parametric;
    assert!(error < 0.01, "Friedmann/parametric error {:.2}%", error*100.0);
}
```

---

## 6. SNIa FORMULA (mu_janus_for_fit)

### Mandatory Test — Consistency with Known Result
```rust
#[test]
fn test_mu_janus_known_values() {
    // Reference values: validated Pantheon+ fit
    // η = 1.045, q0 = -0.022, cst = 23.856

    let eta = 1.045;
    let cst = 23.856;

    // At z=0.5, μ should be ≈ 42.3 (from janus_exact_fit.json)
    let mu = mu_janus_for_fit(0.5, eta, cst);
    assert!((mu - 42.3).abs() < 0.1,
        "μ(z=0.5) expected ~42.3, got {}", mu);

    // Monotonicity: μ must increase with z
    for z in [0.1, 0.5, 1.0, 2.0] {
        let mu_prev = mu_janus_for_fit(z * 0.9, eta, cst);
        let mu_curr = mu_janus_for_fit(z, eta, cst);
        assert!(mu_curr > mu_prev, "μ not monotonic at z={}", z);
    }
}
```

---

## 7. COSMOINTERPOLATOR — Friedmann ↔ N-body Synchronization

### Context
Task 2 (Hubble friction) requires passing a(t) and H(t)
to the CUDA kernel at each step. These values come from friedmann.rs
but time units are different.

### Convention confirmed in friedmann.rs (line 106)
```rust
let a_dot = params.omega_plus.sqrt();
// H₀ = √Ω₊ in dimensionless units
```
For η=1.045: H₀ = √(1/2.045) = 0.6993

### Mandatory Test Before Task 2
```rust
#[test]
fn test_cosmo_interpolator() {
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, 50.0);

    // Past: a = 1/(1+z_init)
    let (a_start, _) = cosmo.get_params_at_tau(cosmo.tau_start);
    assert!((a_start - 1.0/51.0).abs() < 1e-4);

    // Present: a = 1.0 exactly
    let (a_end, h_end) = cosmo.get_params_at_tau(cosmo.tau_end);
    assert!((a_end - 1.0).abs() < 1e-4);

    // H₀ = √Ω₊ (friedmann.rs line 106 convention)
    assert!((h_end - params.omega_plus.sqrt()).abs() < 1e-4);

    // Monotonicity
    let tau_mid = cosmo.tau_start + (cosmo.tau_end - cosmo.tau_start) / 2.0;
    let (a_mid, _) = cosmo.get_params_at_tau(tau_mid);
    assert!(a_mid > a_start && a_mid < a_end);
}
```

### Known Pitfall — Ghost Functions
```
These functions DO NOT EXIST in friedmann.rs:
  integrate_friedmann(eta, t)   ← NONEXISTENT
  interpolate_a(&history, t)    ← NONEXISTENT

Use CosmoInterpolator::new() and get_params_at_tau()
(code in janus_roadmap.md Task 2)
```

### Known Pitfall — Time Unit Mismatch
```
t_nbody (dt=0.01 arbitrary) ≠ τ cosmological (H₀·t)

Mandatory conversion:
  dtau_cosmo = (tau_end - tau_start) / total_nbody_steps

Never pass t_nbody directly to get_params_at_tau()
```

---

## 8. N-BODY FORCE METHOD

### Method Test — Before Any Simulation > 100K Particles
```
The force method must be validated on a simple physical case:

MANDATORY TEST: Gravitational collapse
  - 1000 same-sign particles, spherical distribution
  - After 50 steps: particles must move toward center
  - Mean radius must decrease by > 5%

ANTI-PM TEST: Segregation on 100K
  - 50K positive particles, 50K negative particles
  - After 200 steps: segregation must increase by > 10%
  - If segregation stagnates: PM method unusable for Janus
  - → Use Barnes-Hut mandatory

VALIDATED METHODS for Janus:
  ✅ Barnes-Hut CPU (Rust) — Phase 1b validated
  ✅ Barnes-Hut GPU (Rust+CUDA) — validated 0% deviation
  ❌ Particle-Mesh (Python+CuPy) — zero segregation, unusable
```

---

## 9. SIMULATION LAUNCH PROCEDURE

### Mandatory Checklist Before Any Launch > 1M Particles

```
□ 1. Parameters tested on 100K, 50 steps
     → KE/KE₀ < 50 at step 50 ✓
     → Segregation increases between step 0 and step 50 ✓

□ 2. Seeds synchronized CPU/GPU if comparison
     → Segregation deviation < 5% ✓

□ 3. Total time estimate calculated
     → steps × time/step = acceptable duration ✓

□ 4. Disk space verified
     → NPZ snapshots + PNG frames ✓

□ 5. nohup + PID saved

□ 6. Monitoring at steps 20, 50, 100 planned
     → Stop criteria defined BEFORE launch

□ 7. DO NOT LAUNCH without explicit validation
```

---

## 10. CLAUDE CLI BEHAVIOR RULES

```
1. Never launch a simulation without explicit instruction.

2. Always report step 20 and wait for confirmation
   before continuing beyond step 50.

3. If a stop criterion is met → stop and report.
   Do not "let it run for the frames".

4. Any parameter change (softening, dt, method)
   from the validated version must be reported
   and re-validated on 100K before application.

5. Never interpret an ambiguous result as "correct".
   Always ask the question explicitly.

6. Every physics measurement function (segregation, energy,
   COM, distance) must have its trivial test in the code.
```

---

## HISTORY OF ENCOUNTERED BUGS

| Bug | Cause | Lesson |
|-----|-------|--------|
| Incorrect acceleration equations | Local densities instead of conserved E | Always verify vs source paper |
| Analytical/numerical deviation 0.8 mag | Standard Friedmann H(z) mixed with Janus acc | Theoretical consistency before implementation |
| Zero segregation with PM | PM method smooths short-range interactions | Validate method on known physical case |
| OOM 10M f64 on 12GB VRAM | Incorrect VRAM estimate (tree ×8) | Calculate VRAM before launching |
| Wrong COM with periodic conditions | Simple average ignores wrap | Always use minimum image convention |
| Launch without instruction | Autonomous Claude CLI | Mandatory checklist before launch |
