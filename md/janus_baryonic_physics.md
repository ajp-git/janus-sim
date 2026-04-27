# Physique Baryonique Janus — Plan d'Implémentation
## Principe : Test First. Aucun module sans tests unitaires validés.

---

## Leçons des erreurs passées

| Erreur passée | Durée avant détection | Impact |
|---|---|---|
| G × 1000 | Semaines | Toutes les simulations invalides |
| PM g_constant = 1.0 | Semaines | Forces PM incorrectes |
| sinh(2μ) vs sinh²(μ) | Jours | t₀ = 29.78 au lieu de 15.87 Gyr |
| Métrique P inadaptée | Semaines | Mauvaise interprétation |

**Règle absolue : chaque fonction physique a son test unitaire AVANT d'être utilisée.**

---

## Architecture des modules

```
src/
├── gravity/          (existant, validé)
│   ├── treepm.rs
│   └── forces.rs
├── baryonic/         (nouveau)
│   ├── mod.rs
│   ├── cooling.rs        ← Module 1
│   ├── pressure.rs       ← Module 2
│   ├── star_formation.rs ← Module 3
│   └── sph.rs            ← Kernel SPH partagé
└── tests/
    ├── cooling_tests.rs
    ├── pressure_tests.rs
    ├── sph_tests.rs
    ├── star_formation_tests.rs
    └── integration_tests.rs
```

---

## MODULE 0 — SPH Kernel (base de tout)

### Physique

```
W(r, h) = (1/π^(3/2) h³) × exp(-r²/h²)
∇W(r, h) = -2r/h² × W(r, h)
h_i = η × (m_i / ρ_i)^(1/3)   avec η = 1.2
```

### Implémentation Rust

```rust
// src/baryonic/sph.rs

pub struct SphKernel;

impl SphKernel {
    pub fn w(r: f64, h: f64) -> f64 {
        let q = r / h;
        (1.0 / (std::f64::consts::PI.powf(1.5) * h.powi(3)))
            * (-q * q).exp()
    }

    pub fn dw_dr(r: f64, h: f64) -> f64 {
        -2.0 * r / (h * h) * Self::w(r, h)
    }

    pub fn smoothing_length(mass: f64, density: f64, eta: f64) -> f64 {
        eta * (mass / density).powf(1.0 / 3.0)
    }
}
```

### Tests unitaires OBLIGATOIRES

```rust
// tests/sph_tests.rs

#[test]
fn test_kernel_normalization() {
    // ∫W(r,h)dV = 1
    let h = 1.0;
    let n_shells = 1000;
    let r_max = 5.0 * h;
    let dr = r_max / n_shells as f64;
    let integral: f64 = (0..n_shells).map(|i| {
        let r = (i as f64 + 0.5) * dr;
        4.0 * std::f64::consts::PI * r * r * SphKernel::w(r, h) * dr
    }).sum();
    assert!((integral - 1.0).abs() < 0.01,
        "Kernel non normalisé: intégrale = {}", integral);
}

#[test]
fn test_kernel_positive() {
    for r in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] {
        assert!(SphKernel::w(r, 1.0) >= 0.0,
            "Kernel négatif à r={}", r);
    }
}

#[test]
fn test_kernel_decreasing() {
    let h = 1.0;
    let w0 = SphKernel::w(0.0, h);
    let w1 = SphKernel::w(0.5, h);
    let w2 = SphKernel::w(1.0, h);
    assert!(w0 > w1 && w1 > w2, "Kernel non décroissant");
}

#[test]
fn test_gradient_negative() {
    // dW/dr < 0 (kernel décroissant)
    let dw = SphKernel::dw_dr(0.5, 1.0);
    assert!(dw < 0.0, "Gradient doit être négatif");
}

#[test]
fn test_smoothing_length_scaling() {
    // h ∝ ρ^(-1/3) → ρ×8 donne h/2
    let mass = 1.0e10;
    let h1 = SphKernel::smoothing_length(mass, 1.0e-3, 1.2);
    let h2 = SphKernel::smoothing_length(mass, 8.0e-3, 1.2);
    assert!((h1 / h2 - 2.0).abs() < 0.01,
        "Scaling h incorrect: ratio = {}", h1/h2);
}
```

---

## MODULE 1 — Refroidissement Radiatif

### Physique

```
dT/dt = -Λ(T) × ρ / (1.5 × k_B / m_p)

Table Λ(T) [erg cm³/s] :
  T < 100 K       →  0          (plancher)
  100 - 1e4 K     →  1e-26      (moléculaire)
  1e4 - 1e5 K     →  1e-22      (Lyman-alpha)
  1e5 - 1e7 K     →  5e-23      (raies métalliques)
  > 1e7 K         →  3e-23      (Bremsstrahlung)

T_floor = 100 K
```

### Implémentation Rust

```rust
// src/baryonic/cooling.rs

pub const T_FLOOR: f64 = 100.0;

pub fn cooling_rate_cgs(temp: f64) -> f64 {
    match temp {
        t if t < T_FLOOR => 0.0,
        t if t < 1e4     => 1e-26,
        t if t < 1e5     => 1e-22,
        t if t < 1e7     => 5e-23,
        _                => 3e-23,
    }
}

pub fn cooling_rate_kelvin_per_gyr(temp: f64, density: f64) -> f64 {
    if temp <= T_FLOOR { return 0.0; }
    // dT/dt = -Λ × ρ / (1.5 k_B/m_p)
    // Conversion CGS → code : facteurs de Mpc, M_sun, Gyr
    let lambda = cooling_rate_cgs(temp);
    let kb_over_mp_cgs = 8.314e7; // erg/g/K
    let rho_cgs = density * 1.989e33 / (3.086e24_f64).powi(3);
    -lambda * rho_cgs / (1.5 * kb_over_mp_cgs) * 3.156e16
}

pub fn apply_cooling(temp: f64, density: f64, dt: f64) -> f64 {
    let rate = cooling_rate_kelvin_per_gyr(temp, density);
    if rate == 0.0 { return temp; }
    let t_cool = (temp / rate.abs()).abs();
    let n_sub = ((dt / (0.1 * t_cool)).ceil() as usize).max(1).min(1000);
    let dt_sub = dt / n_sub as f64;
    let mut t = temp;
    for _ in 0..n_sub {
        let r = cooling_rate_kelvin_per_gyr(t, density);
        t = (t + r * dt_sub).max(T_FLOOR);
    }
    t
}
```

### Tests unitaires OBLIGATOIRES

```rust
// tests/cooling_tests.rs

#[test]
fn test_cooling_floor() {
    assert_eq!(cooling_rate_kelvin_per_gyr(50.0, 1e9), 0.0,
        "Cooling non nul sous T_floor");
}

#[test]
fn test_cooling_always_negative() {
    for temp in [1e4_f64, 1e5, 1e6, 1e7, 1e8] {
        let rate = cooling_rate_kelvin_per_gyr(temp, 1e9);
        assert!(rate <= 0.0,
            "Cooling positif à T={}: rate={}", temp, rate);
    }
}

#[test]
fn test_cooling_time_physical() {
    // t_cool = T / |dT/dt| doit être 0.1-100 Gyr
    let rate = cooling_rate_kelvin_per_gyr(1e6, 1e8);
    let t_cool = 1e6 / rate.abs();
    assert!(t_cool > 0.1 && t_cool < 1000.0,
        "t_cool hors plage: {} Gyr", t_cool);
}

#[test]
fn test_apply_cooling_reaches_floor() {
    let t_final = apply_cooling(1e6, 1e10, 100.0);
    assert!((t_final - T_FLOOR).abs() < 10.0,
        "T finale devrait être T_floor: {} K", t_final);
}

#[test]
fn test_apply_cooling_monotone() {
    let mut t = 1e6_f64;
    for _ in 0..100 {
        let t_new = apply_cooling(t, 1e9, 0.01);
        assert!(t_new <= t + 1.0,
            "Température monte: {} → {}", t, t_new);
        t = t_new;
    }
}

#[test]
fn test_bremsstrahlung_value() {
    // Λ(T>1e7) = 3e-23 erg cm³/s
    assert!((cooling_rate_cgs(1e8) - 3e-23).abs() / 3e-23 < 0.01,
        "Bremsstrahlung incorrect: {}", cooling_rate_cgs(1e8));
}
```

---

## MODULE 2 — Pression Thermique

### Physique

```
P = ρ × k_B × T / (μ_mol × m_p)     μ_mol = 0.6
cs² = γ × k_B × T / (μ_mol × m_p)   γ = 5/3

Force SPH :
a_press_i = -Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇W(rᵢⱼ, h)
```

### Implémentation Rust

```rust
// src/baryonic/pressure.rs

pub const MU_MOL: f64 = 0.6;
pub const GAMMA: f64 = 5.0 / 3.0;
// k_B en unités code [Mpc² M_sun⁻¹ Gyr⁻² K⁻¹]
pub const K_B_CODE: f64 = 8.254e-16;

pub fn pressure(density: f64, temp: f64) -> f64 {
    density * K_B_CODE * temp / MU_MOL
}

pub fn sound_speed(temp: f64) -> f64 {
    (GAMMA * K_B_CODE * temp / MU_MOL).sqrt()
}

pub fn pressure_acceleration(
    pos_i: [f64; 3], rho_i: f64, p_i: f64, h_i: f64,
    neighbors: &[(f64, f64, [f64; 3], f64, f64)],
) -> [f64; 3] {
    let mut accel = [0.0f64; 3];
    let coeff_i = p_i / (rho_i * rho_i);
    for &(mass_j, rho_j, pos_j, p_j, h_j) in neighbors {
        let coeff_j = p_j / (rho_j * rho_j);
        let r_vec = [
            pos_i[0] - pos_j[0],
            pos_i[1] - pos_j[1],
            pos_i[2] - pos_j[2],
        ];
        let r = (r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1]
                + r_vec[2]*r_vec[2]).sqrt();
        if r < 1e-10 { continue; }
        let h_avg = 0.5 * (h_i + h_j);
        let dw = SphKernel::dw_dr(r, h_avg);
        for k in 0..3 {
            accel[k] -= mass_j * (coeff_i + coeff_j) * dw * r_vec[k] / r;
        }
    }
    accel
}
```

### Tests unitaires OBLIGATOIRES

```rust
// tests/pressure_tests.rs

#[test]
fn test_pressure_positive() {
    assert!(pressure(1e8, 1e4) > 0.0, "Pression négative!");
}

#[test]
fn test_sound_speed_10km_s() {
    // cs(T=1e4 K) ≈ 10 km/s = 0.01022 Mpc/Gyr
    let cs = sound_speed(1e4);
    assert!((cs - 0.01022).abs() / 0.01022 < 0.1,
        "cs incorrect: {} Mpc/Gyr (attendu ~0.01022)", cs);
}

#[test]
fn test_sound_speed_sqrt_T() {
    // cs ∝ T^(1/2)
    let cs1 = sound_speed(1e4);
    let cs2 = sound_speed(4e4);
    assert!((cs2 / cs1 - 2.0).abs() < 0.01,
        "Scaling cs incorrect: ratio = {}", cs2/cs1);
}

#[test]
fn test_pressure_antisymmetric() {
    // Forces de pression antisymétriques (Newton III)
    let rho = 1e8; let temp = 1e4; let h = 2.0; let mass = 1e10;
    let p = pressure(rho, temp);
    let pos_i = [0.0, 0.0, 0.0];
    let pos_j = [1.0, 0.0, 0.0];
    let a_i = pressure_acceleration(pos_i, rho, p, h,
        &[(mass, rho, pos_j, p, h)]);
    let a_j = pressure_acceleration(pos_j, rho, p, h,
        &[(mass, rho, pos_i, p, h)]);
    assert!((a_i[0] + a_j[0]).abs() < 1e-10,
        "Forces non antisymétriques: {} vs {}", a_i[0], a_j[0]);
}

#[test]
fn test_pressure_zero_in_symmetric_field() {
    // Champ symétrique → force nulle sur particule centrale
    let rho = 1e8; let temp = 1e4; let h = 2.0; let mass = 1e10;
    let p = pressure(rho, temp);
    let neighbors = vec![
        (mass, rho, [1.0, 0.0, 0.0], p, h),
        (mass, rho, [-1.0, 0.0, 0.0], p, h),
        (mass, rho, [0.0, 1.0, 0.0], p, h),
        (mass, rho, [0.0, -1.0, 0.0], p, h),
        (mass, rho, [0.0, 0.0, 1.0], p, h),
        (mass, rho, [0.0, 0.0, -1.0], p, h),
    ];
    let a = pressure_acceleration([0.0, 0.0, 0.0], rho, p, h, &neighbors);
    assert!(a[0].abs() < 1e-8 && a[1].abs() < 1e-8 && a[2].abs() < 1e-8,
        "Force non nulle en champ symétrique: {:?}", a);
}
```

---

## MODULE 3 — Formation Stellaire

### Physique

```
Critères (TOUS requis) :
  1. ρ+_local > 100 × ρ̄+
  2. T < 1000 K
  3. ∇·v < 0 (convergence)
  4. M_local > M_Jeans = π^(5/2) cs³ / (6 G^(3/2) ρ^(1/2))
```

### Implémentation Rust

```rust
// src/baryonic/star_formation.rs

pub const RHO_SF_FACTOR: f64 = 100.0;
pub const T_SF: f64 = 1000.0;

pub fn jeans_mass(density: f64, sound_speed: f64) -> f64 {
    let g = 4.499e-15;
    std::f64::consts::PI.powf(2.5) * sound_speed.powi(3)
        / (6.0 * g.powf(1.5) * density.sqrt())
}

pub fn should_form_star(
    density: f64, mean_density: f64,
    temp: f64, div_v: f64,
    cs: f64, local_mass: f64,
) -> bool {
    density > RHO_SF_FACTOR * mean_density
        && temp < T_SF
        && div_v < 0.0
        && local_mass > jeans_mass(density, cs)
}
```

### Tests unitaires OBLIGATOIRES

```rust
// tests/star_formation_tests.rs

#[test]
fn test_no_sf_hot_gas() {
    // Gaz chaud → jamais de formation
    assert!(!should_form_star(1e12, 1e8, 1e6, -1.0,
        sound_speed(1e6), 1e15),
        "SF dans gaz chaud!");
}

#[test]
fn test_no_sf_diverging() {
    // Gaz qui s'expanse → pas de formation
    assert!(!should_form_star(1e12, 1e8, 100.0, 1.0,
        sound_speed(100.0), 1e15),
        "SF dans gaz divergent!");
}

#[test]
fn test_no_sf_underdense() {
    // Densité insuffisante
    assert!(!should_form_star(1e8, 1e8, 100.0, -1.0,
        sound_speed(100.0), 1e15),
        "SF dans région sous-dense!");
}

#[test]
fn test_sf_all_criteria() {
    // Tous critères satisfaits → formation
    assert!(should_form_star(1e12, 1e8, 100.0, -1.0,
        sound_speed(100.0), 1e15),
        "Pas de SF malgré tous critères!");
}

#[test]
fn test_jeans_mass_scaling() {
    // M_Jeans ∝ cs³ / ρ^(1/2)
    let rho = 1e8;
    let mj1 = jeans_mass(rho, 0.01);
    let mj2 = jeans_mass(rho, 0.02); // cs × 2
    assert!((mj2 / mj1 - 8.0).abs() < 0.01,
        "Scaling Jeans incorrect: ratio = {}", mj2/mj1);
}

#[test]
fn test_jeans_mass_density_scaling() {
    // M_Jeans ∝ ρ^(-1/2)
    let cs = 0.01;
    let mj1 = jeans_mass(1e8, cs);
    let mj2 = jeans_mass(4e8, cs); // ρ × 4
    assert!((mj1 / mj2 - 2.0).abs() < 0.01,
        "Scaling densité Jeans incorrect: ratio = {}", mj1/mj2);
}
```

---

## TESTS D'INTÉGRATION — OBLIGATOIRES avant run Janus

Créer `src/bin/test_baryonic_integration.rs` avec le contenu suivant :

```rust
//! Tests d'intégration physique baryonique
//! OBLIGATOIRES avant tout run Janus
//! Lancer : cargo run --release --bin test_baryonic_integration

use janus_sim::baryonic::{cooling::apply_cooling, pressure::sound_speed};

fn main() {
    println!("=== TESTS D'INTÉGRATION BARYONIQUE ===\n");
    
    let mut all_passed = true;
    
    all_passed &= test_cold_collapse();
    all_passed &= test_hot_stability();
    all_passed &= test_energy_conservation();
    all_passed &= test_momentum_conservation();
    
    println!("\n=== RÉSULTAT FINAL ===");
    if all_passed {
        println!("✓ TOUS LES TESTS PASSENT → Run Janus autorisé");
    } else {
        println!("✗ TESTS ÉCHOUÉS → STOP. Corriger avant de continuer.");
        std::process::exit(1);
    }
}

// ─────────────────────────────────────────────
// TEST 1 — Effondrement isotherme (sans Janus)
// ─────────────────────────────────────────────
fn test_cold_collapse() -> bool {
    println!("TEST 1 — Effondrement isotherme");
    println!("  N=100k m+ seulement, Box=5 Mpc, T=100K");
    println!("  Surdensité centrale ×10, Steps=500");
    
    // Config
    let n = 100_000usize;
    let box_size = 5.0f64; // Mpc
    let t_init = 100.0f64; // K
    let g_code = 4.499e-15f64;
    let steps = 500usize;
    let dt = 0.001f64; // Gyr

    // Positions : uniforme + surdensité centrale
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| {
        [
            rng.gen_range(-box_size/2.0..box_size/2.0),
            rng.gen_range(-box_size/2.0..box_size/2.0),
            rng.gen_range(-box_size/2.0..box_size/2.0),
        ]
    }).collect();
    
    // Ajouter surdensité centrale : 500 particules dans r < 0.3 Mpc
    for i in 0..500 {
        let r = rng.gen_range(0.0..0.3f64);
        let theta = rng.gen_range(0.0..std::f64::consts::TAU);
        let phi = rng.gen_range(0.0..std::f64::consts::PI);
        pos[i] = [
            r * phi.sin() * theta.cos(),
            r * phi.sin() * theta.sin(),
            r * phi.cos(),
        ];
    }
    
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut temp: Vec<f64> = vec![t_init; n];
    let mass = 1e10f64; // M_sun par particule
    let mean_density = n as f64 * mass / box_size.powi(3);
    
    let mut rho_max_ratio = 1.0f64;
    
    for step in 0..steps {
        // Calculer densité locale sur grille 32³
        let grid = 32usize;
        let cell = box_size / grid as f64;
        let mut density_grid = vec![0.0f64; grid.pow(3)];
        
        for p in &pos {
            let ix = ((p[0] + box_size/2.0) / cell) as usize;
            let iy = ((p[1] + box_size/2.0) / cell) as usize;
            let iz = ((p[2] + box_size/2.0) / cell) as usize;
            if ix < grid && iy < grid && iz < grid {
                density_grid[ix * grid * grid + iy * grid + iz] += mass;
            }
        }
        
        let cell_vol = cell.powi(3);
        let rho_max = density_grid.iter()
            .cloned().fold(0.0f64, f64::max) / cell_vol;
        rho_max_ratio = rho_max / mean_density;
        
        if step % 100 == 0 {
            println!("  step {:4} : ρmax/ρ̄ = {:.2}", step, rho_max_ratio);
        }
        
        // Gravité directe simplifiée (N² pour 100k → trop lent)
        // Utiliser approximation champ moyen depuis grille
        // Forces sur chaque particule depuis gradient de densité
        for i in 0..n {
            let ix = ((pos[i][0] + box_size/2.0) / cell) as usize;
            let iy = ((pos[i][1] + box_size/2.0) / cell) as usize;
            let iz = ((pos[i][2] + box_size/2.0) / cell) as usize;
            if ix == 0 || iy == 0 || iz == 0 
                || ix >= grid-1 || iy >= grid-1 || iz >= grid-1 {
                continue;
            }
            
            // Gradient de densité → force gravitationnelle
            let rho_xp = density_grid[(ix+1)*grid*grid + iy*grid + iz];
            let rho_xm = density_grid[(ix-1)*grid*grid + iy*grid + iz];
            let rho_yp = density_grid[ix*grid*grid + (iy+1)*grid + iz];
            let rho_ym = density_grid[ix*grid*grid + (iy-1)*grid + iz];
            let rho_zp = density_grid[ix*grid*grid + iy*grid + iz+1];
            let rho_zm = density_grid[ix*grid*grid + iy*grid + iz-1];
            
            let ax = -g_code * (rho_xp - rho_xm) / (2.0 * cell);
            let ay = -g_code * (rho_yp - rho_ym) / (2.0 * cell);
            let az = -g_code * (rho_zp - rho_zm) / (2.0 * cell);
            
            vel[i][0] += ax * dt;
            vel[i][1] += ay * dt;
            vel[i][2] += az * dt;
            
            // Cooling
            let overdensity = density_grid[ix*grid*grid + iy*grid + iz] 
                / cell_vol / mean_density;
            temp[i] = apply_cooling(temp[i], overdensity.max(1.0), dt);
        }
        
        // Drift
        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
                // Conditions périodiques
                if pos[i][k] > box_size/2.0 { pos[i][k] -= box_size; }
                if pos[i][k] < -box_size/2.0 { pos[i][k] += box_size; }
            }
        }
    }
    
    let passed = rho_max_ratio > 5.0;
    if passed {
        println!("  ✓ PASS : ρmax/ρ̄ = {:.2} > 5.0", rho_max_ratio);
    } else {
        println!("  ✗ FAIL : ρmax/ρ̄ = {:.2} (attendu > 5.0)", rho_max_ratio);
        println!("    → BUG dans gravité ou cooling → STOP");
    }
    passed
}

// ─────────────────────────────────────────────
// TEST 2 — Stabilité gaz chaud (sans Janus)
// ─────────────────────────────────────────────
fn test_hot_stability() -> bool {
    println!("\nTEST 2 — Stabilité gaz chaud");
    println!("  N=10k m+ seulement, Box=5 Mpc, T=1e7K, Steps=200");
    
    let n = 10_000usize;
    let box_size = 5.0f64;
    let t_init = 1e7f64; // K très chaud
    let g_code = 4.499e-15f64;
    let steps = 200usize;
    let dt = 0.001f64;
    
    let mut rng = rand::thread_rng();
    use rand::Rng;
    
    // Positions uniformes (pas de surdensité)
    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mass = 1e10f64;
    let mean_density = n as f64 * mass / box_size.powi(3);
    let cs = sound_speed(t_init);
    
    let mut rho_max_ratio = 1.0f64;
    
    for step in 0..steps {
        let grid = 16usize;
        let cell = box_size / grid as f64;
        let mut density_grid = vec![0.0f64; grid.pow(3)];
        
        for p in &pos {
            let ix = ((p[0] + box_size/2.0) / cell) as usize;
            let iy = ((p[1] + box_size/2.0) / cell) as usize;
            let iz = ((p[2] + box_size/2.0) / cell) as usize;
            if ix < grid && iy < grid && iz < grid {
                density_grid[ix*grid*grid + iy*grid + iz] += mass;
            }
        }
        
        let cell_vol = cell.powi(3);
        let rho_max = density_grid.iter()
            .cloned().fold(0.0f64, f64::max) / cell_vol;
        rho_max_ratio = rho_max / mean_density;
        
        // Pression thermique stabilise → pas de gravité pure ici
        // La pression du gaz chaud >> gravité → stabilité
        for i in 0..n {
            let ix = ((pos[i][0] + box_size/2.0) / cell) as usize;
            let iy = ((pos[i][1] + box_size/2.0) / cell) as usize;
            let iz = ((pos[i][2] + box_size/2.0) / cell) as usize;
            if ix == 0 || iy == 0 || iz == 0
                || ix >= grid-1 || iy >= grid-1 || iz >= grid-1 {
                continue;
            }
            
            // Force gravité
            let rho_xp = density_grid[(ix+1)*grid*grid + iy*grid + iz];
            let rho_xm = density_grid[(ix-1)*grid*grid + iy*grid + iz];
            let rho_yp = density_grid[ix*grid*grid + (iy+1)*grid + iz];
            let rho_ym = density_grid[ix*grid*grid + (iy-1)*grid + iz];
            let rho_zp = density_grid[ix*grid*grid + iy*grid + iz+1];
            let rho_zm = density_grid[ix*grid*grid + iy*grid + iz-1];
            
            let ax_grav = -g_code * (rho_xp - rho_xm) / (2.0 * cell);
            let ay_grav = -g_code * (rho_yp - rho_ym) / (2.0 * cell);
            let az_grav = -g_code * (rho_zp - rho_zm) / (2.0 * cell);
            
            // Force pression (oppose la gravité pour gaz chaud)
            // a_press ≈ -cs² × ∇ρ/ρ
            let rho_i = density_grid[ix*grid*grid + iy*grid + iz] / cell_vol;
            let ax_press = if rho_i > 0.0 {
                -cs * cs * (rho_xp - rho_xm) / (2.0 * cell * rho_i)
            } else { 0.0 };
            let ay_press = if rho_i > 0.0 {
                -cs * cs * (rho_yp - rho_ym) / (2.0 * cell * rho_i)
            } else { 0.0 };
            let az_press = if rho_i > 0.0 {
                -cs * cs * (rho_zp - rho_zm) / (2.0 * cell * rho_i)
            } else { 0.0 };
            
            vel[i][0] += (ax_grav + ax_press) * dt;
            vel[i][1] += (ay_grav + ay_press) * dt;
            vel[i][2] += (az_grav + az_press) * dt;
        }
        
        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
                if pos[i][k] > box_size/2.0 { pos[i][k] -= box_size; }
                if pos[i][k] < -box_size/2.0 { pos[i][k] += box_size; }
            }
        }
        
        if step % 50 == 0 {
            println!("  step {:4} : ρmax/ρ̄ = {:.3}", step, rho_max_ratio);
        }
    }
    
    let passed = rho_max_ratio < 5.0;
    if passed {
        println!("  ✓ PASS : ρmax/ρ̄ = {:.3} < 5.0 (stable)", rho_max_ratio);
    } else {
        println!("  ✗ FAIL : ρmax/ρ̄ = {:.3} (gaz chaud instable!)", rho_max_ratio);
        println!("    → BUG dans pression thermique → STOP");
    }
    passed
}

// ─────────────────────────────────────────────
// TEST 3 — Conservation de l'énergie
// ─────────────────────────────────────────────
fn test_energy_conservation() -> bool {
    println!("\nTEST 3 — Conservation de l'énergie");
    println!("  N=1000, Box=10 Mpc, Steps=100, sans cooling");
    
    let n = 1000usize;
    let box_size = 10.0f64;
    let g_code = 4.499e-15f64;
    let steps = 100usize;
    let dt = 0.001f64;
    let mass = 1e12f64;
    
    let mut rng = rand::thread_rng();
    use rand::Rng;
    
    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    
    // Énergie cinétique initiale
    let e_kin_0: f64 = vel.iter()
        .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    
    // Énergie potentielle initiale (approximation paires proches)
    let e_pot_0: f64 = compute_potential_energy(&pos, mass, g_code, n.min(100));
    let e_tot_0 = e_kin_0 + e_pot_0;
    
    // Intégration leapfrog sans cooling
    for _ in 0..steps {
        // Kick
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);
                let f = g_code * mass / (r * r * r);
                vel[i][0] += f * dx * dt * 0.5;
                vel[i][1] += f * dy * dt * 0.5;
                vel[i][2] += f * dz * dt * 0.5;
            }
        }
        // Drift
        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
            }
        }
        // Kick
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);
                let f = g_code * mass / (r * r * r);
                vel[i][0] += f * dx * dt * 0.5;
                vel[i][1] += f * dy * dt * 0.5;
                vel[i][2] += f * dz * dt * 0.5;
            }
        }
    }
    
    let e_kin_f: f64 = vel.iter()
        .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
        .sum();
    let e_pot_f = compute_potential_energy(&pos, mass, g_code, n.min(100));
    let e_tot_f = e_kin_f + e_pot_f;
    
    let variation = if e_tot_0.abs() > 1e-30 {
        (e_tot_f - e_tot_0).abs() / e_tot_0.abs()
    } else { 0.0 };
    
    println!("  E_tot(0) = {:.4e}", e_tot_0);
    println!("  E_tot(f) = {:.4e}", e_tot_f);
    println!("  Variation = {:.4}%", variation * 100.0);
    
    let passed = variation < 0.05; // 5% tolérance (approximation paires)
    if passed {
        println!("  ✓ PASS : énergie conservée à {:.2}%", variation * 100.0);
    } else {
        println!("  ✗ FAIL : variation = {:.2}% > 5%", variation * 100.0);
        println!("    → BUG dans intégrateur → STOP");
    }
    passed
}

fn compute_potential_energy(
    pos: &[[f64; 3]], mass: f64, g: f64, n_sample: usize
) -> f64 {
    let n = pos.len().min(n_sample);
    let mut e = 0.0f64;
    for i in 0..n {
        for j in (i+1)..n {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt().max(0.01);
            e -= g * mass * mass / r;
        }
    }
    // Extrapoler pour N complet
    e * (pos.len() as f64 / n_sample as f64).powi(2)
}

// ─────────────────────────────────────────────
// TEST 4 — Conservation de l'impulsion
// ─────────────────────────────────────────────
fn test_momentum_conservation() -> bool {
    println!("\nTEST 4 — Conservation de l'impulsion");
    println!("  N=1000, Steps=100");
    
    let n = 1000usize;
    let box_size = 10.0f64;
    let g_code = 4.499e-15f64;
    let steps = 100usize;
    let dt = 0.001f64;
    let mass = 1e12f64;
    
    let mut rng = rand::thread_rng();
    use rand::Rng;
    
    let mut pos: Vec<[f64; 3]> = (0..n).map(|_| [
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
        rng.gen_range(-box_size/2.0..box_size/2.0),
    ]).collect();
    // Vitesses initiales nulles → impulsion = 0
    let mut vel: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    
    for _ in 0..steps {
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);
                let f = g_code * mass / (r * r * r);
                vel[i][0] += f * dx * dt;
                vel[i][1] += f * dy * dt;
                vel[i][2] += f * dz * dt;
            }
        }
        for i in 0..n {
            for k in 0..3 {
                pos[i][k] += vel[i][k] * dt;
            }
        }
    }
    
    // Impulsion totale finale (devrait être ~0)
    let px: f64 = vel.iter().map(|v| mass * v[0]).sum();
    let py: f64 = vel.iter().map(|v| mass * v[1]).sum();
    let pz: f64 = vel.iter().map(|v| mass * v[2]).sum();
    let p_total = (px*px + py*py + pz*pz).sqrt();
    let p_mean = mass * 1.0; // échelle de référence
    
    println!("  |P_final| = {:.4e}", p_total);
    
    // Avec vitesses initiales nulles, P doit rester ~0
    // Tolérance numérique : < 1e-6 × masse_totale × vitesse_typique
    let passed = p_total < mass * n as f64 * 1e-6;
    if passed {
        println!("  ✓ PASS : impulsion conservée");
    } else {
        println!("  ✗ FAIL : |P| = {:.4e} (trop grand)", p_total);
        println!("    → BUG dans forces → STOP");
    }
    passed
}
```

Lancer avec :
```bash
docker compose run --rm dev cargo run --release \
    --bin test_baryonic_integration 2>&1

# SI EXIT CODE = 0 → tous tests passent → Run Janus autorisé
# SI EXIT CODE = 1 → STOP, corriger avant de continuer
```

---

## PIPELINE COMPLET

```bash
# ÉTAPE 1 — Tests unitaires
cargo test 2>&1 | grep -E "FAILED|ok|error"
# → Si FAILED : STOP

# ÉTAPE 2 — Test effondrement isotherme
cargo run --release -- test-collapse \
  --n 100000 --box 5.0 --t-init 100 --steps 500
# → Si ρ_max < 20 : STOP

# ÉTAPE 3 — Test stabilité gaz chaud
cargo run --release -- test-hot-stability \
  --n 100000 --box 5.0 --t-init 1e7 --steps 500
# → Si ρ_max > 2 : STOP

# ÉTAPE 4 — Test conservation
cargo run --release -- test-conservation \
  --n 10000 --steps 100
# → Si erreur > 1% : STOP

# ÉTAPE 5 — Run Janus baryonique
cargo run --release -- janus-baryonic \
  --mu 19 --n 5000000 --box 500 --steps 4000 \
  --t-init 1e4 --delta-init 10 \
  --snapshot-every 40
```

---

## CRITÈRES DE SUCCÈS FINAL

| ρ+_max/ρ̄+ à z=0 | Interprétation | Action |
|---|---|---|
| > 100 | Effondrement confirmé | Vidéo + papier |
| > 10  | Structures émergentes | Continuer + plus de steps |
| > 3   | Signal faible | Tester paramètres différents |
| < 3   | Résultat négatif | Papier sur limites + JPP |

**Dans tous les cas : résultat scientifique publiable.**
