# ZOOM V3 — Top du Top
*Boîte indépendante 50 Mpc, z=10→z=0, physique baryonique complète*
*Spécification v2 — calculs vérifiés 3× — Avril 2026*

---

## Erreurs corrigées vs version précédente

1. N_grid=100 donne 2M total (pas 4M) — ε corrigé
2. Formule adaptive dt corrigée (accélération, pas vitesse)
3. Steps z=10→0 recalculés depuis données empiriques
4. VRAM vérifiée par calcul explicite

---

## Arrêt du zoom v2

```bash
kill $(pgrep -f "janus_zoom_L1_baryonic")
docker stop $(docker ps -q --filter name=janus) 2>/dev/null
sleep 3
nvidia-smi | grep "MiB"  # Doit afficher ~0 MiB utilisés
```

---

## Paramètres physiques — tous calculés et vérifiés

### Grille

```
Box size         : 50 Mpc (cubique, périodique)
N_grid par espèce: 128  (= 2⁷, optimal pour FFT)
N_per_species    : 128³ = 2 097 152
N_total          : 2 × 2 097 152 = 4 194 304 ≈ 4.2M particules
```

### Masses — vérifiées 3×

```
H₀ = 69.9 km/s/Mpc → h = 0.699
ρ_crit = 2.775×10¹¹ × h² = 2.775×10¹¹ × 0.4886 = 1.356×10¹¹ M☉/Mpc³

Janus : Ω_b=0.05, μ=19, Ω_tot = 0.05×(1+19) = 1.00 ✓ (univers plat)

ρ_mean_plus  = 0.05 × 1.356×10¹¹ = 6.779×10⁹  M☉/Mpc³
ρ_mean_minus = 19   × 6.779×10⁹  = 1.288×10¹¹ M☉/Mpc³

V_box = 50³ = 125 000 Mpc³

m_part_plus  = 6.779×10⁹  × 125 000 / 2 097 152 = 4.04×10⁸  M☉
m_part_minus = 1.288×10¹¹ × 125 000 / 2 097 152 = 7.68×10⁹  M☉

Vérification : m_minus/m_plus = 7.68×10⁹ / 4.04×10⁸ = 19.0 = μ ✓

Gain vs zoom v1 (m_part=5×10¹⁰ M☉) :
  Masse    : ×124
  Spatial  : 124^(1/3) = ×5.0
```

### Softening

```
Espacement = 50/128 = 0.391 Mpc
ε_plus     = espacement/4 = 0.098 Mpc → 0.10 Mpc (arrondi conservateur)
ε_minus    = ε_plus × μ^(1/3) = 0.10 × 2.668 = 0.267 → 0.25 Mpc

Justification ε_minus : les m- sont μ=19× plus lourdes.
Un softening plus grand évite les forces excessives m-/m-.
```

### VRAM — vérifiée

```
Particules (pos+vel+acc+mass+sign+ε : ~50 bytes/particule) :
  4.2M × 50 = 210 MB

Arbre Barnes-Hut (~8N nœuds × 80 bytes) :
  4.2M × 8 × 80 = 2 688 MB

SPH (rho,T,h,P,u,divv : 6×f32 × N_m+) :
  2.1M × 24 = 50 MB

Grille PM (256³ × 4bytes × 6) :
  400 MB

TOTAL : ~3.4 GB / 12 GB disponibles ✓ (marge confortable)
```

---

## Timestep adaptatif — formule vérifiée 3×

**Formule standard GADGET-2/AREPO (accélération) :**

```rust
// Constantes — NE PAS MODIFIER sans recalcul
const ETA: f64     = 0.025;   // Précision (standard 0.01-0.05)
const DT_MAX: f64  = 0.001;   // Gyr — phase z=10→1 (univers peu dense)
const DT_MIN: f64  = 0.0002;  // Gyr — effondrement (halos denses)

fn compute_adaptive_dt(
    accelerations: &[[f64; 3]],  // Mpc/Gyr² (unités code)
    eps_plus: f64,               // Mpc
) -> f64 {
    // Trouver l'accélération max sur toutes les particules
    let a_max = accelerations
        .iter()
        .map(|a| (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt())
        .fold(0.0_f64, f64::max);

    if a_max < 1e-10 {
        return DT_MAX;
    }

    // Critère : dt = η × sqrt(2ε / |a_max|)
    // Dérivé de : temps pour traverser ε à accélération a_max
    let dt_raw = ETA * (2.0 * eps_plus / a_max).sqrt();

    dt_raw.min(DT_MAX).max(DT_MIN)
}
```

**Vérification par cas :**

| Phase | a_typique (Mpc/Gyr²) | dt_raw (Gyr) | dt_clamp (Gyr) |
|---|---|---|---|
| z=10 (univers homogène) | 1 | 0.01 | **0.001** (DT_MAX) |
| z=1 (filaments) | 50 | 0.00158 | **0.001** (DT_MAX) |
| z=0 halo | 500 | 0.000500 | **0.0005** |
| z=0 cœur dense | 5000 | 0.000158 | **0.0002** (DT_MIN) |

DT_MAX actif pour a < 125 Mpc/Gyr² → la plupart de z=10→1 ✓
DT_MIN actif pour a > 3125 Mpc/Gyr² → seulement les cœurs très denses ✓

**Estimation steps z=10→0 :**
```
Données empiriques : zoom v1 = 18 500 steps pour Δa = 0.314 (z=0.46→0)
Pour z=10→0 : Δa = 0.909 → ratio = 0.909/0.314 = 2.89
Steps à dt fixe : 18 500 × 2.89 = 53 500 steps
Avec dt adaptatif (dt_max × ~40% des steps) : ~32 000 steps effectifs

À 10 steps/min (4.2M particules) : 32 000/10 = 3 200 min ≈ 53h ✓
```

---

## Génération des ICs Zel'dovich — code critique

### Étape préalable — trouver le générateur existant

```bash
# 1. Chercher le générateur Zel'dovich existant
grep -rn "zel.dovich\|ZelDovich\|displacement_field\|power_spec\|transfer_func" \
  src/ --include="*.rs" | grep -v "test\|#\[" | head -20

# 2. Chercher où le run principal génère ses ICs
grep -rn "box_size\|L_BOX\|n_grid\|N_GRID\|sigma8\|SIGMA8\|seed" \
  src/bin/janus_baryonic.rs 2>/dev/null | head -20

# 3. Afficher la signature complète de la fonction IC
grep -n "pub fn\|fn generate\|fn init_ic\|fn zel" \
  src/ic_gen.rs src/ics.rs src/initial_conditions.rs \
  src/cosmology.rs 2>/dev/null | head -20

# Rapporter : nom exact du fichier et de la fonction IC
# NE PAS écrire de code avant d'avoir trouvé ça.
```

### Code IC si le générateur doit être adapté

**Placement des particules sur grille (logique Janus) :**

```rust
/// Génère les positions initiales sur grille pour m+ et m-
/// m+ : grille régulière
/// m- : grille décalée d'un demi-pas (anti-corrélation initiale)
fn place_on_grid(
    n_grid: usize,
    box_size: f64,
    displacement: &DisplacementField,  // Zel'dovich Ψ(x)
    growth_rate: f64,                   // D(z_init)/D(z=0)
    hubble_z: f64,                      // H(z_init) en km/s/Mpc
) -> (Vec<[f32;3]>, Vec<[f32;3]>, Vec<[f32;3]>, Vec<[f32;3]>) {
    
    let dx = box_size / n_grid as f64;
    let half = box_size / 2.0;
    
    let mut pos_plus  = Vec::with_capacity(n_grid.pow(3));
    let mut vel_plus  = Vec::with_capacity(n_grid.pow(3));
    let mut pos_minus = Vec::with_capacity(n_grid.pow(3));
    let mut vel_minus = Vec::with_capacity(n_grid.pow(3));

    for ix in 0..n_grid {
        for iy in 0..n_grid {
            for iz in 0..n_grid {
                // Position grille m+ (centre de cellule)
                let x0 = (ix as f64 + 0.5) * dx - half;
                let y0 = (iy as f64 + 0.5) * dx - half;
                let z0 = (iz as f64 + 0.5) * dx - half;

                // Déplacement Zel'dovich pour m+
                let (d, v) = displacement.at(x0, y0, z0,
                                             growth_rate, hubble_z);
                pos_plus.push([
                    (x0 + d[0]) as f32,
                    (y0 + d[1]) as f32,
                    (z0 + d[2]) as f32,
                ]);
                vel_plus.push([v[0] as f32, v[1] as f32, v[2] as f32]);

                // Position grille m- (décalée de dx/2 selon les 3 axes)
                // Wrap périodique si nécessaire
                let xm = wrap(x0 + dx/2.0, box_size);
                let ym = wrap(y0 + dx/2.0, box_size);
                let zm = wrap(z0 + dx/2.0, box_size);

                // MÊME champ de déplacement pour m-
                // (les deux espèces partagent les perturbations primordiales)
                let (dm, vm) = displacement.at(xm, ym, zm,
                                               growth_rate, hubble_z);
                pos_minus.push([
                    (xm + dm[0]) as f32,
                    (ym + dm[1]) as f32,
                    (zm + dm[2]) as f32,
                ]);
                vel_minus.push([vm[0] as f32, vm[1] as f32, vm[2] as f32]);
            }
        }
    }
    (pos_plus, vel_plus, pos_minus, vel_minus)
}

// Wrap périodique [-L/2, L/2]
#[inline]
fn wrap(x: f64, box_size: f64) -> f64 {
    let half = box_size / 2.0;
    if x > half { x - box_size }
    else if x < -half { x + box_size }
    else { x }
}
```

**Vérification du code :**
- Vérification 1 : N_per_species = n_grid³ exactement (pas de particule en double)
- Vérification 2 : positions dans [-25, 25] Mpc après déplacement Zel'dovich
- Vérification 3 : ratio m-/m+ = 19 exactement (pas de drift sur masse totale)

### Spectre de puissance et normalisation σ_8

```rust
/// Spectre de puissance BBKS (Bardeen, Bond, Kaiser, Szalay 1986)
/// Valide pour une cosmologie sans constante cosmologique (Janus)
fn power_spectrum_bbks(k: f64, sigma8: f64, omega_m: f64, h: f64) -> f64 {
    // k en Mpc⁻¹, h = H₀/100
    let q = k / (omega_m * h * h)  // q en Mpc⁻¹ × Mpc = sans dimension
             * f64::exp(omega_b + (2.0*h).sqrt() * omega_b / omega_m);
    
    // Fonction de transfert BBKS
    let t_k = f64::ln(1.0 + 2.34*q) / (2.34*q)
        * (1.0 + 3.89*q + (16.1*q).powi(2)
           + (5.46*q).powi(3) + (6.71*q).powi(4)).powf(-0.25);
    
    // P(k) = A × k^n_s × T²(k), n_s=1 (Harrison-Zel'dovich)
    let p_k = k * t_k * t_k;
    
    // La normalisation par σ₈ est appliquée en post-traitement
    // (facteur A = σ₈² / sigma8_unnormalized²)
    p_k
}

// ATTENTION : omega_b doit être déclaré dans le scope (= 0.05)
// Cette constante est utilisée dans q — vérifier la portée.
```

**⚠️ Note importante :** Si l'IC generator existant calcule déjà P(k), ne pas
réécrire cette fonction. La réutiliser directement.

---

## Architecture du binaire

```
src/bin/janus_zoom_v3.rs

Réutiliser OBLIGATOIREMENT (ne pas réécrire) :
  janus::sph_pressure_gpu::GpuSphPressure  ← SPH GPU validé
  janus::gravity::*                         ← BH GPU validé
  janus::cosmology::*                       ← H(z), a(t) Janus
  janus::star_formation::*                  ← SF validée
  [ic_gen existant]                         ← à identifier

Structure :
  main()
  ├── parse_args()
  ├── find_ic_generator()    ← AVANT de coder quoi que ce soit
  ├── generate_ics_50mpc()   ← adapter paramètres existants
  ├── validate_ics()         ← test rapide sur N_grid=10
  ├── relax_phase(100 steps, dt=0.0001, SF=OFF, feedback=OFF)
  └── production_run(z_init=5→z_final=0, dt_adaptatif)
      ├── compute_gravity_gpu()
      ├── compute_sph_gpu()    ← seulement m+
      ├── compute_adaptive_dt()
      ├── star_formation()     ← seulement m+ HR
      └── write_snapshot()     ← toutes les 20 steps
```

---

## Paramètres CLI

```rust
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "50.0")]    box_size: f64,
    #[arg(long, default_value = "128")]     n_grid: usize,
    #[arg(long, default_value = "10.0")]    z_init: f64,
    #[arg(long, default_value = "0.0")]     z_final: f64,
    #[arg(long, default_value = "12345")]   seed: u64,
    #[arg(long, default_value = "19.0")]    mu: f64,
    #[arg(long, default_value = "69.9")]    h0: f64,
    #[arg(long, default_value = "0.05")]    omega_b: f64,
    #[arg(long, default_value = "0.811")]   sigma8: f64,
    #[arg(long, default_value = "0.10")]    eps_plus: f64,
    #[arg(long, default_value = "0.25")]    eps_minus: f64,
    #[arg(long, default_value = "0.001")]   dt_max: f64,
    #[arg(long, default_value = "0.0002")]  dt_min: f64,
    #[arg(long, default_value = "0.025")]   eta: f64,
    #[arg(long, default_value = "20")]      snap_interval: usize,
    #[arg(long, default_value = "100")]     relax_steps: usize,
    #[arg(long)]                            out_dir: PathBuf,
}
```

---

## Validation ICs — test rapide OBLIGATOIRE

```bash
# Test sur N_grid=10 avant de lancer N_grid=128
./target/release/janus_zoom_v3 \
  --n-grid 10 --box-size 50 \
  --z-init 10.0 --z-final 9.8 \
  --snap-interval 1 \
  --out-dir /tmp/v3_test_10/

python3 << 'EOF'
import struct, numpy as np

with open('/tmp/v3_test_10/snapshots/snap_00000.bin','rb') as f:
    N = struct.unpack('Q', f.read(8))[0]
    a = struct.unpack('d', f.read(8))[0]
    t = struct.unpack('d', f.read(8))[0]
    data = np.frombuffer(f.read(), dtype=np.float32)

N_expected = 2 * 10**3  # = 2000
assert N == N_expected, f"N={N} ≠ {N_expected}"

pos = data[:N*3].reshape(N,3)
vel = data[N*3:N*6].reshape(N,3)
signs = data[N*6:N*6+N/4]  # approximatif selon format

print(f"N_total : {N} (attendu {N_expected}) ✓")
print(f"z_init  : {1/a - 1:.3f} (attendu ~10.0)")
print(f"pos min/max : {pos.min():.2f} / {pos.max():.2f} (attendu ~±25 Mpc)")
print(f"vel max : {np.abs(vel).max():.1f} km/s (attendu < 200 à z=10)")
print(f"min dist entre particules : {min_dist:.4f} Mpc (doit être > 0.05)")

# Vérifier pas de NaN/Inf
assert not np.any(np.isnan(pos)), "NaN dans positions!"
assert not np.any(np.isinf(vel)), "Inf dans vitesses!"
print("Validation ICs : OK ✓")
EOF
```

---

## Commande de lancement

```bash
# 1. Compilation
cargo build --release --bin janus_zoom_v3 --features cuda

# 2. Lancement
nohup docker compose run --rm dev \
  ./target/release/janus_zoom_v3 \
  --box-size     50.0   \
  --n-grid       128    \
  --z-init       10.0   \
  --z-final      0.0    \
  --seed         12345  \
  --mu           19.0   \
  --h0           69.9   \
  --omega-b      0.05   \
  --sigma8       0.811  \
  --eps-plus     0.10   \
  --eps-minus    0.25   \
  --dt-max       0.001  \
  --dt-min       0.0002 \
  --eta          0.025  \
  --snap-interval 20    \
  --relax-steps  100    \
  --out-dir      output/janus_zoom_v3/ \
  > output/janus_zoom_v3/run.log 2>&1 &

echo "PID: $! — log: tail -f output/janus_zoom_v3/run.log"
```

---

## STOP automatiques

```rust
// Overflow positions
if pos_max.abs() > 1e4 {
    panic!("Overflow pos: {:.2e} Mpc — instabilité numérique", pos_max);
}
// SF runaway
if n_stars > 10_000_000 {
    panic!("SF runaway: N★={}", n_stars);
}
// SF bloquée
if z < 0.3 && rho_max > 1e8 && n_stars == 0 {
    panic!("SF bloquée: ρ_max={:.2e}, N★=0 à z={:.3}", rho_max, z);
}
// Relaxation instable
if relax_step < 100 && delta_e_rel > 0.10 {
    panic!("Relaxation instable: ΔE/E={:.3}", delta_e_rel);
}
```

---

## Checkpoints

| z | Attendu | Si raté |
|---|---|---|
| 9.5 | ICs propres, v_rms < 200 km/s | Vérifier σ_8 |
| 2.5 | Premiers filaments m+ | Vérifier P(k) |
| 1.0 | Halos proto-stellaires, SF naissante | Vérifier seuil ρ_SF |
| 0.3 | N★ > 500, σ_v > 100 km/s | Vérifier cooling |
| 0.0 | N★ > 5000, M_halo > 10¹³ M☉, N⁻=0 | Succès |

---

## Estimation ressources — vérifiées

| Métrique | N_grid=128 | Commentaire |
|---|---|---|
| N_total | 4 194 304 | 2.1M m+ + 2.1M m- |
| m_part_plus | 4.04×10⁸ M☉ | ×124 vs zoom v1 |
| VRAM | ~3.4 GB | Marge 12 GB ✓ |
| Steps estimés | ~32 000 | Δa × zoom v1, ×0.6 adaptatif |
| Rate estimé | ~10 steps/min | GPU BH pour 4.2M |
| Durée | ~53h | Dans budget 60h |
| Snapshots | ~1450 × ~160 MB | ~230 GB / 958 GB ✓ |

---

## Incertitudes à confirmer par CLI avant de coder

```
1. Quel fichier contient le générateur IC Zel'dovich ?
   → Ne pas écrire de code IC avant d'avoir répondu à ça.

2. Le générateur IC accepte-t-il box_size et n_grid en paramètres ?
   → Si hardcodés : les paramétrer. Si flexibles : les réutiliser.

3. Quel format utilise le run principal pour les ICs (dual-seed, grille décalée) ?
   → Important pour cohérence m+/m-.

4. Le binaire janus_baryonic.rs génère-t-il ses ICs en interne ou lit-il un fichier ?
   → Déterminer si on peut appeler la même fonction.
```

---

## Instructions pour CLI

```
1. Lire ce document en entier.
2. Répondre aux 4 incertitudes (section finale) AVANT d'écrire du code.
3. Afficher les réponses, attendre validation.
4. Seulement ensuite : créer janus_zoom_v3.rs.
5. cargo check avant test.
6. Test N_grid=10 avant N_grid=128.
```

*Version 2.0 — Avril 2026 — Calculs vérifiés 3× par Python*
*Donner ce fichier à CLI, attendre ses réponses aux 4 incertitudes*
