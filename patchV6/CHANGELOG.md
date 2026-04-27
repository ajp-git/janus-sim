# Fix IC v6 — positions aléatoires pures + interpolation CIC

## Contexte

Le run v6 lancé avec `janus_adaptive_zoom.rs` patché (versions précédentes) a
produit à step 1860 (z=1.69) un **artefact de filaments en croix** alignés
sur les axes, identique à celui du test split (seuils /100).

Diagnostic dans l'historique (février 2026, chat « Janus et la transition
vers la 3D ») : avec une grille régulière d'ICs, même avec jitter ou amplitude
réduite, le motif de grille persiste à z intermédiaire car le champ Fourier ψ
est calculé sur cette même grille → résonance.

## Correction appliquée dans ce fichier

**Anciennes ICs (v6 précédent, grille + signes shuffle) :**
```rust
// Pour chaque (ix, iy, iz) de la grille:
x0 = (ix + 0.5) * spacing - half_box  // position grille
dx = psi_x[ix,iy,iz] * scale          // déplacement indexé pareil
position = (x0 + dx) % L
```

**Nouvelles ICs (v6 ic_fix, random + CIC) :**
```rust
// Pour chaque particule i:
x0 = random_uniform(-L/2, L/2)         // position aléatoire pure
psi_xi = CIC_interp(psi_x, x0, y0, z0) * scale  // ψ interpolé à la position random
position = (x0 + psi_xi) % L
velocity = psi_xi * vel_scale
```

C'est **exactement la méthode validée pour le run `40M_v3`** (préprint JPP).

## Autres patches déjà présents (rappel)

- `compute_particle_masses` : convention Janus m+ = m- = Ω_b ρ_crit L³ (1+μ)/N
- `generate_zeldovich_ics` prend `mu` en paramètre, assignation de signes shuffled N+ = N/(1+μ)
- `adaptive_split_check_with_thresholds` préserve l'énergie interne à travers splits
- `save_snapshot` transmet n_stars et sfr
- Seuils split : RESTAURÉS à la valeur production [1e4, 3e4, 1e5, ...] × ρ_mean_plus

## Commande relance

```bash
cargo build --release --bin janus_adaptive_zoom --features cuda

./target/release/janus_adaptive_zoom \
  --n-grid 215 --l-box 500 --z-init 10.0 --z-final 0.0 \
  --snap-interval 20 --steps-check 50 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir /app/output/janus_adaptive_v6_icfix \
  --run-label v6_icfix \
  2>&1 | tee /app/output/janus_adaptive_v6_icfix/run.log
```

## Critères à vérifier rapidement (step 280, z ≈ 6)

**Frame** (faire demande du render) :
- **PLUS de motif de grille visible** dans les zooms m+/m− ±50 Mpc
- Distribution diffuse, granulaire, PAS rectiligne

**time_series.csv** :
- v_rms plausible : quelques milliers de km/s à z=6 (comme le test N_grid=50)
- ρ_max cohérent avec Ω_tot=1 : quelques × ρ_crit à z=6

Si ces deux critères sont OK à step 280, continuer vers la production complète.
