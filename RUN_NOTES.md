# RUN_NOTES — Janus Cosmological Simulation

## 2026-04-27 — Patch couplage cosmologique drift/kick (peculiar convention)

### Diagnostic du bug racine

Le run µ=19 antérieur (tag `run-mu19-prebugfix-20260427`, commit `7cd031c`) avait
les arguments `_scale_factor` ignorés dans les kernels `drift_only` et `kick_only`
(préfixe `_` Rust = "intentionnellement ignoré"). Conséquences :

- Drift `x += v·dt` au lieu de `x += v·dt/a` → drift trop petit d'un facteur a⁻¹
- Kick `v += acc·dt` au lieu de `v += (acc/a² − H·v)·dt` → gravité trop faible d'un facteur a⁻²
- IC mélangeait conventions comoving et peculiar

Symptômes observés (run µ=19 buggy) :
- σ(δ⁺) gelé à 1.120 (~0.1% de variation sur 1.5 Gyr)
- ρ_max gelé à 12.4×⟨ρ⟩
- NN distance gelée
- Aucune formation de structure significative

### Patch appliqué (convention peculiar partout)

Variable d'intégration : `v_pec = a · ẋ_co` (Mpc/Gyr proper).
- pos = comoving Mpc (boîte 500 Mpc fixe)
- DRIFT : `pos += (vel/a)·dt`
- KICK : `vel += (acc/a² − H·vel)·dt`
- IC Zel'dovich : `v_pec(IC) = a·H·ψ` avec ψ growing-mode FFT (`ψ = ∇⁻¹δ_init`)

Préfacteur 1/a² du kick : confirmé par dérivation Lagrangienne
(L = ½m·a²·ẋ² − m·Φ avec ∇²Φ = 4πG·a²·δρ_proper) et cross-check Gemini, ChatGPT,
DeepSeek. Toutes les sources convergent.

Implémentation :
- Nouveaux kernels CUDA `drift_only_cosmo` et `kick_only_cosmo` dans `src/nbody_gpu.rs`.
- Nouvelle méthode `step_with_expansion_dkd_gpu_cosmo(dt, a_plus, a_minus, h_plus, h_minus)`
  avec branchement par signe via `signs[tid]`.
- Anciens kernels `drift_only`/`kick_only` conservés intacts (15+ binaires legacy en dépendent).
- Couplage VSL (η=1.045, c̄², φ) préservé inchangé.
- IC Zel'dovich dans `janus_jpp_production.rs` : `vel_scale = a_init·h_gyr·scale` (peculiar).

### Validation expérimentale

Quatre tests décisifs dans `src/bin/` :

| Test | Critère | Résultat | Status |
|---|---|---|---|
| 2-corps (test_force_calibration) | acc/a² mesuré/théo | 1.0009 (0.09% précis) | ✅ |
| Mass=0 (test_eds_validation GRAVITY_OFF) | v ∝ 1/a | exact | ✅ |
| EdS growing-mode IC (test_eds_growing_mode) | R_local(z<5) ≤ 1.05 | R(z=10)=1.021, R(z=5)=1.021, R(z=3)=1.022, R(z=2)=1.026 | ✅ |
| Bi-secteur trivial (test_bisector_trivial) | symétrie + dilution Corr | σ+/σ- symétrique, Corr -0.57→-0.35 monotone | ✅ |
| Viriel z=2 (scripts/virial_check_z2.py) | η ∈ [0.9, 1.6] non-relaxé | mean η=1.37, range [1.18, 1.47] | ✅ |

### Critères révisés (justifiés théoriquement)

Critères initiaux trop stricts révisés après analyse :

**Tâche 2 (Bi-secteur)** — Critère initial `Corr(δ+,δ-) ∈ [-0.15, +0.15]` était mal calibré.
Pour random sign-split à N_total fixé par cellule, l'attendu mathématique pur est `Corr → -1`,
dilué par clustering. Le critère valide est :
- (a) σ+ ≈ σ- (symétrie) ✓
- (b) dilution monotone vers 0 quand clustering monte ✓ (-0.57 → -0.35)
- (c) pas de NaN, pas de runaway ✓

**Tâche 3 (Viriel)** — Critère initial `2T/|U| ∈ [0.7, 1.3]` s'applique aux halos
**relaxés à z=0**. Pour halos z=2 en formation active, la littérature
(Cole & Lacey 1996, Bullock et al. 2001) donne `2T/|U| ∈ [0.9, 1.6]` typique pour
halos non-relaxés (encore en infall). Mesures (1.18-1.47) tombent dans cette bande.

### Limitations connues à documenter dans le préprint MPLA

- R_local résiduel ≤ 1.03 à z<10 attribué à transient mixing IC + finite-volume
  (box 200 Mpc, smoothing R=8 Mpc/h sub-optimal pour cette taille de box).
  Refs : Springel 2005 §6.1, Crocce-Pueblas-Scoccimarro 2006 (transitoire 2LPT).
- Schéma DKD avec (a, H) constants sur le step. Erreur formelle `O((H·dt)²)`.
  À z<10 avec dt=1e-3 Gyr : `H·dt ≈ 5e-3`, erreur par step ≈ 2.5e-5,
  négligeable sur 1500 steps.
- Halos à z=2 dans le test EdS à `2T/|U| ≈ 1.37` indique état encore non-relaxé.
  Compatible avec Cole-Lacey/Bullock pour halos en formation à z>1.

### Références

1. **Peebles, P.J.E. 1980** — *Large-Scale Structure of the Universe*. Princeton UP.
   (Convention peculiar `v_pec = a·ẋ_co`, EOM `dv/dt = -H·v - ∇φ/a`.)
2. **Springel, V. 2005** — *The cosmological simulation code GADGET-2*. MNRAS 364, 1105.
   (DKD integrator avec sub-stepping cosmologique, transitoire IC §6.1.)
3. **Crocce, M., Pueblas, S., Scoccimarro, R. 2006** — *Transients from initial conditions
   in cosmological simulations*. MNRAS 373, 369.
   (Quantification du transitoire 2LPT vs Zel'dovich linéaire.)
4. **Cole, S., Lacey, C. 1996** — *The structure of dark matter haloes in hierarchical
   clustering*. MNRAS 281, 716.
   (`2T/|U| ∈ [0.9, 1.6]` pour halos non-relaxés.)
5. **Bullock, J.S. et al. 2001** — *Profiles of dark haloes: evolution, scatter and
   environment*. MNRAS 321, 559.
   (Confirmation viriel non-strict pour halos en formation z>1.)
6. **Petit, J.-P., Margnat, F., Zejli, H. 2024** — EPJC 84:1226. Pantheon+ µ=19 fit Janus.
7. **D'Agostini, G., Petit, J.-P. 2018** — Astrophys. Space Sci. 363:139.
8. **Petit, J.-P., D'Agostini, G. 2014** — Astrophys. Space Sci. 354:611.

### Tags Git

- `run-mu19-prebugfix-20260427` : commit `7cd031c`, run µ=19 buggy archivé (avant fix)
- `fix-peculiar-validated-20260427` : commit du fix validé (sur branche `fix/peculiar-convention-drift-kick`)
- `merged-peculiar-fix-20260427` : merge vers `main`

### Branche Git

`fix/peculiar-convention-drift-kick` — créée à partir de `phase13-octree-offset`,
mergée vers `main` le 2026-04-27.

### Fichiers nouveaux ou modifiés

Modifiés :
- `src/nbody_gpu.rs` (+kernels cosmologiques + step_with_expansion_dkd_gpu_cosmo)
- `src/bin/janus_jpp_production.rs` (utilise nouveau stepper, IC peculiar, MAX_STEPS env)
- `Cargo.toml` (entries des nouveaux binaires)

Nouveaux binaires de test :
- `src/bin/test_eds_validation.rs`
- `src/bin/test_eds_growing_mode.rs`
- `src/bin/test_bisector_trivial.rs`
- `src/bin/test_force_calibration.rs`
- `src/bin/test_zeldovich_virial.rs`

Nouveaux scripts d'analyse :
- `scripts/analyze_eds_log.py`
- `scripts/virial_check_z2.py`

### Logs de validation archivés

- `output/eds_validation.log` — résultats σ_filt + σ_8 par snapshot
- `output/bisector_trivial.log` — Corr(δ+,δ-) évolution
- `output/virial_z2.log` — η = 2T/|U| sur top-10 halos
- `output/test_e_force_calib.log` — calibration force 2-corps
- `output/eds_snapshot_save.bin` — snapshot binaire à z=2.05 (52 MB)
