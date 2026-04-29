# Décision GO/NO-GO — Run TreePM Janus 1M particules z=10→0

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Commit head** : c752d87
**CLI** : Claude Opus 4.7 (1M context), exécution autonome plan PLAN_CLI_TREEPM_JANUS.md

---

## Résumé exécutif

**Verdict** : **GO-AVEC-RÉSERVES**

Toutes les fondations CPU des phases 1-8 sont validées avec **81 tests passants / 0 failures / 3 ignored** (les 3 ignored sont des cas pathologiques documentés). Le port physique JPP→TreePM avec corrections GrGadget (gradient ord 4, CIC W⁻², Laplacien continu) et conventions Janus complètes (drift/kick cosmo per-particle, cross-coupling φ et c̄²) est **prêt pour intégration GPU**.

**Réserve principale** : l'intégration des corrections Phase 2/3/5 dans le kernel CUDA `nbody_gpu_twopass.rs` (étape Phase 5 GPU déférée) reste à faire avant le run 1M. Le pipeline CPU intégré valide la physique mais n'est pas performant pour 1M × 10000 steps.

---

## Tableaux de résultats par phase

### Phase 1 — Audit (read-only)

| Élément | Statut |
|---|---|
| Module TreePM existant (1865 LOC, 7 fichiers) | Production-ready Newton |
| Convention vélocité | Peebles peculiar v_pec = a·dx_co/dt ✅ |
| Gradient | Ordre 2 → à upgrade |
| CIC déconvolution | ABSENT → à ajouter |
| Laplacien forme | Continue OK ✅ |
| GPU caps RTX 3060 sm_86 | 12 GB VRAM, suffit jusqu'à 10M |

### Phase 2 — Corrections numériques

| Test | Résultat | Tolérance | Status |
|---|---|---|---|
| Gradient ord 4 convergence (n=32→64→128) | ratio ≈ 16 | > 12 | ✅ |
| Gradient constant field zero | max_err < 1e-13 | < 1e-13 | ✅ |
| CIC window inv at Nyquist (3D corner) | (π/2)⁶ | exact | ✅ |
| CIC window symmetric (sign flip) | invariant | exact | ✅ |
| Laplacien k² known modes | exact | 1e-14 | ✅ |
| Poisson sinusoidal source (n=64, n_mode=4) | err 2.6% | < 5% | ✅ |

### Phase 3 — Optimisations GPU PhotoNs (CPU-side foundations)

| Module | Tests | Status |
|---|---|---|
| gpu_layout (SoA ParticleArrays) | roundtrip, memory 53 B/particle | ✅ |
| task_list (P2PTask, GroupIdx) | sort + group_idx consistency | ✅ |
| truncation_table (erfc/exp 512 pts) | T(0)=1, T(1)=0.572, T(2)=0.046, monotonic | ✅ |
| Kernel CUDA optimisé | DÉFÉRÉ Phase 5 GPU integration | ⚠️ |

### Phase 4 — Paramètres canoniques

| Paramètre | Valeur | Source |
|---|---|---|
| r_s = 1.2 × Δg | 1.17 Mpc @ N_pm=256, L=250 | PhotoNs §2 |
| r_cut = 6 × Δg | 5.86 Mpc | PhotoNs §2 |
| θ = 0.5 | | GreeM (z<10) |
| N_leaf = 10, N_crit = 300 | | GreeM, PhotoNs |
| recommended_pm_grid(N) | 100K→64, 1M→128, 10M→256 | Plan §1.4 |

Sweep paramétrique (§4.2) déféré : nécessite Phase 5 GPU intégration pour timing significatif.

### Phase 5 — Extension Janus

| Convention | Source | Statut |
|---|---|---|
| Cross-coupling cross_minus_plus = c̄²·φ⁻¹·repulsion_scale | Petit 2024 | ✅ implémenté JanusCoupling |
| Cross-coupling cross_plus_minus = φ·repulsion_scale | Petit 2024 | ✅ |
| drift_cosmo: pos += vel·dt/a_eff per-particle | Peebles | ✅ |
| kick_cosmo: vel += (acc/a_eff² - h_eff·vel)·dt | Peebles | ✅ |
| Convention sign u8 (snap) ↔ i8 (TreePM) ↔ i32 (BH) | Audit 05 | ✅ documenté |
| Force factor m+/m+ et m-/m- = +1 | BH existant | ✅ |
| Force factor m+/m- = -cross_plus_minus | BH existant | ✅ |
| Tests JanusCoupling factors | 12 tests | ✅ |

### Phase 6 — Validation Newton

| Test | Résultat | Tolérance | Status |
|---|---|---|---|
| PM force single source (non-grid-aligned) | 54%/30%/12% err à 6/12/24 dg | <2× à 12 dg | ✅ |
| grad4 vs grad2 consistency | both attractive, < 70% diff | < 100% | ✅ |
| Sinusoidal Poisson | 2.6% err | < 5% | ✅ |
| Test P(k) Newton vs PP | DÉFÉRÉ (full pipeline GPU) | | ⚠️ |

### Phase 7 — Validation Janus

| Test | Résultat | Status |
|---|---|---|
| 2 m+ attractive (1 step) | vel signs corrects | ✅ |
| m+/m- répulsif (Janus) | vel signs corrects | ✅ |
| 2 m- attractive (Petit p.36) | vel signs corrects | ✅ |
| 20 steps no runaway | v_max bounded | ✅ |
| Mini-run 100K z=10→2 (§7.1) | DÉFÉRÉ (full GPU pipeline) | ⚠️ |

### Phase 8 — Performance

| Bench | Résultat | Cible | Status |
|---|---|---|---|
| N=100 CPU integrated step | 4 ms/step | < 100 ms | ✅ |
| N=1K CPU integrated step | 59 ms/step | < 1 s | ✅ |
| Memory 1M particles SoA | 50.5 MB | < 60 MB | ✅ |
| Memory PmGrid N_pm=256 | 512 MB | < 1 GB | ✅ |
| Memory PmGrid N_pm=512 | 4.3 GB | tight on 12 GB | ⚠️ |
| GPU scaling 100K/1M/10M | DÉFÉRÉ (Phase 5 GPU) | | ⚠️ |
| GPU profiling nvprof | DÉFÉRÉ | | ⚠️ |

---

## Liste des FLAGS ouverts

1. **Phase 5 GPU integration manquante** : les corrections Phase 2/3/5 (gradient ord 4, CIC W⁻², Janus cross-coupling) sont dans les modules CPU `src/treepm/{gradient,cic_correction,janus}.rs` mais PAS encore portées dans le kernel CUDA `src/nbody_gpu_twopass.rs`. Le pipeline GPU existant utilise toujours gradient ord 2 + cic_gather sans déconvolution + factors 1.0 (pas de Janus complet).

2. **Tests P(k) Newton vs PP (§6.2)** et **mini-run 100K (§7.1)** déférés : nécessitent le pipeline GPU intégré.

3. **Sweep paramétrique (§4.2)** : différé pour la même raison.

4. **Convention g_constant** : pmgrid::solve_poisson attend `g_constant = G_phys / V_cell`. Le caller (jonction GPU) doit appliquer ce scaling. C'est documenté mais à vérifier dans le code GPU.

5. **PmGrid N_pm=512 mémoire tight** : 4 grids × 8 B × 512³ = 4.3 GB sur 12 GB. Pour 10M particules, marge restante ~7 GB pour BVH, particles, FFT workspace. Surveillance OOM nécessaire.

---

## Recommandation pour le run 1M

### Configuration recommandée (POST Phase 5 GPU integration)

```toml
n_particles = 1_000_000
n_plus = 50_000        # 5% de m+
n_minus = 950_000      # 95% de m- (ratio μ=19)
box_size_mpc = 250.0
n_pm = 128             # cell size = 1.95 Mpc
mu = 19.0
eta = 1.045
z_init = 10.0
z_target = 0.0
n_steps = 15000
dynamic_vsl = true
softening_plus = 0.05
softening_minus = 0.25
output_freq_steps = 500
checkpoint_freq_steps = 2000

# TreePM canonical (PhotoNs §2)
split_scale_factor = 1.2     # r_s = 1.2 × Δg = 2.34 Mpc
cutoff_factor = 6.0          # r_cut = 6 × Δg = 11.7 Mpc
opening_angle = 0.5          # GreeM
```

### Critères de succès du run 1M

- Run termine à z=0 sans crash
- **P(|k|) 3D radial sans pic isolé** à |k|=4, 8, 16, 32 (ratios < 1.5) — critère anti-résonance octree
- Corr(δ⁺, δ⁻) finale ∈ [-0.10, -0.05]
- σ8 finale ∈ [0.65, 0.75]
- t₀ ∈ [14, 17] Gyr
- v_rms+ et v_rms- < 3000 km/s à z=0

### Travaux restants avant lancement run 1M

1. **Port GPU Phase 5** (1-3 jours) :
   - Modifier `src/nbody_gpu_twopass.rs::cic_gather` pour utiliser gradient ord 4 (porter `treepm::gradient::grad4_*`)
   - Ajouter CIC W⁻² deconvolution dans `solve_device` cuFFT (ou pre-multiply ρ en host avant FFT)
   - Implémenter `step_treepm_gpu_cosmo(dt, a_plus, a_minus, h_plus, h_minus, coupling)` :
     * Remplacer `drift_f32`/`kick_f32` par versions cosmo (per-particle 1/a, 1/a², h)
     * Modifier `cic_gather` et `forces_treepm_short_range` pour appliquer cross-coupling Janus
   - Wrapper `janus_jpp_production.rs` pour utiliser TwoPass.step_treepm_gpu_cosmo

2. **Mini-run validation 500 steps** (1 jour) :
   - Mêmes IC que run squared archivé
   - Critère : P(|k|) 3D radial sans pic L/8 ou L/16 (ratios < 1.2)
   - Frame visuel sans grille à z=4

3. **Full prod 1M ou 10M** (50-65h GPU) :
   - Après validation OK
   - Monitoring continu directional Pk

### ETA total

| Étape | Durée |
|---|---|
| Phase 5 GPU port | 1-3 jours |
| Mini-run 500 validation | 1 jour |
| Full prod 1M | 1 jour wall (~24h GPU) |
| Full prod 10M | 2-3 jours wall |
| **Total** | **~1 semaine pour résultats préprint** |

---

## Statut final Phase 9

| Phase | Status | Tests | Commit |
|---|---|---|---|
| 1 — Audit | ✅ Complete | 4 livrables | treepm-phase1 |
| 2 — Corrections numériques | ✅ Complete | 30/30 | treepm-phase2 |
| 3 — GPU optims (CPU-side) | ✅ Complete | 49/49 | treepm-phase3 |
| 4 — Paramètres | ✅ Complete | 59/59 | treepm-phase4 |
| 5 — Extension Janus | ✅ Complete (CPU) | 71/71 | treepm-phase5 |
| 6 — Validation Newton | ✅ Complete | 73/73 | treepm-phase6 |
| 7 — Validation Janus | ✅ Complete (small-N) | 78/78 | treepm-phase7 |
| 8 — Benchmarks | ✅ Complete (CPU) | 81/81 | treepm-phase8 |
| 9 — Décision | ✅ This document | | (next commit) |

**Total** : **81 tests passants / 0 failed / 3 ignored**, 9 commits sur `feat/treepm-jpp-port`.

---

## Recommandation finale

**GO-AVEC-RÉSERVES** — fondations physiques validées, port GPU restant. Pas de blocage critique identifié. Le travail peut continuer immédiatement avec :

1. Phase 5 GPU integration (port modules CPU vers kernels CUDA)
2. Mini-run 500 steps validation
3. Full prod 1M

Aucun bloqueur scientifique majeur. Les 5 FLAGS ouverts sont tous des dépendances séquentielles claires sur Phase 5 GPU.

**CLI s'arrête ici comme demandé par le plan §9.3.** Pas de lancement automatique du run 1M sans validation humaine.
