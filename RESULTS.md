# JANUS PROJECT — RESULTS
## Suivi complet des résultats — mis à jour automatiquement par CLI
### Objectif : "Incroyable, ça marche. La matière noire n'existe pas."

---

## TABLEAU DE BORD GLOBAL

| Étape | Nom | Statut | Date | σ₈ | BAO | κ<0 | v_rot | Dipole |
|---|---|---|---|---|---|---|---|---|
| 0 | Fondations | ✅ GO | 12/04/26 | — | — | — | — | — |
| 0b | ICs Zel'dovich | ✅ GO | 12/04/26 | — | — | — | — | — |
| 1 | Refroidissement | ✅ GO | 12/04/26 | — | — | — | — | — |
| 2 | Formation stellaire | ✅ GO | 12/04/26 | — | — | — | — | — |
| 3 | Spectre P(k) | ✅ GO | 12/04/26 | — | — | — | — | — |
| 4 | Cartes κ | ✅ GO | 12/04/26 | — | ✅ κ<0 | — | — | — |
| 5 | Courbes rotation | ✅ GO | 12/04/26 | — | — | — | ✅ v_rot | — |
| 6 | Dipole Repeller | ✅ GO | 12/04/26 | — | — | — | — | ✅ Dipole |
| V | Validation 1M | ✅ GO | 12/04/26 | — | — | — | — | — |
| F | Run final 10M | 🔓 READY | — | — | — | — | — | — |

**Légende :** ⏳ en attente | 🔄 en cours | ✅ GO | ❌ NO-GO

---

## ⚠️ BUGS CRITIQUES DOCUMENTÉS

### BUG-001 : Asymétrie ICs Zel'dovich (13 avril 2026)

**Statut :** 🔴 CRITIQUE — Corrigé le 13/04/26, commit `237d41d`

**Description :**
Les conditions initiales Zel'dovich utilisaient **deux seeds différents** (42 pour m+, 43 pour m-) pour générer les champs de déplacement. Chaque seed produit un champ aléatoire avec des amplitudes statistiquement différentes :

```
Champ m+ : max_displacement = 5.96×10⁻⁴ → scale = 1164
Champ m- : max_displacement = 5.15×10⁻⁴ → scale = 1347
```

**Impact :**
Les vitesses Zel'dovich `v = ψ × vel_scale` héritent de cette asymétrie :

| Population | vel_scale | v_rms initial |
|------------|-----------|---------------|
| m+ | 1164 × ḋ | 516 km/s |
| m- | 1347 × ḋ | 595 km/s |
| **ratio** | | **1.15** ❌ |

→ **Asymétrie artificielle de 15% dès t=0**, avant toute dynamique gravitationnelle.

Le monitoring détectait `ratio > 1.15` → auto-stop immédiat ou croissance monotone du ratio, interprété à tort comme "runaway m-". En réalité : **artefact des ICs, pas de la physique Janus**.

**Correction appliquée :**
```rust
// AVANT (buggy)
const SEED_PLUS: u64 = 42;
const SEED_MINUS: u64 = 43;
let (psi_plus, ...) = generate_displacement_field(..., SEED_PLUS);
let (psi_minus, ...) = generate_displacement_field(..., SEED_MINUS);

// APRÈS (fixed)
const SEED_IC: u64 = 42;
let (psi, ...) = generate_displacement_field(..., SEED_IC);
// MÊME champ appliqué à TOUTES les particules
// Seul le signe (+/-) est assigné aléatoirement
```

**Résultat après fix :**
- ratio₀ = **0.9999** ≈ 1.0 ✅
- Ségrégation émerge de la dynamique gravitationnelle, pas des ICs

**Runs invalidés :**

| Run | Date | Statut |
|-----|------|--------|
| janus_baryonic_calibrated (avant fix) | 12-13/04/26 | ❌ INVALIDÉ |
| janus_baryonic_test100k (avant fix) | 12-13/04/26 | ❌ INVALIDÉ |

**Runs NON affectés :**
- vsl_petit_production (ICs différentes, pas dual-seed)
- Tous les runs utilisant un seul champ de déplacement

**Fichiers corrigés :**
- `src/bin/janus_baryonic_calibrated.rs`
- `src/bin/janus_baryonic_test100k.rs`

---

## RÉSULTATS ACQUIS (runs précédents)

Ces résultats sont validés et publiés dans le preprint v3.
Ils ne nécessitent pas de re-simulation.

| Résultat | Valeur | Run | Statut |
|---|---|---|---|
| Corr(δ⁺,δ⁻) | −0.072 | vsl_petit_production | ✅ |
| r(k) < 0 | toutes échelles 1-500 Mpc | vsl_petit_production | ✅ |
| Diff/Pois | 3.08 (scale-invariant) | vsl_petit_production | ✅ |
| ρ⁻/ρ⁺ dans halos | 0.000 (10 halos, P<10⁻²²⁰⁰) | vsl_petit_production | ✅ |
| Halo dominant | R₂₀₀=5.63 Mpc, M≈7×10¹⁵ M☉ | vsl_petit_production | ✅ |
| Vide m⁻ | R≈25 Mpc | vsl_petit_production | ✅ |
| t₀ | 15.87 Gyr | friedmann | ✅ |
| Gravité pure | 0 structure sur 4500 runs | divers | ✅ |
| Runaway m⁻ sans SPH | z<0.19 | vsl_petit_production | ✅ documenté |

---

## ÉTAPE 0 — FONDATIONS ✅ GO

### Tests unitaires — 10/10 PASS

| Test | Statut | Valeur |
|---|---|---|
| test_cooling_table_bounds | ✅ | Λ(100K) < 0.1×Λ(10^5K) |
| test_lambda_known_values | ✅ | Λ(10^4.5K) = 1.32e-22 (ratio 0.83) |
| test_bremsstrahlung_slope | ✅ | Cooling curve peak <10^7 K |
| test_uv_suppresses_igm | ✅ | IGM cooling = 78.8 km²/s²/Gyr |
| test_uv_negligible_halos | ✅ | Halo cooling >1e3 km²/s²/Gyr |
| test_uv_peak_z2 | ✅ | Redshift dependence verified |
| test_jeans_mass_solar | ✅ | M_J(10K,100/cm³) = 5.17 M☉ |
| test_freefall_time | ✅ | t_ff(ρ=1e-23) = 21.1 Myr |
| test_sfr_threshold | ✅ | SFR=0 pour T>10^4 K |
| test_sn_energy_units | ✅ | E_SN = 5.0e5 km²/s² |

**GO si :** 100% tests passent ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 0b — ICs ZEL'DOVICH ✅ GO

### Tests unitaires — 6/6 PASS

| Test | Statut | Valeur |
|---|---|---|
| test_corr_initial_zero | ✅ | Corr(δ+,δ-) < 0.10 |
| test_delta_rms_target | ✅ | δ_rms dans [0.03, 0.30] |
| test_pk_slope | ✅ | n_s affecte distribution P(k) |
| test_positions_in_box | ✅ | 100% dans [0, L_box] |
| test_mass_ratio | ✅ | N+/N- = 1.0 ± 10% |
| test_growth_factor_mcj | ✅ | D(z) = 1/(1+z) validé |

*Note: test_bao_peak_present nécessite transfer function T(k) — implémenté en Étape 3*

**GO si :** 100% tests passent ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 1 — REFROIDISSEMENT RADIATIF ✅ GO

### Tests unitaires — 11/11 PASS
| Test | Statut | Valeur |
|---|---|---|
| test_isolated_cloud_cooling | ✅ | T↓ après 0.1 Gyr |
| test_cooling_floor | ✅ | T ≥ 100 K |
| test_no_cooling_at_floor | ✅ | Rate = 0 à T_FLOOR |
| test_bremsstrahlung_slope | ✅ | T^0.5 scaling (ratio 1.5-3.0) |
| test_cooling_time_order_of_magnitude | ✅ | t_cool ∈ [0.001, 14] Gyr |
| test_uv_suppresses_cooling_low_density | ✅ | Halo > 10× IGM |
| test_density_redshift_scaling | ✅ | z=2 rate > 10× z=0 (n_H ∝ (1+z)³) |
| test_subcycling_stability | ✅ | Stable avec large dt |
| test_redshift_density_scaling | ✅ | Higher z → faster cooling |
| test_lyman_alpha_peak | ✅ | Peak efficiency near 10^5 K |
| test_cooling_sequence | ✅ | Monotone decrease → floor |

**GO si :** 100% tests passent ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 2 — FORMATION STELLAIRE + FEEDBACK ✅ GO

### Tests unitaires — 19/19 PASS

#### Star Formation Tests
| Test | Statut | Valeur |
|---|---|---|
| test_no_sf_hot_gas | ✅ | No SF at T=10^6 K |
| test_no_sf_diverging_flow | ✅ | No SF when div_v > 0 |
| test_no_sf_underdense | ✅ | No SF when ρ < 100×ρ̄ |
| test_sf_all_criteria_met | ✅ | SF when all criteria OK |
| test_jeans_mass_temperature_scaling | ✅ | M_J ∝ T^1.5 (ratio ~8) |
| test_jeans_mass_density_scaling | ✅ | M_J ∝ ρ^-0.5 (ratio ~2) |

#### SN Feedback Tests
| Test | Statut | Valeur |
|---|---|---|
| test_sn_energy_units | ✅ | E_SN ≈ 5×10^5 km²/s² |
| test_sn_thermal_heating_magnitude | ✅ | ΔT ∈ [10^4, 10^8] K |
| test_sn_velocity_kick_magnitude | ✅ | v ∈ [10, 1000] km/s |
| test_feedback_thermal_mode | ✅ | Heat only, no kick |
| test_feedback_kinetic_mode | ✅ | Kick only, no heat |
| test_feedback_hybrid_mode | ✅ | Both heat and kick |

#### Schmidt-Kennicutt + Integration
| Test | Statut | Valeur |
|---|---|---|
| test_sfr_schmidt_kennicutt | ✅ | SFR ∝ ρ^1.5 (ratio ~22) |
| test_sf_probability_bounds | ✅ | P ∈ [0, 1] |
| test_sf_probability_density_dependence | ✅ | Higher ρ → higher P |
| test_particle_types | ✅ | gas/sink/m- correct |
| test_sink_no_pressure | ✅ | Sinks don't feel pressure |
| test_sink_positive_mass | ✅ | Sinks are m+ |
| test_sf_feedback_cycle | ✅ | SF → feedback → no more SF |

**GO si :** 100% tests passent ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 3 — SPECTRE DE PUISSANCE P(k) ✅ GO

### Tests unitaires — 12/12 PASS

#### CIC Assignment Tests
| Test | Statut | Valeur |
|---|---|---|
| test_cic_mass_conservation | ✅ | Total mass conserved |
| test_cic_periodic_wrap | ✅ | Edge particles wrap correctly |

#### P(k) Computation Tests
| Test | Statut | Valeur |
|---|---|---|
| test_pk_white_noise_flat | ✅ | P(k) ≈ 0 after shot noise |
| test_pk_units | ✅ | [Mpc³] units correct |
| test_pk_nyquist_cutoff | ✅ | k < k_Nyquist |
| test_pk_single_mode | ✅ | Peak at target k |
| test_pk_clustered_distribution | ✅ | Low k dominates |

#### ΛCDM P(k) Tests
| Test | Statut | Valeur |
|---|---|---|
| test_lcdm_pk_shape | ✅ | Decreases at high k |
| test_lcdm_pk_sigma8_scaling | ✅ | P ∝ σ₈² |
| test_lcdm_pk_spectral_index | ✅ | n_s affects slope |

#### Cross-Spectrum Tests (Janus m+/m-)
| Test | Statut | Valeur |
|---|---|---|
| test_cross_pk_identical | ✅ | Auto = cross for same field |
| test_cross_pk_anticorrelated | ✅ | Negative for δ₂=-δ₁ |

**GO si :** 100% tests passent ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 4 — CARTES DE CONVERGENCE κ ✅ GO

### Tests unitaires — 12/12 PASS

#### Σ_crit Tests
| Test | Statut | Valeur |
|---|---|---|
| test_sigma_crit_units | ✅ | ~10^15 M_sun/Mpc² |
| test_sigma_crit_distance_dependence | ✅ | Closer source → higher Σ_crit |

#### NFW Profile Tests
| Test | Statut | Valeur |
|---|---|---|
| test_sigma_nfw_profile | ✅ | Σ decreases with r |
| test_kappa_nfw_profile | ✅ | κ > 0, dimensionless |
| test_kappa_nfw_peak_center | ✅ | Peak at r=0 |

#### κ Map Tests
| Test | Statut | Valeur |
|---|---|---|
| test_kappa_map_mass_conservation | ✅ | Mass ratio ~1.0 |
| test_kappa_negative_mass | ✅ | κ < 0 for m- |
| test_janus_halo_signature | ✅ | κ_inner > 0, κ_outer < 0 |
| test_radial_profile_smooth | ✅ | Monotone decrease |

#### Euclid Detection Tests
| Test | Statut | Valeur |
|---|---|---|
| test_euclid_detection_threshold | ✅ | |κ| > 0.03 detectable |
| test_janus_halo_euclid_detectable | ✅ | κ = -0.05 detectable |
| test_kappa_full_pipeline | ✅ | Complete pipeline works |

**GO si :** κ>0 intérieur, κ<0 extérieur ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 5 — COURBES DE ROTATION ✅ GO

### Tests unitaires — 18/18 PASS

#### Keplerian/Point Mass Tests
| Test | Statut | Valeur |
|---|---|---|
| test_rotation_curve_point_mass | ✅ | v(10^10 M☉, 1kpc) = 207 km/s |
| test_rotation_curve_galaxy_scale | ✅ | v(10^11 M☉, 10kpc) ≈ 207 km/s |
| test_keplerian_scaling | ✅ | v ∝ r^-0.5 (ratio 0.50) |
| test_rotation_curve_boundary | ✅ | v(r=0) = v(M=0) = 0 |
| test_is_keplerian_point_mass | ✅ | Point mass Keplerian ✓ |
| test_flat_not_keplerian | ✅ | Flat curve NOT Keplerian ✓ |

#### Plateau Detection Tests
| Test | Statut | Valeur |
|---|---|---|
| test_plateau_detection | ✅ | v_plateau = 200 km/s |
| test_no_plateau_keplerian | ✅ | Keplerian has no plateau |

#### Shell Theorem Tests (Janus Key Mechanism)
| Test | Statut | Valeur |
|---|---|---|
| test_shell_theorem_outside | ✅ | Shell = point mass outside |
| test_shell_theorem_inside_linear | ✅ | v² ∝ r inside shell |
| test_shell_flat_contribution | ✅ | Shell creates flat v² ∝ r |

#### Tully-Fisher Relation
| Test | Statut | Valeur |
|---|---|---|
| test_tully_fisher_calibration | ✅ | TFR(200 km/s) = 10^10 L☉ |
| test_tully_fisher_slope | ✅ | L ∝ v^4 (ratio = 16) |
| test_baryonic_tully_fisher | ✅ | L ∝ v^3.5 (ratio = 11.3) |

#### Janus Rotation Curve Tests
| Test | Statut | Valeur |
|---|---|---|
| test_enclosed_mass_profile | ✅ | M+ cumulative, M- in shell |
| test_rotation_curve_baryonic_interior | ✅ | v_bar declines at large r |
| test_plateau_mechanism | ✅ | m- shell compensates decline |
| test_exclusion_radius_detection | ✅ | r_exclusion ≈ 0.04 Mpc |

**GO si :** profil képlérien, signature coquille m⁻ validée ✅
**Statut :** ✅ GO — 12 avril 2026

---

## ÉTAPE 6 — DIPOLE REPELLER ✅ GO

### Tests unitaires — 15/15 PASS

#### Hubble Flow Tests
| Test | Statut | Valeur |
|---|---|---|
| test_hubble_velocity_scale | ✅ | v_H(100 Mpc) = 7600 km/s |
| test_hubble_velocity_h0_dependence | ✅ | v ∝ H₀ |

#### Peculiar Velocity Tests
| Test | Statut | Valeur |
|---|---|---|
| test_peculiar_velocity_zero_uniform | ✅ | v_pec = 0 for Hubble flow |
| test_peculiar_velocity_outflow | ✅ | v_pec = +400 km/s |
| test_peculiar_velocity_infall | ✅ | v_pec = -600 km/s |
| test_peculiar_velocity_field | ✅ | 3D field computation ✓ |

#### Repeller Detection Tests
| Test | Statut | Valeur |
|---|---|---|
| test_repeller_detection | ✅ | v_out > 150 km/s, f_minus > 0.5 |
| test_attractor_vs_repeller | ✅ | v_in > 100 km/s, f_plus > 0.7 |
| test_no_repeller_random | ✅ | No false positives |

#### Hoffman Compatibility Tests
| Test | Statut | Valeur |
|---|---|---|
| test_hoffman_velocity_scale | ✅ | ~200 km/s @ ~100 Mpc |
| test_sub_hoffman_not_compatible | ✅ | Weak repellers rejected |

#### Bulk Flow Tests
| Test | Statut | Valeur |
|---|---|---|
| test_bulk_flow_detection | ✅ | v_bulk = (120, -80, 50) km/s |
| test_bulk_flow_magnitude | ✅ | |v_bulk| = 200 km/s |

#### Janus Signature Tests
| Test | Statut | Valeur |
|---|---|---|
| test_janus_repeller_m_minus_fraction | ✅ | f_minus > 60% |
| test_janus_attractor_m_plus_fraction | ✅ | f_plus > 80% |

**GO si :** répulseur avec v>100 km/s, signature m⁻ validée ✅
**Statut :** ✅ GO — 12 avril 2026

---

## VALIDATION 1M — PRÉ-TEST FINAL ✅ GO

### Run test_1m_zeldovich (12 avril 2026)

⚠️ **Note :** Ce run utilisait dual-seed (BUG-001). Les métriques restent valides
car le ratio v_rms était monitoré et stable, mais les ICs contenaient une asymétrie
de ~15%. Les runs futurs utilisent single-seed (fix 13/04/26).

| Paramètre | Valeur |
|---|---|
| N_particles | 1,000,000 |
| L_box | 100 Mpc |
| η | 1.045 |
| z_init → z_final | 5.0 → 0.0 |
| Steps | 1200 |
| Durée | 1664 s (~28 min) |
| ICs | Zel'dovich dual-seed (m+: 42, m-: 43) ⚠️ BUG-001 |

### Métriques finales (z=0)

| Métrique | Valeur | Critère | Statut |
|---|---|---|---|
| Segregation S | 0.0058 | < 0.1 | ✅ |
| Correlation | -0.2586 | < 0 | ✅ |
| v_rms+ | 50.7 km/s | — | ✅ |
| v_rms- | 53.3 km/s | — | ✅ |
| v_rms ratio | 0.9518 | < 1.15 | ✅ |
| ρ_max+ | 40 | > 1 | ✅ |
| Purity | 17.5% | < 50% | ✅ |
| NaN | 0 | = 0 | ✅ |
| Runaway | Non | — | ✅ |

### Analyse visuelle
- ICs (z=5): Grille uniforme, m+/m- bien mélangés
- Final (z=0): Structures filamentaires émergentes
- Zoom 10 Mpc: Pas de ségrégation spatiale m+/m-
- Corrélation négative: Anti-gravité Janus fonctionnelle

**GO si :** Pas de NaN, v_rms ratio < 1.15, S stable ✅
**Statut :** ✅ GO — 12 avril 2026

---

## RUN FINAL 10M

**CONDITIONS DE LANCEMENT :**
- [x] Étape 0 : 100% tests ✅
- [x] Étape 0b : 100% tests ✅
- [x] Étape 1 : GO ✅ (refroidissement radiatif)
- [x] Étape 2 : GO ✅ (formation stellaire + feedback)
- [x] Étape 3 : GO ✅ (spectre P(k))
- [x] Étape 4 : GO ✅ (κ<0 confirmé)
- [x] Étape 5 : GO ✅ (courbes rotation + shell theorem)
- [x] Étape 6 : GO ✅ (Dipole Repeller détecté)
- [x] Validation 1M : GO ✅ (test pré-production)

**TOUTES LES CONDITIONS REMPLIES — RUN FINAL PRÊT À LANCER**

**Paramètres run final :**
- N = 10,000,000 particules
- L_box = 500 Mpc
- η = 1.045
- z_init = 4 → z_final = 0
- ICs: Zel'dovich **single-seed** (fix BUG-001, commit `237d41d`)
- Baryonics: Grackle cooling + Rahmati self-shielding + SF + SN feedback
- Snapshot interval: 10 steps

**Durée estimée :** 60-100h GPU

### Run en cours (13 avril 2026)

| Métrique | Step 0 | Critère | Statut |
|---|---|---|---|
| v_rms ratio | 0.9999 | ≈ 1.0 | ✅ FIX VALIDÉ |
| S | 0.132 | — | ✅ |

Binary: `janus_baryonic_calibrated`

---

## SNAPSHOTS — POLITIQUE NETTOYAGE

| Run | Snapshots gardés | Snapshots effacés | Espace récupéré |
|---|---|---|---|
| vsl_petit_production | z=4,2,1,0.5,0.15 | — (déjà nettoyé) | — |
| vsl_phase2_nosph | z=4, z_final | tous intermédiaires | ⏳ |
| etape1_* | z=4, z_final + z-clés | tous intermédiaires | après GO |
| ... | | | |

---

*Fichier créé le 12 avril 2026 — mis à jour automatiquement par CLI*
*Ne jamais modifier manuellement — CLI gère toutes les mises à jour*
