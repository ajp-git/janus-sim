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
| 3 | Spectre P(k) | ⏳ en attente | — | — | — | — | — | — |
| 4 | Cartes κ | ⏳ en attente | — | — | — | — | — | — |
| 5 | Courbes rotation | ⏳ en attente | — | — | — | — | — | — |
| 6 | Dipole Repeller | ⏳ en attente | — | — | — | — | — | — |
| F | Run final 10M | ⏳ bloqué | — | — | — | — | — | — |

**Légende :** ⏳ en attente | 🔄 en cours | ✅ GO | ❌ NO-GO

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

## ÉTAPE 3 — SPECTRE DE PUISSANCE P(k)

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_cic_mass_conservation | ⏳ | — |
| test_pk_white_noise | ⏳ | — |
| test_pk_single_mode | ⏳ | — |
| test_cross_spectrum_anticorrelation | ⏳ | — |
| test_pk_units | ⏳ | — |
| test_nyquist_cutoff | ⏳ | — |

### Résultats P(k)
| z | σ₈ | BAO position | χ²/dof | r(k)<0 | Statut |
|---|---|---|---|---|---|
| 4.0 | ⏳ | — | — | — | — |
| 2.0 | ⏳ | — | — | — | — |
| 1.0 | ⏳ | — | — | — | — |
| 0.5 | ⏳ | — | — | — | — |
| 0.0 | ⏳ | — | — | — | — |

**Valeurs cibles :**
- σ₈ ∈ [0.65, 0.85] (KiDS: 0.70, Planck: 0.80)
- BAO à k=0.05, 0.10, 0.15 h/Mpc ±5%
- χ²/dof < 2 vs SDSS DR16

**GO si :** σ₈ dans intervalle, BAO détectées, χ²<2
**Statut :** ⏳ en attente

---

## ÉTAPE 4 — CARTES DE CONVERGENCE κ

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_kappa_nfw_profile | ⏳ | — |
| test_kappa_negative_shell | ⏳ | — |
| test_sigma_crit_units | ⏳ | — |
| test_projection_mass_conservation | ⏳ | — |
| test_euclid_detection_threshold | ⏳ | — |

### Résultats κ(r)
| Halo | R₂₀₀ | κ(r<R₂₀₀) | κ(r>3R₂₀₀) | Détectable Euclid | Statut |
|---|---|---|---|---|---|
| #1 | ⏳ | — | — | — | — |
| #2 | ⏳ | — | — | — | — |
| ... | | | | | |

**GO si :** κ>0 intérieur, κ<0 extérieur, |κ_outer|>10⁻³
**Statut :** ⏳ en attente

---

## ÉTAPE 5 — COURBES DE ROTATION

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_rotation_curve_baryonic_interior | ⏳ | — |
| test_shell_theorem_negative_mass | ⏳ | — |
| test_rotation_curve_point_mass | ⏳ | — |
| test_exclusion_radius_detection | ⏳ | — |
| test_tully_fisher_calibration | ⏳ | — |
| test_plateau_mechanism | ⏳ | — |

### Résultats v_rot(r)
| Halo | R₂₀₀ | r_plateau | v_plateau | Képlérien intérieur | Statut |
|---|---|---|---|---|---|
| #1 | ⏳ | — | — | — | — |
| ... | | | | | |

**GO si :** profil képlérien r<3R₂₀₀, signature coquille m⁻ détectée
**Statut :** ⏳ en attente

---

## ÉTAPE 6 — DIPOLE REPELLER

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_peculiar_velocity_zero_uniform | ⏳ | — |
| test_repeller_detection | ⏳ | — |
| test_hoffman_velocity_scale | ⏳ | — |
| test_attractor_vs_repeller | ⏳ | — |

### Résultats
| Métrique | Obtenu | Hoffman 2017 | Statut |
|---|---|---|---|
| N répulseurs majeurs | ⏳ | ≥1 | — |
| Vitesse max répulseur | ⏳ | ~200 km/s | — |
| Taille vide associé | ⏳ | >30 Mpc | — |

**GO si :** ≥1 répulseur avec v>100 km/s, taille compatible
**Statut :** ⏳ en attente

---

## RUN FINAL 10M

**CONDITIONS DE LANCEMENT :**
- [ ] Étape 0 : 100% tests ✅
- [ ] Étape 0b : 100% tests ✅
- [ ] Étape 1 : GO ✅
- [ ] Étape 2 : GO ✅ (paramètres feedback fixés)
- [ ] Étape 3 : GO ✅ (σ₈ + BAO OK)
- [ ] Étape 4 : GO ✅ (κ<0 confirmé)
- [ ] Étape 5 : GO ✅ (courbes rotation OK)
- [ ] Étape 6 : GO ✅ (Dipole Repeller détecté)

**Paramètres run final :** à compléter après Étapes 1-6
**ETA lancement :** ~2-4 semaines
**Durée estimée :** 40-100h GPU

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
