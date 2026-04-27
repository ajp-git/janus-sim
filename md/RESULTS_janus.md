# JANUS PROJECT — RESULTS
## Suivi complet des résultats — mis à jour automatiquement par CLI
### Objectif : "Incroyable, ça marche. La matière noire n'existe pas."

---

## TABLEAU DE BORD GLOBAL

| Étape | Nom | Statut | Date | σ₈ | BAO | κ<0 | v_rot | Dipole |
|---|---|---|---|---|---|---|---|---|
| 0 | Fondations | ⏳ en attente | — | — | — | — | — | — |
| 0b | ICs Zel'dovich | ⏳ en attente | — | — | — | — | — | — |
| 1 | Refroidissement | ⏳ en attente | — | — | — | — | — | — |
| 2 | Formation stellaire | ⏳ en attente | — | — | — | — | — | — |
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

## ÉTAPE 0 — FONDATIONS

### Tests unitaires
*À compléter par CLI*

| Test | Statut | Valeur |
|---|---|---|
| test_cooling_table_bounds | ⏳ | — |
| test_lambda_known_values | ⏳ | — |
| test_bremsstrahlung_slope | ⏳ | — |
| test_uv_suppresses_igm | ⏳ | — |
| test_uv_negligible_halos | ⏳ | — |
| test_uv_peak_z2 | ⏳ | — |
| test_jeans_mass_solar | ⏳ | — |
| test_freefall_time | ⏳ | — |
| test_sfr_threshold | ⏳ | — |
| test_sn_energy_units | ⏳ | — |

**GO si :** 100% tests passent
**Statut :** ⏳ en attente

---

## ÉTAPE 0b — ICs ZEL'DOVICH

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_corr_initial_zero | ⏳ | — |
| test_delta_rms_target | ⏳ | — |
| test_pk_slope | ⏳ | — |
| test_positions_in_box | ⏳ | — |
| test_bao_peak_present | ⏳ | — |
| test_growth_factor_loaded | ⏳ | — |

**GO si :** 100% tests passent
**Statut :** ⏳ en attente

---

## ÉTAPE 1 — REFROIDISSEMENT RADIATIF

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_isolated_cloud_cooling | ⏳ | — |
| test_cooling_floor | ⏳ | — |
| test_bremsstrahlung_slope | ⏳ | — |
| test_t_cool_order_of_magnitude | ⏳ | — |
| test_uv_suppresses_cooling_low_density | ⏳ | — |
| test_uv_negligible_halos | ⏳ | — |
| test_uv_peak_at_z2 | ⏳ | — |

### Simulations de validation
| Niveau | N | Durée | Temps GPU | Statut | T_mean(z=2) | ρ+_max(z=2) | ratio |
|---|---|---|---|---|---|---|---|
| 1a — 100K | 100K | <5min | ⏳ | ⏳ | — | — | — |
| 1b — 500K | 500K | <30min | ⏳ | ⏳ | — | — | — |
| 1c — 1M | 1M | <2h | ⏳ | ⏳ | — | — | — |

**GO si :** T_mean décroissant, ρ+_max croissant, ratio<1.10
**Statut :** ⏳ en attente

---

## ÉTAPE 2 — FORMATION STELLAIRE + FEEDBACK

### Tests unitaires
| Test | Statut | Valeur |
|---|---|---|
| test_star_formation_probability | ⏳ | — |
| test_no_stars_hot_gas | ⏳ | — |
| test_sn_energy_injection | ⏳ | — |
| test_sfr_schmidt_kennicutt | ⏳ | — |
| test_stellar_mass_growth | ⏳ | — |

### Scan paramétrique feedback (48 runs courts)
| ε | n_th | Mode | N_stars | SFR(z=1) | ρ⁻/ρ⁺ halos | Statut |
|---|---|---|---|---|---|---|
| 0.1% | 1 | thermique | ⏳ | — | — | — |
| 0.3% | 10 | cinétique | ⏳ | — | — | — |
| 1.0% | 50 | cinétique | ⏳ | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

**Paramètres retenus :** ε=?, n_th=?, mode=?
**GO si :** N_stars>0, SFR cohérent Madau plot, ρ⁻/ρ⁺<0.01 après SN
**Statut :** ⏳ en attente

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
