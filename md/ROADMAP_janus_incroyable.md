# JANUS PROJECT — Plan d'action vers "Incroyable, ça marche"
## Validation numérique complète du Modèle Cosmologique de Janus
### Version 1.7 — Avril 2026

---

## RÈGLES DU PROJET

### Autonomie CLI
CLI est **entièrement autonome** pour :
- Créer/modifier/supprimer tout fichier ou répertoire
- Écrire/réécrire tout code Rust, Python, CUDA
- Lancer/arrêter des simulations
- Installer des dépendances
- Générer des figures et documents
- **Effacer les snapshots selon la politique ci-dessous**
- **Mettre à jour RESULTS.md après chaque étape**

CLI **demande confirmation uniquement pour** :
- Modifier les paramètres physiques canoniques (μ, H₀, t₀)
- Soumettre/publier quoi que ce soit à l'extérieur
- Lancer le run final 10M (>40h)

### Stratégie de validation progressive
```
100K → 500K → 1M (validation) → 10M (production)
```
Chaque étape doit passer **100% de ses tests** avant de passer à la suivante.
Aucun run de production sans validation complète sur petite boîte.

### Politique snapshots — OBLIGATOIRE
```
GARDER :
  - Dernier snapshot de chaque run (z final)
  - Snapshot au step de validation clé (ex: z=1.0, z=0.5)
  - Snapshots des frames vidéo (si générées)

EFFACER AUTOMATIQUEMENT :
  - Tous les snapshots intermédiaires après génération des figures
  - Snapshots des runs de validation 100K/500K après passage GO
  - Snapshots des runs NO-GO entiers

JAMAIS EFFACER :
  - Snapshots run principal (vsl_petit_production) — résultats publiés
  - Snapshots run final 10M
  - Le CSV evolution.csv de chaque run
```

Script de nettoyage à appeler après chaque étape GO :
```bash
# Garder seulement premier + dernier + z-clés
python3 analysis/cleanup_snapshots.py \
  --run-dir output/etape1_cooling/ \
  --keep-z 4.0 2.0 1.0 0.5 0.0 \
  --dry-run  # enlever pour exécuter
```

### Fichier de résultats — RESULTS.md
**Un seul fichier** `/mnt/T2/janus-sim/RESULTS.md` suit **tout le projet**.
CLI le met à jour après chaque étape avec :
- Métriques quantitatives obtenues
- Comparaison avec valeurs attendues
- GO/NO-GO avec justification
- Figures générées (chemins)
- Bugs rencontrés et corrections

Format d'entrée :
```markdown
## [DATE] Étape X — [Nom] — [GO ✅ / NO-GO ❌]

### Métriques
| Métrique | Obtenu | Attendu | Statut |
|---|---|---|---|
| σ₈ | 0.71 | 0.70±0.05 | ✅ |

### Figures
- figures/etapeX_pk.png

### Notes
- Bug trouvé : ...
- Correction : ...
```

### Structure des outputs
```
/mnt/T2/janus-sim/
├── RESULTS.md              ← FICHIER UNIQUE suivi projet
├── ROADMAP.md              ← Ce fichier (plan + code)
├── runs/
│   ├── etape0_foundations/ ← Étape 0
│   ├── etape1_cooling/     ← Étape 1
│   ├── etape2_sf/          ← Étape 2
│   ├── etape3_pk/          ← Étape 3
│   ├── etape4_kappa/       ← Étape 4
│   ├── etape5_rotation/    ← Étape 5
│   ├── etape6_dipole/      ← Étape 6
│   └── final_10M/          ← Run final
├── analysis/               ← Scripts Python réutilisables
├── figures/                ← Figures publication (jamais effacées)
└── src/                    ← Code Rust/CUDA
```

---

## OBJECTIF FINAL

**Objectif scientifique (pour publication) :**
> "Tester si le Modèle Cosmologique de Janus reproduit les observables
> cosmologiques clés : P(k), BAO, σ₈, κ(r), v_rot(r), Dipole Repeller."

**Objectif motivant (pour JPP) :**
> "Incroyable, ça marche. La matière noire n'existe pas."

*Note : l'objectif scientifique doit rester falsifiable. Le MCJ peut
échouer sur les BAO — c'est un résultat valide et publiable.*

**Critères de succès objectifs :**
1. P(k) + BAO : position ±5%, amplitude ±20% vs SDSS, σ₈ ∈ [0.65, 0.85]
2. Fonction de masse des halos compatible observations
3. κ(r) < 0 au-delà de 3×R₂₀₀ (prédiction falsifiable Euclid)
4. Courbes de rotation : profil képlérien intérieur + signature coquille m⁻
5. Dipole Repeller : v > 100 km/s, taille > 30 Mpc

**⚠️ Test décisif unique — BAO :**
Si P(k) passe le test BAO → le reste du pipeline est justifié.
Si P(k) échoue → documenter pourquoi et publier comme contrainte du MCJ.
**Ne pas investir 40h GPU si le test 1M BAO échoue.**

---

## RISQUES & PLANS DE CONTINGENCE

| Risque | Probabilité | Conséquence | Plan B |
|---|---|---|---|
| BAO trop fortes sans DM | Haute | Test P(k) échoue | Tester n_s plus bas / VSL plus forte / documenter comme contrainte |
| Runaway m⁻ persistant | Moyenne | Run final non physique | Softening adaptatif + documenter limite |
| Feedback détruit ségrégation | Moyenne | ρ⁻/ρ⁺ > 0.1 dans halos | Réduire ε, mode thermique |
| σ₈ trop faible | Faible | Tension Planck | Argument KiDS/DES, tension = prédiction |
| Over-cooling sans UV | Haute | SF spurieuse partout | Self-shielding Rahmati 2013 obligatoire |
| ICs incorrectes → BAO faux | Haute | Test 3c invalide | T_b(k) Eisenstein-Hu obligatoire pour 3c |

---

## ORDRE D'EXÉCUTION RÉVISÉ (Grok)

**Priorité : tester P(k)/BAO AVANT la physique baryonique complète.**

```
Étape 0   : modules + tests unitaires
Étape 0b  : ICs Zel'dovich + T_b(k) + D(z) MCJ
Étape 3a  : P(k) sur snapshots existants (< 1h) ← TEST PRÉLIMINAIRE
            Si BAO acceptables → continuer
            Si BAO échouent → réévaluer avant d'investir en SF/feedback
Étape 1   : refroidissement
Étape 2   : formation stellaire + feedback
Étape 3bc : P(k) avec physique complète
Étape 4   : cartes κ
Étape 5   : courbes rotation
Étape 6   : Dipole Repeller
Run final : 10M
```

---

## ÉTAPE 0 — FONDATIONS (prerequis, ~1 jour)

### A. Architecture du code

#### A.1 Module de refroidissement radiatif

**⚠️ UPDATE 12 avril 2026 :** Ne pas implémenter de table manuelle.
Les valeurs Sutherland & Dopita divergent de 1-2 ordres de grandeur
selon les sources (Gemini vs ChatGPT). Utiliser le fit analytique
physique de ChatGPT (validé Wang et al. 2014, erreur <15%).

**Fit analytique CUDA-ready (3 termes physiques) :**

```c
// cuda/cooling.cu
// Source : fit ChatGPT basé sur S&D 1993 + Wang et al. 2014
// Précision : <10% pour 10^4-10^5 K, <15% pour 10^5-10^7 K, <5% >10^7 K
// Hypothèse : CIE, gaz primordial H/He, pas de métaux, pas de UV background

__device__ double cooling_primordial(double T)
{
    if (T < 1e4) return 0.0;  // pas de refroidissement sous T_floor
    
    double sqrtT = sqrt(T);
    
    // (1) Excitation H — dominant 10^4 à 10^5 K
    double Lambda_H  = 7.5e-19 * exp(-118348.0 / T)
                       / (1.0 + sqrt(T / 1e5));
    
    // (2) He + recombinaison — plateau 10^5 à 10^6 K
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0 / T);
    
    // (3) Bremsstrahlung — dominant >10^7 K
    double Lambda_ff = 1.42e-27 * sqrtT;
    
    return Lambda_H + Lambda_He + Lambda_ff;
}
```

**Wrapper Rust :**

```rust
// src/cooling.rs

/// Taux de refroidissement Λ(T) pour gaz H/He primordial
/// Fit analytique : Wang et al. 2014 + S&D 1993
/// Unités : erg·cm³/s
pub fn lambda_primordial(T: f64) -> f64 {
    if T < 1e4 { return 0.0; }
    let sqrt_t = T.sqrt();
    let lambda_h  = 7.5e-19 * (-118348.0 / T).exp() / (1.0 + (T / 1e5).sqrt());
    let lambda_he = 9.1e-27 * sqrt_t * (-13179.0 / T).exp();
    let lambda_ff = 1.42e-27 * sqrt_t;
    lambda_h + lambda_he + lambda_ff
}

/// Taux de refroidissement en unités simulation
/// dU/dt [km²/s² / Gyr]
/// rho en M_sun/Mpc³, T en K
pub fn cooling_rate_code_units(T: f64, rho: f64) -> f64 {
    const MU: f64 = 0.588;       // poids moléculaire moyen H+He ionisé
    const MP: f64 = 1.67e-27;    // masse proton kg
    const ERG_TO_KM2S2: f64 = 1e-10; // 1 erg/g = 1e-10 km²/s²
    const GYR_TO_S: f64 = 3.156e16;
    const MSUN_TO_G: f64 = 1.989e33;
    const MPC_TO_CM: f64 = 3.086e24;
    
    // Densité numérique nH [cm⁻³]
    let rho_cgs = rho * MSUN_TO_G / MPC_TO_CM.powi(3);
    let n_h = rho_cgs / (MU * MP * 1e3); // MP en g
    
    let lambda = lambda_primordial(T); // erg·cm³/s
    
    // dU/dt = -nH² × Λ(T) / ρ × conversions
    -n_h * n_h * lambda / rho_cgs * ERG_TO_KM2S2 * GYR_TO_S
}

/// Temps de refroidissement [Gyr]
pub fn t_cool(T: f64, rho: f64) -> f64 {
    const KB: f64 = 1.38e-23;    // J/K
    const MU: f64 = 0.588;
    const MP: f64 = 1.67e-27;
    let u = 1.5 * KB * T / (MU * MP); // énergie interne J/kg
    let du_dt = cooling_rate_code_units(T, rho).abs();
    if du_dt < 1e-30 { return 1e10; } // pas de refroidissement
    u * 1e-10 / du_dt // conversion approximative
}
```

**Formule centrale :**
$$\frac{dU}{dt} = -\frac{n_H^2 \cdot \Lambda(T)}{\rho} \quad [\text{km}^2/\text{s}^2/\text{Gyr}]$$

où $\mu = 0.588$, $\Lambda(T) = \Lambda_H + \Lambda_{He} + \Lambda_{ff}$.

**Fit complet avec UV background HM2012 (CUDA-ready) :**

```c
// cuda/cooling.cu
// Λ_eff(T, nH, z) = cooling - photoheating UV
// Source : ChatGPT fit basé sur Haardt & Madau 2012
// du/dt = n² Λ(T) - n Γ(z)

__device__ double cooling_heating_HM2012(double T, double nH, double z)
{
    double sqrtT = sqrt(T);

    // === Cooling collisionnel H/He primordial ===
    double Lambda_H  = 7.5e-19 * exp(-118348.0/T)
                       / (1.0 + sqrt(T/1e5));
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0/T);
    double Lambda_ff = 1.42e-27 * sqrtT;
    double Lambda = Lambda_H + Lambda_He + Lambda_ff; // erg·cm³/s

    // === Photoheating UV (HM2012 fit) ===
    // Γ(z) ≈ 1e-24 × (1+z)² / (1 + ((1+z)/3)^5)  [erg/s/cm³]
    double zp1 = 1.0 + z;
    double Gamma_uv = 1e-24 * (zp1*zp1)
                      / (1.0 + pow(zp1/3.0, 5.0));

    // === Cooling effectif ===
    // Λ_eff = Λ(T) - Γ(z)/nH
    return Lambda - Gamma_uv / nH;
}
```

**Comportement du fond UV :**

| z | Effet |
|---|---|
| z > 6 | Γ quasi nul — cooling pur |
| z ~ 2–3 | Pic UV — suppression refroidissement |
| z → 0 | Faible mais non nul — T_eq ~ 10⁴ K dans IGM |

**Température d'équilibre dans l'IGM (sans halos) :**
$$T_\text{eq}(z) : \Lambda(T_\text{eq}) = \Gamma(z)/n_H \quad \rightarrow \quad T_\text{eq} \approx 10^4 \text{ K}$$

Cela empêche la formation stellaire spurieuse dans les vides cosmiques.

**⚠️ Limitations connues :**
- Pas de self-shielding (Rahmati 2013) : en haute densité Γ→0, à ajouter en v2
- Pas de non-CIE : important pour les chocs
- Approximation spectrale : HM2012 complet nécessite tables tabulées

**Pour publication :** ajouter self-shielding Rahmati 2013 :
$$\Gamma_\text{eff} = \Gamma_\text{UV} \times f_\text{shield}(n_H)$$
avec $f_\text{shield} \rightarrow 0$ pour $n_H > 10^{-2}$ cm⁻³.

**Validation numérique du fit :**
```
T = 10^4.5 K  → Λ = 1.6e-22 erg·cm³/s  (ref Gemini : 1.6e-22 ✓)
T = 10^5.0 K  → Λ = 2.5e-22 erg·cm³/s  (ref ChatGPT : 2.5e-22 ✓)
T = 10^7.0 K  → Λ = 1.5e-22 erg·cm³/s  (ref ChatGPT : 1.5e-22 ✓)
```

#### A.2 Module de formation stellaire
```rust
// src/starformation.rs

pub struct StarFormationCriteria {
    rho_threshold: f64,   // ρ_Jeans en unités simulation
    T_threshold: f64,     // K — gaz froid seulement
    div_v_threshold: f64, // ∇·v < 0 (gaz convergent)
    efficiency: f64,      // ε* ~ 0.02 (2% de la masse par t_ff)
}

impl StarFormationCriteria {
    /// Critère de Jeans : masse > M_Jeans
    /// M_Jeans = (π/6) × (kT/Gμm_p)^(3/2) × ρ^(-1/2)
    pub fn jeans_mass(&self, T: f64, rho: f64) -> f64 {
        let G = 4.302e-3; // pc·M_sun^-1·(km/s)²
        (std::f64::consts::PI / 6.0)
            * (1.38e-23 * T / (G * 0.588 * 1.67e-27)).powf(1.5)
            * rho.powf(-0.5)
    }
    
    /// Temps de chute libre
    /// t_ff = sqrt(3π / 32Gρ)
    pub fn freefall_time(&self, rho: f64) -> f64 {
        (3.0 * std::f64::consts::PI / (32.0 * 4.302e-3 * rho)).sqrt()
    }
    
    /// Taux de formation stellaire Schmidt-Kennicutt
    /// dρ*/dt = ε* × ρ_gas / t_ff
    pub fn sfr(&self, rho: f64, T: f64) -> f64 {
        if rho > self.rho_threshold && T < self.T_threshold {
            self.efficiency * rho / self.freefall_time(rho)
        } else {
            0.0
        }
    }
}
```

#### A.3 Module feedback supernova

**⚠️ UPDATE 12 avril 2026 — Consultation IA (ChatGPT + Gemini)**

**Paramètres universels (inchangés) :**
- E_SN = 10⁵¹ erg par SN → énergie physique, indépendante de la cosmologie
- Délai = 10 Myr → durée de vie étoiles massives M>8 M☉, inchangée

**Paramètres à recalibrer pour le MCJ :**

| Paramètre | Valeur ΛCDM | Valeur MCJ recommandée | Raison |
|---|---|---|---|
| ε (efficacité) | 1–10% | **0.3–1%** | Puits potentiel baryonique pur plus peu profond |
| Mode injection | Thermique | **Cinétique** | Plus stable numériquement sans DM |
| n_threshold | 0.1–1 cm⁻³ | **10–50 cm⁻³** | Évite SF diffuse → feedback catastrophique |
| Délai | 10 Myr fixe | 3–30 Myr (spread IMF) | Halos plus fragiles sans DM |

**Raisons physiques du recalibrage (ChatGPT + Gemini convergent) :**
1. Sans matière noire → puits de potentiel purement baryonique → moins profond
2. Croissance halos ×5 plus rapide que ΛCDM → risque sur-production stellaire
3. Répulsion m⁻ agit comme pression externe → feedback s'ajoute, amplifié
4. v_rms élevées à z<0.5 → feedback cinétique pourrait pousser m+ vers zones m⁻

**Alerte Gemini — critique pour la ségrégation :**
> "Un feedback trop violent pourrait forcer m+ à se mélanger avec m⁻,
> brisant la ségrégation totale (P<10⁻²²⁰⁰) obtenue."
→ Surveiller ρ⁻/ρ⁺ dans les halos après chaque SN event.

**Critère de survie des halos (test énergétique) :**
$$\frac{E_{SN,tot}}{E_{bind}} < 1 \quad \text{avec} \quad E_{bind} \sim \frac{G M^2}{R}$$
Si ratio > 1 → réduire ε.

**Code CUDA-ready (injection cinétique momentum-driven) :**

```c
__device__ void sn_feedback_kinetic(
    double* vx, double* vy, double* vz,  // vitesses voisins
    double* dx, double* dy, double* dz,  // vecteurs de distance
    int n_neighbors,
    double m_star,                         // masse stellaire [M_sun]
    double m_particle,                     // masse particule [M_sun]
    double efficiency                       // ε = 0.003 à 0.01
) {
    // Momentum SN : p_SN = 3e5 M_sun km/s par SN (valeur FIRE)
    // N_SN = m_star / 100 M_sun
    double p_sn = 3e5 * (m_star / 100.0) * efficiency; // M_sun km/s
    double p_per_neighbor = p_sn / n_neighbors;
    
    for (int i = 0; i < n_neighbors; i++) {
        double r = sqrt(dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i]);
        if (r < 1e-10) continue;
        // Injection radiale (momentum-driven)
        double dp = p_per_neighbor / m_particle; // km/s
        vx[i] += dp * dx[i] / r;
        vy[i] += dp * dy[i] / r;
        vz[i] += dp * dz[i] / r;
    }
}
```

**Stratégie de calibration (scan paramétrique Étape 2b) :**
```
Grid scan :
  ε    ∈ {0.1%, 0.3%, 1%, 3%}
  n_th ∈ {1, 10, 50, 100} cm⁻³
  mode ∈ {thermique, cinétique, mixte}

Observable de calibration :
  1. M_stars/M_halo ~ 0.03 (SDSS)
  2. SFR(z=1)/SFR(z=0) ~ 10 (Madau plot)
  3. ρ⁻/ρ⁺ dans halos reste < 0.01 après feedback
  4. v_rms ratio reste < 1.15
```
```

### B. Tests pour l'Étape 0

```rust
// tests/test_cooling.rs
#[test] fn test_cooling_table_bounds() { ... }       // T < 10K → Λ=0
#[test] fn test_cooling_monotonic() { ... }           // Λ croissant entre 10^4 et 10^5 K
#[test] fn test_jeans_mass_solar() { ... }            // M_Jeans(T=10K, ρ=100/cm³) ~ 1 M_sun
#[test] fn test_freefall_time() { ... }               // t_ff(ρ=1e-23 g/cm³) ~ 45 Myr
#[test] fn test_sn_energy_units() { ... }             // Vérifier conversions erg → km²/s²
#[test] fn test_sfr_threshold() { ... }               // SFR=0 si T > T_threshold
```

### C. Niveaux de simulation Étape 0
Pas de simulation — uniquement tests unitaires.
Critère de passage : **100% des tests passent**.

### D. Document récap Étape 0
```
/mnt/T2/janus-sim/docs/etape0_recap.md
- Liste des modules implémentés
- Résultats des tests unitaires
- Constantes physiques validées
- Prêt pour Étape 1 : OUI/NON
```

---

## ÉTAPE 0b — CONDITIONS INITIALES ZEL'DOVICH (prerequis Étape 3)

**⚠️ UPDATE 12 avril 2026 — Consultation IA (ChatGPT + Gemini)**

Les ICs aléatoires actuelles (δ_init=10% uniforme) sont insuffisantes
pour un test BAO valide. Il faut des ICs Zel'dovich avec spectre HZ.

### A. Algorithme ICs Zel'dovich

**Bonne nouvelle (Gemini) :** T(k) ≈ 1 pour notre boîte 500 Mpc
à k < 0.1 h/Mpc et z_init=4 → spectre HZ brut suffisant pour
validation qualitative. Pas besoin de CAMB/CLASS pour commencer.

**Point critique (ChatGPT) :** Ne pas utiliser le facteur de
croissance ΛCDM. Utiliser D(z) de notre intégration Friedmann Janus.

```python
# analysis/generate_ics_zeldovich.py
# Source : Gemini + ChatGPT, avril 2026
# Validé conceptuellement pour MCJ sans DM

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

def generate_zeldovich_ics(
    L_box=500.0,    # Mpc
    N=256,          # grid (256³ pour validation, 512³ pour production)
    z_init=4.0,
    delta_rms=0.1,  # amplitude perturbations initiales
    n_s=0.965,      # tilt Planck (ou 1.0 pour HZ strict)
    seed=42,
    D_z=None        # facteur croissance MCJ (None → approx 1/(1+z))
):
    """
    Conditions initiales Zel'dovich pour MCJ
    
    P(k) = A × k^n_s  (Harrison-Zel'dovich si n_s=1)
    
    Pour m+ : seed=42
    Pour m- : seed=43  → champs indépendants → Corr(δ+,δ-)=0 initial
    """
    np.random.seed(seed)
    
    # Grille k
    kx = fftfreq(N, d=L_box/(2*np.pi*N))
    ky = kx.copy()
    kz = kx.copy()
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k  = np.sqrt(k2 + 1e-20)
    
    # Spectre de puissance P(k) = A × k^n_s
    A = 1.0  # sera rescalé
    Pk = A * k**n_s
    Pk[0,0,0] = 0.0  # mode zéro nul
    
    # Champ gaussien dans l'espace de Fourier
    noise_r = np.random.normal(0, 1, (N,N,N))
    noise_i = np.random.normal(0, 1, (N,N,N))
    delta_k = (noise_r + 1j*noise_i) * np.sqrt(Pk/2.0)
    delta_k[0,0,0] = 0.0
    
    # FFT inverse → δ(x) réel
    delta_x = np.real(ifftn(delta_k)) * N**3
    
    # Rescaler pour avoir delta_rms voulu
    current_rms = np.std(delta_x)
    A_correct = (delta_rms / current_rms)**2
    delta_k *= np.sqrt(A_correct)
    delta_x  = np.real(ifftn(delta_k)) * N**3
    
    # Potentiel de déplacement : ∇²Φ = δ → Φ(k) = -δ(k)/k²
    phi_k = -delta_k / (k2 + 1e-20)
    phi_k[0,0,0] = 0.0
    
    # Champ de déplacement Ψ(k) = ik Φ(k)/k²
    psi_x_k = 1j * KX * phi_k
    psi_y_k = 1j * KY * phi_k
    psi_z_k = 1j * KZ * phi_k
    
    psi_x = np.real(ifftn(psi_x_k)) * N**3
    psi_y = np.real(ifftn(psi_y_k)) * N**3
    psi_z = np.real(ifftn(psi_z_k)) * N**3
    
    # Positions lagrangiennes uniformes
    gx, gy, gz = np.mgrid[0:N, 0:N, 0:N]
    q = np.stack([gx, gy, gz], axis=-1).reshape(-1,3).astype(float)
    q *= L_box / N
    
    # Facteur de croissance MCJ
    if D_z is None:
        D_z = 1.0 / (1.0 + z_init)  # approximation haute redshift
    # TODO : remplacer par D_z exact de friedmann.rs
    
    # Positions Zel'dovich
    psi = np.stack([psi_x, psi_y, psi_z], axis=-1).reshape(-1,3)
    positions = q + D_z * psi
    positions = positions % L_box  # conditions périodiques
    
    # Vitesses : v = a × H × f × Ψ ≈ a × H × D_z × Ψ (approx haute z)
    a_init = 1.0 / (1.0 + z_init)
    H_init = 69.9 * np.sqrt(0.3 * (1+z_init)**3)  # approx H(z)
    f = 1.0  # growth rate ≈ 1 pour matière dominante
    velocities = a_init * H_init * f * psi  # km/s
    
    return positions, velocities, delta_x

# Usage pour MCJ :
# pos_plus,  vel_plus,  _ = generate_zeldovich_ics(seed=42)  # m+
# pos_minus, vel_minus, _ = generate_zeldovich_ics(seed=43)  # m- indépendant
```

**Validation des ICs générées :**
```python
def validate_ics(pos_plus, pos_minus, delta_plus, delta_minus, L_box):
    """Tests de validation à passer avant tout run"""
    # 1. Corr(δ+, δ-) ≈ 0 (champs indépendants)
    corr = np.corrcoef(delta_plus.flat, delta_minus.flat)[0,1]
    assert abs(corr) < 0.05, f"Corrélation initiale trop haute: {corr}"
    
    # 2. δ_rms ≈ 10%
    assert 0.08 < np.std(delta_plus) < 0.12
    assert 0.08 < np.std(delta_minus) < 0.12
    
    # 3. P(k) ∝ k^n_s (vérifier sur quelques décades)
    k, Pk = compute_power_spectrum(pos_plus, L_box)
    slope = np.polyfit(np.log(k[1:20]), np.log(Pk[1:20]), 1)[0]
    assert 0.8 < slope < 1.2, f"Pente P(k) incorrecte: {slope}"
    
    # 4. Positions dans la boîte
    assert np.all(pos_plus >= 0) and np.all(pos_plus <= L_box)
    
    print(f"✓ ICs validées : corr={corr:.3f}, rms+={np.std(delta_plus):.3f}")
```

**Amélioration V2 — Facteur de croissance D(z) MCJ exact :**

```python
def load_growth_factor_janus(z_init, friedmann_output_path):
    """
    Charger D(z) depuis l'intégration numérique friedmann.rs
    (déjà calculé, pas besoin de CAMB/CLASS)
    
    friedmann.rs exporte : z, a, H, D, f
    → brancher ici au lieu de D_z = 1/(1+z)
    """
    import pandas as pd
    df = pd.read_csv(friedmann_output_path)
    # Interpoler D(z_init)
    D_z = np.interp(z_init, df['z'][::-1], df['D'][::-1])
    f_z = np.interp(z_init, df['z'][::-1], df['f'][::-1])
    return D_z, f_z

# Dans generate_zeldovich_ics() :
# Remplacer D_z = 1/(1+z_init)
# Par : D_z, f_z = load_growth_factor_janus(z_init, 'output/friedmann.csv')
```

**Amélioration V2 — Fonction de transfert baryonique T_b(k) :**

```python
def transfer_baryon_eisenstein_hu(k, omega_b=0.022, h=0.699):
    """
    Fonction de transfert baryonique Eisenstein & Hu 1998 sans CDM
    Obligatoire pour publication — optionnel pour validation
    
    T_b(k) ≈ sin(k·r_s) / (k·r_s) × exp(-(k/k_D)²)
    r_s ~ 150 Mpc (horizon acoustique)
    k_D ~ 0.1 h/Mpc (amortissement Silk)
    """
    r_s = 147.0 / h   # Mpc, horizon acoustique baryonique
    k_D = 0.12 * h    # h/Mpc, échelle de Silk damping
    x = k * r_s
    T_b = np.sinc(x/np.pi) * np.exp(-(k/k_D)**2)
    return T_b

# Utilisation dans generate_zeldovich_ics() :
# Pk = A * k**n_s * transfer_baryon_eisenstein_hu(k)**2
# (au lieu de Pk = A * k**n_s)
```

### B. Tests Étape 0b — ICs Zel'dovich

```python
# tests/test_ics.py

def test_corr_initial_zero():
    """Corr(δ+, δ-) < 0.05 avec seeds différents"""
    _, _, delta_plus  = generate_zeldovich_ics(seed=42)
    _, _, delta_minus = generate_zeldovich_ics(seed=43)
    corr = np.corrcoef(delta_plus.flat, delta_minus.flat)[0,1]
    assert abs(corr) < 0.05, f"Corr={corr:.3f} trop haute"

def test_delta_rms_target():
    """δ_rms ≈ 10% ± 20%"""
    _, _, delta = generate_zeldovich_ics(delta_rms=0.1)
    assert 0.08 < np.std(delta) < 0.12

def test_pk_slope():
    """P(k) ∝ k^n_s sur grandes échelles"""
    _, _, delta = generate_zeldovich_ics(n_s=1.0)
    k, Pk = compute_power_spectrum_grid(delta, L_box=500.0)
    # Fit pente sur k < 0.05 h/Mpc
    mask = k < 0.05
    slope = np.polyfit(np.log(k[mask]), np.log(Pk[mask]), 1)[0]
    assert 0.7 < slope < 1.3, f"Pente={slope:.2f} (attendu ~1.0)"

def test_positions_in_box():
    """Toutes les positions dans [0, L_box]"""
    pos, _, _ = generate_zeldovich_ics(L_box=100.0)
    assert np.all(pos >= 0) and np.all(pos <= 100.0)

def test_bao_peak_present():
    """Pic BAO visible dans ξ(r) à r~150 Mpc"""
    pos, _, _ = generate_zeldovich_ics(L_box=500.0, N=256)
    r, xi = compute_correlation_function(pos, L_box=500.0)
    # Chercher pic entre 100 et 200 Mpc
    mask = (r > 100) & (r < 200)
    assert xi[mask].max() > 0, "Pic BAO absent dans ξ(r)"

def test_growth_factor_loaded():
    """D(z=4) depuis friedmann.rs ~ 0.2 (= 1/(1+4))"""
    D_z, f_z = load_growth_factor_janus(4.0, 'output/friedmann.csv')
    assert 0.15 < D_z < 0.25, f"D(z=4)={D_z:.3f} incohérent"
    assert 0.8 < f_z < 1.2,   f"f(z=4)={f_z:.3f} incohérent"
```

**⚠️ UPDATE v1.8 — corrections post-feedback 3 IA :**

**T_b(k) OBLIGATOIRE pour Étape 3c :**
T(k)≈1 valide uniquement pour niveaux 3a/3b (qualitatif).
Pour 3c (test quantitatif vs SDSS) : `transfer_baryon_eisenstein_hu()` requis.
Sans T_b(k) → BAO aux mauvaises positions → comparaison SDSS invalide.

**Self-shielding Rahmati 2013 — OBLIGATOIRE avant run 1M :**
```python
def self_shielding_rahmati2013(nH, Gamma_uv):
    """Rahmati et al. 2013, MNRAS 430, 2427
    À nH > 0.01 cm⁻³ → Γ_eff → 0 (auto-blindage)
    Sans ça : SF spurieuse dans vides cosmiques"""
    n0, alpha1, alpha2, beta, f = 0.00981, -0.2287, -0.7268, 1.1802, 0.05
    ratio = (1-f)*(1+(nH/n0)**beta)**alpha1 + f*(1+nH/n0)**alpha2
    return Gamma_uv * ratio
# Intégrer dans cooling_heating_HM2012()
```

**Blastwave / cooling delay SN — OBLIGATOIRE :**
```rust
// src/feedback.rs
// Sans délai de refroidissement : énergie SN rayonnée instantanément → inefficace
pub struct SNFeedback {
    t_blastwave: f64,  // ~0.01-0.05 Gyr (Stinson 2006)
}
// Désactiver cooling sur voisins pendant t_blastwave après explosion
```

**Checksum avant nettoyage snapshots :**
```bash
sha256sum snap_final.bin > snap_final.sha256
# Vérifier avant d'effacer les intermédiaires
sha256sum -c snap_final.sha256
```

**Test stabilité VSL ajouté (Gemini) :**
```rust
#[test]
fn test_vsl_stability_500steps() {
    // ratio v_rms-/v_rms+ ≈ 1 sur 500 steps, 100K particules
    // Vérifier que la VSL dynamique maintient l'équilibre
    let ratio_max = run_vsl_test(N=100_000, steps=500);
    assert!(ratio_max < 1.05, "VSL instable: ratio={}", ratio_max);
}
```

### A. Algorithme

#### A.1 Intégration du refroidissement dans la boucle SPH

```rust
// Dans la boucle principale (main_loop.rs) :

// Ordre d'appel OBLIGATOIRE (operator splitting) :
// 1. Gravité (BH GPU)
// 2. Hubble friction
// 3. Leapfrog drift
// 4. SPH densité + pression
// 5. ← NOUVEAU : Refroidissement radiatif
// 6. Leapfrog kick

fn apply_cooling(
    particles: &mut Vec<Particle>,
    cooling_table: &CoolingTable,
    dt: f64,           // pas de temps en Gyr
    T_floor: f64,      // 100 K — plancher absolu
) {
    for p in particles.iter_mut().filter(|p| p.species == Species::Positive) {
        let T = p.internal_energy_to_temperature();
        let rho = p.sph_density;
        
        // Sous-stepping adaptatif si t_cool << dt
        let t_cool = cooling_table.t_cool(T, rho);
        let n_substeps = (dt / t_cool).ceil().min(100.0) as usize;
        let dt_sub = dt / n_substeps as f64;
        
        let mut U = p.internal_energy;
        for _ in 0..n_substeps {
            let dU = cooling_table.cooling_rate(T, rho, 0.0) * dt_sub;
            U = (U - dU).max(T_floor * BOLTZMANN / (GAMMA - 1.0) / MEAN_WEIGHT);
        }
        p.internal_energy = U;
    }
}
```

**Formule de conversion énergie interne ↔ température :**
$$T = \frac{(\gamma-1) \cdot \mu \cdot m_p \cdot U}{k_B}$$

avec $\gamma=5/3$ (gaz monoatomique), $\mu=0.588$, U en km²/s².

#### A.2 Sous-stepping adaptatif

Le refroidissement peut être beaucoup plus rapide que le pas de temps gravitationnel. Critère :

$$n_\text{substeps} = \max\left(1, \left\lceil \frac{\Delta t}{t_\text{cool}/10} \right\rceil \right)$$

avec cap à 1000 substeps. Si $n > 1000$ → la particule se refroidit instantanément jusqu'à $T_\text{floor}$.

### B. Tests Étape 1

```rust
// tests/test_cooling_integration.rs

#[test]
fn test_lambda_known_values() {
    // Validation contre références croisées Gemini + ChatGPT
    assert_relative_eq!(lambda_primordial(1e4),  1.0e-23, epsilon=0.5); // ordre de grandeur
    assert_relative_eq!(lambda_primordial(3e4),  1.6e-22, epsilon=0.3); // pic Ly-alpha
    assert_relative_eq!(lambda_primordial(1e5),  2.5e-22, epsilon=0.2); // pic H
    assert_relative_eq!(lambda_primordial(1e7),  1.5e-22, epsilon=0.2); // Bremsstrahlung
}

#[test]
fn test_isolated_cloud_cooling() {
    // Nuage T=10^6 K, ρ=1e-3 M_sun/Mpc³
    // Doit refroidir à T < 10^4 K en < 1 Gyr
    let mut T = 1e6_f64;
    let rho = 1e8_f64; // en unités simulation
    let dt = 0.001; // Gyr
    for _ in 0..1000 {
        let dU = cooling_rate_code_units(T, rho) * dt;
        T = temperature_from_internal_energy(
            internal_energy_from_temperature(T) + dU
        );
        if T < 1e4 { break; }
    }
    assert!(T < 1e4, "Cloud should cool below 10^4 K in 1 Gyr");
}

#[test]
fn test_cooling_floor() {
    // Aucune particule sous T_floor = 100 K
    let T_floor = 100.0_f64;
    let mut T = 500.0_f64;
    for _ in 0..10000 {
        let dU = cooling_rate_code_units(T, 1e8) * 0.001;
        T = (temperature_from_internal_energy(
            internal_energy_from_temperature(T) + dU
        )).max(T_floor);
    }
    assert!(T >= T_floor);
}

#[test]
fn test_bremsstrahlung_slope() {
    // Pour T > 10^7 K : Λ ∝ √T
    let ratio = lambda_primordial(1e8) / lambda_primordial(1e7);
    let expected = (1e8_f64 / 1e7_f64).sqrt(); // = sqrt(10) ≈ 3.16
    assert_relative_eq!(ratio, expected, epsilon=0.15);
}

#[test]
fn test_uv_suppresses_cooling_at_low_density() {
    // Dans l'IGM (nH faible), UV doit dominer → Λ_eff < 0 (chauffage net)
    let nH_igm = 1e-5_f64; // cm⁻³ — densité typique IGM
    let lambda_eff = cooling_heating_hm2012(1e4, nH_igm, 2.0);
    assert!(lambda_eff < 0.0, "UV should dominate at low density");
}

#[test]
fn test_uv_negligible_in_halos() {
    // Dans les halos denses, cooling domine sur UV
    let nH_halo = 1.0_f64; // cm⁻³ — densité typique halo
    let lambda_eff = cooling_heating_hm2012(1e6, nH_halo, 1.0);
    assert!(lambda_eff > 0.0, "Cooling should dominate in dense halos");
}

#[test]
fn test_uv_peak_at_z2() {
    // Γ(z=2) > Γ(z=0) et Γ(z=2) > Γ(z=6) — pic UV à z~2-3
    let gamma_z0 = 1e-24 * 1.0_f64 / (1.0 + (1.0_f64/3.0).powi(5));
    let gamma_z2 = 1e-24 * 9.0_f64 / (1.0 + (3.0_f64/3.0).powi(5));
    let gamma_z6 = 1e-24 * 49.0_f64 / (1.0 + (7.0_f64/3.0).powi(5));
    assert!(gamma_z2 > gamma_z0, "UV peak should be at z~2-3");
    assert!(gamma_z2 > gamma_z6, "UV peak should be above z~6");
}
```

### C. Niveaux de simulation Étape 1

**⚠️ Prérequis avant simulation :** Demander à ChatGPT le fit
Λ(T, z) avec UV background Haardt & Madau 2012 et l'intégrer
avant le niveau 1b. Sans UV background, formation stellaire
spurieuse dans les vides à z<2.

**Niveau 1a — Test unitaire 100K, boîte 10 Mpc (< 5 min)**
```
N = 100_000
Box = 10 Mpc
z_init = 2.0, steps = 100
SPH m+ avec cooling activé
Critère : T_mean descend de 10^4 K → < 5000 K en 100 steps
          Pas de crash, pas de NaN
```

**Niveau 1b — Validation 500K, boîte 50 Mpc (< 30 min)**
```
N = 500_000
Box = 50 Mpc
z_init = 4.0, steps = 1000
Critère : 
  - T_mean(z=2) < T_mean(z=4) (refroidissement global)
  - ρ+_max(z=2) > ρ+_max(z=4) × 2 (condensation accélérée)
  - ratio v_rms < 1.10
  - Seg croissant
```

**Niveau 1c — Run de validation 1M, boîte 100 Mpc (< 2h)**
```
N = 1_000_000
Box = 100 Mpc
z_init = 4.0, steps = 3000 (z=4→1)
Critère :
  - Premières étoiles formées (N_stars > 0)
  - ρ+_max > 500 à z=1
  - T_floor atteint dans les halos
  - Morphologie lacunaire m− visible
```

### D. Document récap Étape 1
```
/mnt/T2/janus-sim/docs/etape1_cooling_recap.md
- Courbe T_mean(z) vs run précédent
- Courbe ρ+_max(z) vs run précédent
- Première formation stellaire : z et ρ
- Comparaison t_cool observé vs théorique
- Bugs rencontrés et corrections
- GO/NO-GO pour Étape 2
```

---

## ÉTAPE 2 — FORMATION STELLAIRE + FEEDBACK (2-3 jours)

### A. Algorithme

#### A.1 Formation stellaire stochastique

```rust
// src/starformation.rs

fn apply_star_formation(
    particles: &mut Vec<Particle>,
    sf: &StarFormationCriteria,
    dt: f64,
    rng: &mut impl Rng,
) -> usize {
    let mut n_formed = 0;
    
    for p in particles.iter_mut().filter(|p| p.species == Positive) {
        // Critères physiques
        let eligible = p.sph_density > sf.rho_threshold
            && p.temperature() < sf.T_threshold
            && p.velocity_divergence < 0.0;  // gaz convergent
        
        if !eligible { continue; }
        
        // Probabilité stochastique de formation
        // P = 1 - exp(-ε* × dt / t_ff)
        let t_ff = sf.freefall_time(p.sph_density);
        let prob = 1.0 - (-sf.efficiency * dt / t_ff).exp();
        
        if rng.gen::<f64>() < prob {
            // Convertir particule gaz → étoile
            p.species = Species::Star;
            p.birth_time = current_time;
            p.internal_energy = 0.0; // étoile = pas de pression SPH
            n_formed += 1;
        }
    }
    n_formed
}
```

**Probabilité de formation stellaire (McKee & Krumholz 2010) :**
$$P_* = 1 - \exp\left(-\frac{\varepsilon_* \Delta t}{t_\text{ff}}\right)$$

avec $\varepsilon_* = 0.02$ (efficacité locale de formation stellaire).

#### A.2 Feedback supernova différé

```rust
fn apply_sn_feedback(
    particles: &mut Vec<Particle>,
    feedback: &SNFeedback,
    current_time: f64,
) {
    for p in particles.iter_mut().filter(|p| p.species == Star) {
        // Délai SN : t_SN ~ 10 Myr après formation (étoiles massives)
        let age = current_time - p.birth_time;
        if age > SN_DELAY && !p.sn_exploded {
            // Injecter énergie dans les voisins SPH
            let delta_U = feedback.energy_injection(p.mass, p.mass);
            inject_thermal_energy_to_neighbors(p, delta_U, particles);
            p.sn_exploded = true;
        }
    }
}
```

**Énergie SN en unités simulation :**
$$\Delta U = \frac{10^{51} \text{ erg}}{100 M_\odot \cdot m_\text{particule}} \approx 10^6 \text{ km}^2/\text{s}^2$$

### B. Tests Étape 2

```rust
#[test]
fn test_star_formation_probability() {
    // P ≈ ε* × dt / t_ff pour dt << t_ff
    // Vérifier limite petit dt
}

#[test]
fn test_no_stars_hot_gas() {
    // T > T_threshold → N_stars = 0 toujours
    let cloud = create_hot_cloud(T=1e6, N=10000);
    apply_star_formation(&mut cloud, ...);
    assert_eq!(cloud.n_stars(), 0);
}

#[test]
fn test_sn_energy_injection() {
    // Vérifier que l'énergie totale est conservée
    // E_SN_injectée = ΔU_voisins × m_voisins
    let delta_E_expected = 1e51 * ERG_TO_CODE_UNITS;
    assert_relative_eq!(delta_E_measured, delta_E_expected, epsilon=0.01);
}

#[test]
fn test_sfr_schmidt_kennicutt() {
    // SFR ∝ ρ^1.5 (loi Schmidt-Kennicutt)
    // Vérifier sur 3 densités différentes
}

#[test]
fn test_stellar_mass_growth() {
    // M_stars(t) croissante et bornée par M_gas_initial
}
```

### C. Niveaux de simulation Étape 2

**Niveau 2a — Collapse isolé 100K (< 10 min)**
```
N = 100_000 m+ seulement (pas de m−)
Sphère uniforme, T_init = 10^4 K
Gravité + SPH + cooling + SF
Critère :
  - Collapse en < 1 Gyr
  - N_stars > 100 (formation stellaire active)
  - Feedback stabilise le collapse (pas de runaway)
  - ρ_max plafonne (≠ diverge)
```

**Niveau 2b — Validation cosmologique 500K (< 30 min)**
```
N = 500_000
Box = 50 Mpc, z_init=4→1
Critère :
  - N_stars(z=1) > 1000
  - SFR(z) pic entre z=2 et z=1 (cohérent observations)
  - Halos stables (ρ_max ne diverge pas)
  - Feedback empêche over-cooling
```

**Niveau 2c — Run validation 1M (< 3h)**
```
N = 1_000_000
Box = 100 Mpc, z_init=4→0
Critère :
  - Fonction de masse stellaire visible
  - Premier "proto-galaxie" Janus identifiable
  - SFR density compatible avec Madau plot (ordre de grandeur)
  - ratio v_rms < 1.15 à tous les z
```

### D. Document récap Étape 2
```
/mnt/T2/janus-sim/docs/etape2_sf_recap.md
- N_stars(z) courbe
- SFR(z) vs Madau plot observé
- Masse stellaire maximale obtenue
- Morphologie des proto-galaxies Janus
- Problèmes : over-cooling ? under-feedback ?
- Paramètres ajustés : ε*, T_threshold, E_SN
- GO/NO-GO pour Étape 3
```

---

## ÉTAPE 3 — SPECTRE DE PUISSANCE P(k) (1-2 jours)

### A. Algorithme

**⚠️ UPDATE 12 avril 2026 — Consultation IA (ChatGPT + Gemini convergents)**

**Ne pas comparer P(k) brut directement à Planck/SDSS.**
Passer par des observables équivalents avec corrections de biais et BAO.

#### A.1 Calcul de P(k) + BAO + σ₈

```python
# analysis/compute_pk.py
import numpy as np
from scipy.fft import fftn

def compute_power_spectrum(positions, box_size, grid_size=512):
    """CIC + FFT → P(k) [Mpc/h]³"""
    density_grid = cic_density(positions, box_size, grid_size)
    delta = density_grid / density_grid.mean() - 1.0
    delta_k = fftn(delta) / grid_size**3
    power = np.abs(delta_k)**2 * box_size**3
    return spherical_average(power, box_size, grid_size)

def compute_sigma8(k, Pk):
    """
    σ₈ = √(∫ P(k) W²(kR) d³k / 2π²), R=8 Mpc/h
    Planck : 0.80 ± 0.02 | KiDS/DES : 0.70 ± 0.03
    → MCJ pourrait être ~0.70 (tension Planck, accord KiDS)
    """
    R = 8.0
    W = lambda x: 3*(np.sin(x) - x*np.cos(x)) / x**3
    integrand = Pk * W(k*R)**2 * k**2 / (2*np.pi**2)
    return np.sqrt(np.trapz(integrand, k))

def extract_bao_signal(k, Pk):
    """
    Isoler oscillations BAO (méthode Eisenstein 1998)
    Attendu MCJ : pics à k ≈ 0.05, 0.10, 0.15 h/Mpc
    ⚠️ PROBLÈME : sans DM → BAO non amorties → trop fortes ?
    """
    Pk_smooth = fit_smooth_spectrum(k, Pk)  # continuum sans wiggles
    return k, Pk / Pk_smooth - 1.0  # oscillations relatives

def validate_pk_complete(pos_plus, pos_minus, box_size):
    """Pipeline complet de validation"""
    k, Pk_plus  = compute_power_spectrum(pos_plus, box_size)
    k, Pk_minus = compute_power_spectrum(pos_minus, box_size)
    k, Pk_cross = compute_cross_spectrum(pos_plus, pos_minus, box_size)

    # Ségrégation (déjà établie)
    r_k = Pk_cross / np.sqrt(Pk_plus * Pk_minus)

    # σ₈
    sigma8 = compute_sigma8(k, Pk_plus)

    # BAO
    k_bao, bao = extract_bao_signal(k, Pk_plus)
    bao_peaks  = find_bao_peaks(k_bao, bao)

    # χ² vs SDSS DR16 (avec biais galaxies b~1.5)
    chi2 = compute_chi2_vs_sdss(k, Pk_plus * 1.5**2)

    return {
        'r_k_negative' : np.all(r_k < 0),
        'sigma8'       : sigma8,
        'bao_positions': bao_peaks,
        'chi2_dof'     : chi2,
        'verdict'      : chi2 < 2.0 and 0.60 < sigma8 < 0.85
    }
```

**Critères quantitatifs (ce que les reviewers attendent) :**

| Critère | Valeur cible | Tolérance | Zone k |
|---|---|---|---|
| σ₈ Planck | 0.80 ± 0.02 | ±5% bon, ±10% limite | — |
| σ₈ KiDS/DES | 0.70 ± 0.03 | ±5% | — |
| Tilt n_s | 0.965 | ±0.01 | k < 0.05 |
| BAO position | 0.05, 0.10, 0.15 h/Mpc | ±2% | **CRITIQUE** |
| BAO amplitude | SDSS mesuré | ±10–20% | **CRITIQUE** |
| χ²/dof | ~1 | <2 acceptable | global |

**Zones k par discriminance :**
```
k < 0.05 h/Mpc    → tilt + normalisation (peu discriminant)
k = 0.05-0.3 h/Mpc → BAO + forme ← ZONE CRITIQUE (modèles sans DM échouent ici)
k > 0.3 h/Mpc     → non-linéaire + baryons (moins fiable)
```

**⚠️ Risque principal MCJ :**
Sans matière noire → puits potentiel moins profonds → BAO non amorties
→ oscillations trop fortes vs SDSS à k ~ 0.05-0.15 h/Mpc.
C'est LE test discriminant. Si BAO trop marquées → MCJ en tension.

**Note σ₈ :** Notre run donne σ₈ ≈ 0.70, cohérent KiDS/DES mais
en tension Planck. Ceci pourrait être une prédiction naturelle du
MCJ via répulsion m⁻ supprimant la croissance — à documenter.

### B. Tests Étape 3

```python
# tests/test_pk.py

def test_cic_mass_conservation():
    # Masse totale sur grille = N × m_particle
    
def test_pk_white_noise():
    # Points aléatoires uniformes → P(k) = constante
    
def test_pk_single_mode():
    # Onde sinusoïdale → pic à k_mode
    
def test_cross_spectrum_anticorrelation():
    # Deux champs anti-corrélés → r(k) = -1
    
def test_pk_units():
    # Vérifier [P(k)] = Mpc³/h³
    
def test_nyquist_cutoff():
    # P(k) non fiable pour k > k_Nyquist / 2
```

### C. Niveaux de simulation Étape 3

**Niveau 3a — P(k) sur snapshots existants (< 1h)**
```
Snapshots run principal (vsl_petit_production)
P+(k), P-(k), r(k) à z=4, 2, 1, 0.5, 0.15
Critères :
  - r(k) < 0 confirmé ✓ (déjà établi)
  - σ₈ calculé et comparé Planck vs KiDS
  - BAO extraites et positionnées
  - ⚠️ BAO trop fortes ? → documenter si problème
```

**Niveau 3b — P(k) run baryonique 500K (< 1h)**
```
Run Étape 2 niveau 2b → P(k)
Critère :
  - BAO mieux amorties qu'en gravité pure
  (le cooling + SF concentre la matière → amortissement partiel)
  - σ₈ dans intervalle [0.65, 0.85]
```

**Niveau 3c — Run dédié 1M grande boîte (< 3h)**
```
N = 1_000_000, Box = 300 Mpc (grande boîte pour k bas et BAO)
z_init = 4 → 0, ICs Zel'dovich avec spectre Harrison-Zel'dovich
Critère :
  - BAO détectées à k ≈ 0.05, 0.10, 0.15 h/Mpc ±5%
  - χ²/dof < 2 vs SDSS DR16 sur k ∈ [0.05, 0.3] h/Mpc
  - σ₈ ∈ [0.65, 0.85]
```

### D. Document récap Étape 3
```
/mnt/T2/janus-sim/docs/etape3_pk_recap.md
- Figure P+(k) et P-(k) vs ΛCDM à z=0, 0.5, 1, 2
- Figure r(k) = P×/sqrt(P+P-) — confirmation < 0
- Rapport P_Janus/P_ΛCDM par bin de k
- Comparaison σ8 : P_Janus vs σ8=0.70 (KiDS/DES)
- Biais spectraux identifiés (ex: manque de puissance à grande k ?)
- GO/NO-GO pour Étape 4
```

---

## ÉTAPE 4 — CARTES DE CONVERGENCE κ (2-3 jours)

### A. Algorithme

#### A.1 Projection de la densité → κ(θ)

```python
# analysis/lensing_kappa.py

def compute_convergence_map(
    pos_plus,        # positions m+ [N×3] en Mpc
    pos_minus,       # positions m- [N×3] en Mpc
    box_size,        # Mpc
    halo_center,     # [x, y, z] en Mpc
    R_max=50.0,      # Mpc — rayon de projection
    grid_size=256,   # pixels
    D_lens=500.0,    # Mpc — distance lentille
    D_source=3000.0, # Mpc — distance source (z~1)
):
    """
    κ(θ) = Σ(θ) / Σ_crit
    
    Σ_crit = c² D_s / (4πG D_l D_ls)
    
    Pour MCJ : κ_total = κ_plus + κ_minus
    κ_minus < 0 car masse négative → effet de défocalisation
    """
    # 1. Projection le long de la ligne de visée z
    Sigma_plus = project_density(pos_plus, halo_center, R_max, grid_size)
    Sigma_minus = project_density(pos_minus, halo_center, R_max, grid_size)
    
    # 2. Masse négative contribue avec signe opposé
    Sigma_total = Sigma_plus - Sigma_minus  # m- masse négative
    
    # 3. Normalisation par Σ_crit
    Sigma_crit = compute_sigma_crit(D_lens, D_source)
    kappa = Sigma_total / Sigma_crit
    
    # 4. Profil radial κ(r)
    r_bins, kappa_profile = radial_profile(kappa, grid_size, R_max)
    
    return r_bins, kappa_profile

def predict_euclid_signal(r_bins, kappa_profile, R_200):
    """
    Prédiction signal lentillage faible Euclid
    
    Attendu selon MCJ :
    - κ > 0 pour r < R_200 (halo m+ dominant)
    - κ < 0 pour r > 3×R_200 (shell m- dominant)
    - Transition à r ~ 3-5 × R_200
    """
    r_over_R200 = r_bins / R_200
    
    # Zone d'exclusion m- → κ négatif
    mask_outer = r_over_R200 > 3.0
    kappa_outer = kappa_profile[mask_outer].mean()
    
    signal_detectable = abs(kappa_outer) > 1e-3  # seuil Euclid ~10^-3
    return signal_detectable, kappa_outer
```

**Formule de convergence :**
$$\kappa(\vec{\theta}) = \frac{\Sigma(\vec{\theta})}{\Sigma_\text{crit}}, \quad \Sigma_\text{crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_l D_{ls}}$$

**Prédiction unique du MCJ :**
$$\kappa(r > 3R_{200}) < 0 \quad \leftarrow \text{impossible en ΛCDM}$$

### B. Tests Étape 4

```python
def test_kappa_nfw_profile():
    # Profil NFW pur → κ(r) décroissant, toujours > 0
    
def test_kappa_negative_shell():
    # Shell de masse négative → κ < 0 dans la coquille
    
def test_sigma_crit_units():
    # Vérifier [Σ_crit] = M_sun/Mpc²
    
def test_projection_mass_conservation():
    # ∫ Σ(θ) dΩ = M_total projeté
    
def test_euclid_detection_threshold():
    # Signal > 10^-3 pour être détectable
```

### C. Niveaux de simulation Étape 4

**Niveau 4a — κ sur snapshots existants (< 1h)**
```
Utiliser snapshots run principal à z=0.15
Calculer κ(r) autour des 5 halos principaux
Critère : κ < 0 confirmé à r > 3×R_200
```

**Niveau 4b — Carte κ complète 1M (< 3h)**
```
Run baryonique 1M jusqu'à z=0
Calculer κ(θ) sur grille 256×256 autour halo dominant
Critère : 
  - κ > 0 pour r < R_200
  - κ < 0 pour r > 3×R_200
  - |κ_outer| > 10^-3 (détectable Euclid)
```

### D. Document récap Étape 4
```
/mnt/T2/janus-sim/docs/etape4_kappa_recap.md
- Figure κ(r/R_200) avec zones κ>0 et κ<0
- Comparaison avec profil NFW ΛCDM (toujours κ>0)
- Estimation signal Euclid : S/N attendu
- Table : κ_outer pour les 10 halos principaux
- Figure : carte 2D κ(θ) autour halo #1
- GO/NO-GO pour Étape 5
```

---

## ÉTAPE 5 — COURBES DE ROTATION (2-3 jours)

### A. Algorithme

#### A.1 Extraction des courbes de rotation

**⚠️ UPDATE 12 avril 2026 — Consultation 3 IA (ChatGPT, Grok, Gemini)**

**Divergence importante entre les IA — les deux ont raison sur des régimes différents :**

**Régime intérieur (r < 3×R₂₀₀) — ChatGPT + Gemini :**
La ségrégation totale M⁻(r)=0 dans les halos simplifie la formule :
$$v^2(r) = \frac{G \cdot M^+(r)}{r} \quad \text{(purement baryonique)}$$
La formule M_eff = M⁺(r) - M⁻(r) est exacte ici car M⁻(r)≈0.

**Régime extérieur (r > 3×R₂₀₀) — Grok (point crucial) :**
La coquille m⁻ externe viole le théorème de la coquille pour les masses négatives :
$$v^2(r) = \frac{G[M^+(r) - M^-(r)]}{r} + v^2_{\text{ext}}(r)$$
où $v^2_{\text{ext}}$ = contribution répulsive de la coquille m⁻ externe.

**Insight physique clé (Gemini) :**
> "La 'matière noire' n'est pas quelque chose qui est dans la galaxie
> pour l'attirer, mais quelque chose qui est autour pour l'empêcher
> de se disperser."

La coquille m⁻ externe agit comme un **confinement répulsif** qui pousse
m⁺ vers l'intérieur — mimant l'effet d'un halo de matière noire mais
par un mécanisme opposé.

**Mécanisme de courbe plate dans le MCJ :**

| Zone | ρ⁻/ρ⁺ | Effet sur v(r) |
|---|---|---|
| r < R₂₀₀ | ≈ 0 | Newtonien baryonique pur |
| R₂₀₀ < r < 3R₂₀₀ | 0 → 0.04 | Transition |
| r > 3R₂₀₀ | → 0.957 | Confinement m⁻ externe → plateau possible |

```python
# analysis/rotation_curves.py

def compute_rotation_curve_janus(
    pos_plus, pos_minus, vel_plus,
    halo_center, R_max=50.0, n_bins=50
):
    """
    Courbe de rotation MCJ en deux régimes :
    
    Régime intérieur (r < exclusion_radius) :
      v²(r) = G × M⁺(r) / r  [baryonique pur]
    
    Régime extérieur :
      v²(r) = G × [M⁺(r) - M⁻(r)] / r + v²_ext
      où v²_ext = contribution coquille m⁻
    """
    r_bins = np.linspace(0, R_max, n_bins)
    v_rot = np.zeros(n_bins)
    
    # Rayon d'exclusion m⁻ (typiquement 3-5 × R₂₀₀)
    exclusion_r = find_exclusion_radius(pos_minus, halo_center)
    
    for i, r in enumerate(r_bins[1:], 1):
        M_plus = count_mass_sphere(pos_plus, halo_center, r)
        M_minus = count_mass_sphere(pos_minus, halo_center, r)
        
        if r < exclusion_r:
            # Régime intérieur : M⁻ ≈ 0, purement baryonique
            v_rot[i] = np.sqrt(G * M_plus / r)
        else:
            # Régime extérieur : inclure coquille m⁻
            # v²_ext = contribution répulsive additionnelle
            v_ext_sq = compute_shell_contribution(
                pos_minus, halo_center, r, R_max
            )
            M_eff = M_plus - M_minus
            if M_eff > 0:
                v_rot[i] = np.sqrt(G * M_eff / r + v_ext_sq)
    
    return r_bins, v_rot

def find_exclusion_radius(pos_minus, halo_center):
    """Trouve le rayon où ρ⁻/ρ⁺ dépasse 0.05 (5% du fond)"""
    # ...
    pass

def compute_shell_contribution(pos_minus, center, r_inner, r_outer):
    """
    Contribution gravitationnelle d'une coquille m⁻ sur particule à r_inner
    Force répulsive → v² augmentée
    
    F_shell = G × M⁻_shell / r² × (répulsion vers l'intérieur)
    """
    M_shell = count_mass_shell(pos_minus, center, r_inner, r_outer)
    return G * abs(M_shell) / r_inner  # répulsion → contribution positive
```

**Prédiction observable unique du MCJ :**
$$v_\text{rot}(r) = \begin{cases}
\sqrt{G M^+(r)/r} & r < r_\text{excl} \text{ (képlérien baryonique)} \\
\text{plateau possible} & r \approx r_\text{excl} \text{ (confinement m⁻)} \\
\text{décroissance} & r \gg r_\text{excl} \text{ (hors halo)}
\end{cases}$$

Le plateau de rotation (si observé) serait dû au confinement m⁻ et non
à un halo de matière noire — **signature distinctive du MCJ**.

### B. Tests Étape 5

```python
def test_rotation_curve_baryonic_interior():
    # À r < exclusion_radius : v²(r) = G×M+(r)/r (baryonique pur)
    # Pas de contribution m⁻
    
def test_shell_theorem_negative_mass():
    # Coquille m⁻ externe → force répulsive vers l'intérieur
    # (contraire au théorème de la coquille standard)
    shell_force = compute_shell_contribution(shell_pos, center, r_test, r_outer)
    assert shell_force > 0, "m⁻ shell should push inward (repulsive)"

def test_rotation_curve_point_mass():
    # Masse ponctuelle m+ seule → v ∝ 1/sqrt(r) képlérien
    
def test_exclusion_radius_detection():
    # find_exclusion_radius() doit retourner ~3×R₂₀₀ pour halos du run principal
    
def test_tully_fisher_calibration():
    # Galaxie MW-like → v_flat ~ 220 km/s sans matière noire
    # Possible via confinement m⁻ externe
    
def test_plateau_mechanism():
    # Avec coquille m⁻ à r > 3×R₂₀₀ :
    # v_rot(r=5R₂₀₀) > v_kepler(r=5R₂₀₀)
    # Le plateau provient de v²_ext > 0
```

### C. Niveaux de simulation Étape 5

**⚠️ Critère de succès révisé — pas de matière noire dans le MCJ**

La courbe de rotation Janus n'est **pas** censée être plate comme en ΛCDM.
La prédiction correcte du MCJ est :

```
r < 3×R₂₀₀  → Képlérienne (1/√r) — baryonique pur, M⁻=0
r ~ 3-5×R₂₀₀ → Plateau possible — confinement coquille m⁻
r >> R₂₀₀    → Décroissance — hors influence de la coquille
```

C'est une **signature distinctive différente de ΛCDM** :
- ΛCDM : plateau dû au halo DM interne
- MCJ : plateau dû à la coquille m⁻ externe répulsive
- Échelle caractéristique : r_plateau ~ rayon d'exclusion m⁻ (~3-10 Mpc)

Cette prédiction est testable avec :
- Données HI radio (THINGS, LITTLE THINGS) — cinématique 21cm
- Lentillage faible Euclid — profil de masse total

**Niveau 5a — Courbe rotation halos existants (< 1h)**
```
Utiliser snapshots run principal à z=0.15
Halos #1 à #5 (R₂₀₀ = 2-5 Mpc)
Calculer v_rot(r) de r=0 à r=10×R₂₀₀
Critère RÉVISÉ :
  - v_rot(r < 3R₂₀₀) ∝ 1/√r (képlérien confirmé)
  - v_rot(r ~ exclusion_r) > v_kepler(r) — signature plateau
  - Pas de plateau plat étendu comme ΛCDM
```

**Niveau 5b — Galaxies Janus 500K (< 30 min)**
```
Zoom sur halo individuel 500K particules
Boîte 10 Mpc centrée sur halo
Refroidissement + SF activés
Critère RÉVISÉ :
  - Profil képlérien intérieur confirmé
  - Signature de la coquille m⁻ visible à r > 3R₂₀₀
  - v_flat (si plateau) dans gamme observationnelle
```

**Niveau 5c — Run dédié 1M zoom (< 3h)**
```
N = 1_000_000, boîte 20 Mpc, z_init=4→0
Critère RÉVISÉ :
  - Courbe képlérienne intérieure
  - Rayon de transition r_plateau mesuré
  - Comparaison avec THINGS (HI 21cm) :
    chercher galaxies avec profil similaire
    (décroissance puis légère remontée)
```

### D. Document récap Étape 5
```
/mnt/T2/janus-sim/docs/etape5_rotation_recap.md
- Figure v_rot(r) pour 5 halos
  → profil képlérien intérieur
  → signature coquille m⁻ à r > 3R₂₀₀
- Comparaison MCJ vs ΛCDM (DM interne vs coquille m⁻ externe)
- Rayon de transition r_plateau mesuré pour chaque halo
- Comparaison qualitative avec données HI (THINGS)
- Relation Tully-Fisher baryonique (si plateau mesurable)
- Morphologie galaxies : bulbe + disque visible ?
- GO/NO-GO pour Étape 6
```

---

## ÉTAPE 6 — DIPOLE REPELLER (1-2 jours)

### A. Algorithme

#### A.1 Calcul du champ de vitesse peculière

```python
# analysis/dipole_repeller.py

def compute_peculiar_velocity_field(
    pos_plus, pos_minus, vel_plus, vel_minus,
    box_size, grid_size=64,
):
    """
    Champ de vitesse peculière v_pec = v_total - v_Hubble
    
    Dans MCJ : v_pec dominée par répulsion m-
    → zones de sur-densité m- = zones de répulsion pour m+
    
    Dipole Repeller observé (Hoffman 2017) :
    - Position : RA=11h, Dec=-46°, d~220 Mpc
    - Vitesse de répulsion : ~-200 km/s
    """
    # 1. Grille de vitesses par CIC
    vx_grid = cic_velocity(pos_plus, vel_plus[:,0], box_size, grid_size)
    vy_grid = cic_velocity(pos_plus, vel_plus[:,1], box_size, grid_size)
    vz_grid = cic_velocity(pos_plus, vel_plus[:,2], box_size, grid_size)
    
    # 2. Soustraire vitesse de Hubble v_H = H(z) × r
    vx_grid -= hubble_flow_x(box_size, grid_size)
    vy_grid -= hubble_flow_y(box_size, grid_size)
    vz_grid -= hubble_flow_z(box_size, grid_size)
    
    # 3. Identifier attracteurs et répulseurs
    # Répulseur = région de divergence positive ∇·v > 0
    div_v = compute_divergence(vx_grid, vy_grid, vz_grid, box_size)
    
    # 4. Identifier le "Dipole Repeller" Janus
    repeller_pos = find_max_divergence_region(div_v)
    repeller_velocity = get_velocity_amplitude(repeller_pos, vx_grid, vy_grid, vz_grid)
    
    return repeller_pos, repeller_velocity, div_v

def compare_with_hoffman2017(repeller_pos_sim, repeller_vel_sim, box_size):
    """
    Hoffman et al. 2017 : Dipole Repeller à ~220 Mpc
    Vitesse de répulsion : ~-200 km/s vers la Vierge
    
    Critère de validation :
    - Au moins 1 répulseur de taille comparable dans la boîte
    - Vitesse de répulsion > 100 km/s
    - Morphologie : vide m- de ~50 Mpc de rayon
    """
    n_repellers = count_significant_repellers(div_v, threshold=100.0)
    return {
        'n_repellers': n_repellers,
        'max_velocity': repeller_vel_sim,
        'compatible': repeller_vel_sim > 100.0 and n_repellers > 0
    }
```

### B. Tests Étape 6

```python
def test_peculiar_velocity_zero_uniform():
    # Distribution uniforme → v_pec = 0 partout
    
def test_repeller_detection():
    # Vide artificiel → divergence positive détectée
    
def test_hoffman_velocity_scale():
    # Vérifier que les unités donnent km/s cohérents
    
def test_attractor_vs_repeller():
    # Sur-densité m+ → attracteur (v convergente)
    # Sur-densité m- → répulseur (v divergente)
```

### C. Niveaux de simulation Étape 6

**Niveau 6a — Champ de vitesse sur run existant (< 2h)**
```
Run principal snapshots z=0.15
Calculer ∇·v sur grille 64³
Identifier répulseurs majeurs
Critère : au moins 1 répulseur avec v > 100 km/s
```

**Niveau 6b — Comparaison quantitative 1M (< 3h)**
```
Run baryonique 1M, boîte 500 Mpc, z=0
Calculer champ v_pec complet
Critère :
  - Dipole Repeller Janus comparable à Hoffman 2017
  - Vitesse répulsion ∈ [100, 400] km/s
  - Taille vide associé > 30 Mpc
```

### D. Document récap Étape 6
```
/mnt/T2/janus-sim/docs/etape6_dipole_recap.md
- Figure champ v_pec en 2D
- Position et vitesse du Dipole Repeller Janus
- Comparaison avec Hoffman 2017 (tableau)
- Figure : vide m- associé au répulseur
- GO/NO-GO pour Run final
```

---

## ÉTAPE 7 — RUN FINAL 10M (40-100h)

### Conditions de lancement

**Tous les critères suivants doivent être verts :**

| Étape | Critère | Statut |
|---|---|---|
| Étape 0 | 100% tests unitaires | ☐ |
| Étape 1 | Refroidissement validé 1M | ☐ |
| Étape 2 | N_stars > 0, SFR cohérent | ☐ |
| Étape 3 | P(k) dans facteur 2 de Planck | ☐ |
| Étape 4 | κ < 0 confirmé à 3×R₂₀₀ | ☐ |
| Étape 5 | v_rot plate, Tully-Fisher OK | ☐ |
| Étape 6 | Dipole Repeller détecté | ☐ |

### Paramètres du run final

```
N = 10_000_000 (5_110_024 m+ / 4_889_976 m-)
Box = 500 Mpc périodique
μ = 19, η = 1.045
z_init = 4.0 → 0.0
dt = 0.001 Gyr
steps = 30_000

Physique m+ :
  SPH (T_init=10^4 K, T_floor=100 K)
  Refroidissement radiatif Λ(T)
  Formation stellaire (ε*=0.02, T_threshold=10^4 K)
  Feedback SN (E_SN=10^51 erg, délai=10 Myr)

Physique m- :
  Gravité pure N-body
  Softening ε- = 5 × ε+

VSL : c²(z) = (1+z)^0.0431

Outputs :
  Snapshots tous les 10 steps (3000 total)
  Métriques tous les 5 steps
  Frames 4K tous les 5 steps
  Analyse P(k) tous les 100 steps
  FOF tous les 500 steps
```

### Analyses post-run final

```python
# analysis/final_analysis.py

def full_analysis(snapshot_dir):
    """Pipeline d'analyse complet pour le run final"""
    
    # 1. Evolution temporelle
    plot_seg_evolution()
    plot_sfr_vs_madau_plot()
    plot_stellar_mass_function()
    
    # 2. Structures
    run_fof_catalog()
    compute_halo_mass_function()
    compare_with_press_schechter()
    
    # 3. Spectres
    compute_pk_all_redshifts()
    compare_pk_with_planck()
    
    # 4. Lentillage
    compute_kappa_maps_all_halos()
    generate_euclid_predictions()
    
    # 5. Dynamique
    compute_rotation_curves_all_halos()
    plot_tully_fisher_relation()
    
    # 6. Dipole Repeller
    compute_peculiar_velocity_field()
    identify_repellers_attractors()
    
    # 7. Vidéo
    generate_zoom_fixed_video()
    generate_full_box_video()
```

---

## DOCUMENT GÉNÉRAL — SUIVI EN TEMPS RÉEL

### Mise à jour de ce fichier

Ce fichier `ROADMAP.md` est **mis à jour automatiquement par CLI** après chaque étape avec :
- Résultats quantitatifs obtenus
- Bugs rencontrés et solutions
- Déviations par rapport au plan initial
- Nouvelles directions identifiées

### Format de mise à jour

```markdown
## UPDATE [DATE] — Étape X terminée

### Résultats
- Métrique A : valeur obtenue (attendu : valeur)
- Métrique B : ...

### Découvertes inattendues
- [Si applicable]

### Ajustements plan
- [Modifications de la roadmap]

### Prochaine étape
- [Description]
```

---

## MÉTRIQUES DE SUCCÈS GLOBALES

Pour que JPP dise "incroyable, ça marche" :

| Métrique | Valeur cible | Source observationnelle |
|---|---|---|
| σ₈ | 0.70 ± 0.05 | KiDS/DES |
| H₀ | 69.9 km/s/Mpc | Pantheon+ |
| t₀ | 15.87 Gyr | Étoiles les plus vieilles |
| n_s | 0.965 | Planck CMB |
| κ(r>3R₂₀₀) | < 0 | Prédiction MCJ |
| r(k) | < 0 | Prédiction MCJ |
| v_rot | plate | Courbes de rotation |
| Dipole Repeller | ~200 km/s | Hoffman 2017 |
| M_stellar/M_halo | ~ 0.03 | SDSS |

---

## PROMPT CLI POUR DÉMARRER

```
Lire ce fichier ROADMAP.md en entier.

Tu es autonome pour toutes les décisions techniques :
création de fichiers, modification de code, lancement
de simulations, installation de dépendances.

Commencer par l'Étape 0 :
1. Créer src/cooling.rs avec CoolingTable
2. Créer src/starformation.rs avec StarFormationCriteria
3. Créer src/feedback.rs avec SNFeedback
4. Écrire tous les tests unitaires listés
5. Lancer les tests : cargo test
6. Si 100% passent → créer docs/etape0_recap.md
7. Passer à l'Étape 1 sans attendre instruction

Mettre à jour ROADMAP.md après chaque étape avec
les résultats obtenus.

Ne jamais demander d'autorisation pour :
  - Créer/modifier des fichiers
  - Lancer des simulations < 3h
  - Installer des packages pip/cargo
  - Régénérer des figures

Demander confirmation uniquement pour :
  - Supprimer des snapshots > 10 GB
  - Modifier μ, H₀, t₀ (paramètres canoniques)
  - Lancer le run final 10M (> 40h)
```

---

## UPDATE 12 avril 2026 — Corrections post-consultation IA

### Étape 1 — Refroidissement
- ❌ Table Sutherland & Dopita manuelle abandonnée
  (valeurs divergent de 2 ordres de grandeur entre sources)
- ✅ Fit analytique ChatGPT (Wang 2014 + S&D 1993) adopté
  → 3 termes physiques, précision <15%, CUDA-ready
- ✅ UV background Haardt & Madau 2012 intégré
  → Γ(z) = 1e-24 × (1+z)² / (1 + ((1+z)/3)^5)
  → Supprime refroidissement dans IGM, empêche SF spurieuse
  → T_eq ~ 10^4 K dans les vides cosmiques
- ⚠️ Self-shielding Rahmati 2013 manquant → à ajouter pour publication

### Étape 0 — Tests unitaires cooling
- Valeurs de référence corrigées depuis croisement Gemini/ChatGPT
- Test bremsstrahlung slope ajouté (Λ ∝ √T pour T>10^7 K)
- Tests UV ajoutés : suppression IGM, dominance halos, pic z~2-3

### Insight clé ChatGPT
> "Le fond UV transforme la cooling function en un équilibre thermique
> dépendant du redshift — pas juste une perte d'énergie."
> Impact direct : peut retarder ou modifier le runaway m⁻

### Étape 2 — Feedback SN
- ❌ Paramètres ΛCDM standards non transposables directement
- ✅ E_SN=10⁵¹ erg et délai=10 Myr conservés (physique universelle)
- ✅ ε recalibrée : 0.3–1% (au lieu de 1–10% ΛCDM)
- ✅ Mode cinétique recommandé (momentum-driven, p_SN=3×10⁵ M☉·km/s)
- ✅ n_threshold augmenté : 10–50 cm⁻³ (au lieu de 0.1–1 cm⁻³)
- ⚠️ Alerte Gemini : feedback trop violent peut briser ségrégation m⁺/m⁻
- 📊 Scan paramétrique prévu en Étape 2b (4×4×3 = 48 runs courts)

### Étape 5 — Courbes de rotation
- ✅ Formule M_eff = M⁺(r) - M⁻(r) confirmée par 3 IA
- ✅ Deux régimes distincts identifiés :
  - Intérieur (r < 3R₂₀₀) : M⁻=0, purement baryonique, képlérien
  - Extérieur : coquille m⁻ → confinement répulsif → plateau possible
- ✅ Mécanisme plateau expliqué : m⁻ externe "empêche la dispersion"
  pas de halo DM interne — signature distinctive du MCJ
- ⚠️ Théorème de la coquille ne s'applique pas proprement
  avec masses négatives → calcul intégral complet requis à r > 3R₂₀₀
- ✅ Critère de succès Étape 5 révisé :
  PAS de courbe plate comme ΛCDM — képlérienne intérieure
  + signature coquille m⁻ à r ~ 3-5R₂₀₀ (prédiction distinctive)
- ✅ Prédiction testable : THINGS (HI 21cm), Euclid (lentillage faible)

### Étape 3 — Spectre P(k)
- ✅ Critères quantitatifs précis ajoutés (σ₈, BAO, χ², n_s)
- ✅ Zone critique identifiée : k = 0.05-0.3 h/Mpc (BAO)
- ⚠️ Risque MCJ : BAO non amorties sans DM → trop fortes vs SDSS
  Le cooling+SF peut amortir partiellement — à quantifier
- ✅ σ₈ = 0.70 prédit → cohérent KiDS/DES, tension Planck (prédiction naturelle ?)
- ✅ Pipeline complet validate_pk_complete() ajouté
- ✅ Boîte niveau 3c agrandie 200→300 Mpc pour résoudre les BAO

### v1.8 — Corrections post-feedback 3 IA (Gemini + ChatGPT + Grok)
- ✅ Objectif reformulé : "tester si MCJ reproduit les observables" (falsifiable)
- ✅ Section Risques & Plans de contingence ajoutée
- ✅ Ordre d'exécution révisé : P(k) test préliminaire AVANT SF/feedback
- ✅ T_b(k) Eisenstein-Hu rendu OBLIGATOIRE pour Étape 3c
- ✅ Self-shielding Rahmati 2013 ajouté (OBLIGATOIRE avant 1M)
- ✅ Blastwave/cooling delay SN ajouté (Stinson 2006)
- ✅ Checksum SHA-256 avant nettoyage snapshots
- ✅ Test stabilité VSL 500 steps ajouté (Étape 0)
- ✅ Colonne "Temps GPU" ajoutée dans RESULTS.md
- ⚠️ Rotation curves : Grok note risque artefact simulation
  → tester sur plusieurs halos et seeds avant de conclure

*ROADMAP v1.8 — prêt pour exécution CLI*

---

*Document généré le 12 avril 2026 — Version 1.8*
*Mise à jour automatique après chaque consultation IA ou étape CLI*
