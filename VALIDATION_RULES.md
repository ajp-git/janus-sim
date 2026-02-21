# VALIDATION_RULES.md
# Règles de validation obligatoires — Projet Janus
# À lire au début de chaque session Claude CLI

## RÈGLE FONDAMENTALE

**Toute nouvelle fonction physique doit être validée sur un cas trivial
avec résultat connu AVANT d'être utilisée dans une simulation.**

Ordre obligatoire :
1. Écrire le test trivial
2. Vérifier que le test passe
3. Seulement ensuite utiliser la fonction
4. Ne jamais bypasser cette procédure même sous pression de temps

---

## 1. SÉGRÉGATION

### Test obligatoire
```rust
#[test]
fn test_segregation_trivial() {
    // 4 particules + à droite, 4 particules - à gauche
    // COM+ = (10, 0, 0), COM- = (-10, 0, 0)
    // Distance attendue = 20.0 EXACTEMENT
    let positions_plus  = vec![(10.0, 0.0, 0.0); 4];
    let positions_minus = vec![(-10.0, 0.0, 0.0); 4];
    let distance = compute_segregation(&positions_plus, &positions_minus);
    assert!((distance - 20.0).abs() < 1e-10,
        "Distance COM attendue 20.0, obtenu {}", distance);
}
```

### Normalisation obligatoire
```rust
// La ségrégation doit TOUJOURS être normalisée par box_size
let segregation = distance_com / box_size;
// Valeurs typiques : 0.0 (mélangé) à ~0.5 (ségrégé)
// Phase 1b validée : Seg₀ ≈ 0.029, Seg_final ≈ 0.163 (+294%)
```

### Piège connu — Conditions périodiques
```
ERREUR CLASSIQUE : Calculer COM avec moyenne simple des positions
quand les particules wrappent autour de la boîte.

EXEMPLE DU BUG :
  Boîte : half_size = 50
  Particule A : x = +49
  Particule B : x = -49
  Moyenne naive : (49 + (-49)) / 2 = 0  ← FAUX
  Distance réelle : 2                    ← CORRECT

SOLUTION : Algorithme "minimum image convention"
```

```rust
fn minimum_image(dx: f64, box_size: f64) -> f64 {
    let mut d = dx % box_size;
    if d > box_size / 2.0  { d -= box_size; }
    if d < -box_size / 2.0 { d += box_size; }
    d
}

fn compute_com_periodic(positions: &[(f64,f64,f64)], box_size: f64) -> (f64,f64,f64) {
    // Utiliser position de la première particule comme référence
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

### Test minimum image obligatoire
```rust
#[test]
fn test_minimum_image() {
    let box_size = 100.0;
    // Deux particules "proches" à travers la frontière périodique
    assert!((minimum_image(49.0 - (-49.0), box_size) + 2.0).abs() < 1e-10,
        "Distance périodique attendue -2.0");
    assert!((minimum_image(1.0, box_size) - 1.0).abs() < 1e-10,
        "Distance normale attendue 1.0");
}
```

---

## 2. CONSERVATION D'ÉNERGIE

### Test obligatoire
```rust
#[test]
fn test_energy_conservation_2body() {
    // 2 particules de même signe, loin l'une de l'autre
    // KE initial = KE final à 0.01% près sur 100 steps
    let ke_initial = simulate_2body(steps=100);
    let ke_final = ...;
    let drift = (ke_final - ke_initial).abs() / ke_initial;
    assert!(drift < 1e-4,
        "Dérive énergie {:.4}% > 0.01%", drift * 100.0);
}
```

### Seuils de validation par simulation
| Simulation | KE/KE₀ max acceptable | Ségrégation attendue |
|------------|----------------------|----------------------|
| 2 corps    | 1.001 (0.1%)         | Non applicable       |
| 100K test  | < 50 à step 50       | Doit augmenter       |
| 1M Phase1b | 113 à step 200       | +294% validé         |
| 5M Phase1c | < 200 à step 100     | À déterminer         |

---

## 3. FORCES JANUS

### Test obligatoire — Lois d'interaction
```rust
#[test]
fn test_janus_force_laws() {
    let origin = [0.0, 0.0, 0.0];
    let right   = [1.0, 0.0, 0.0];

    // Même signe → ATTRACTION (force vers +x)
    let f_pp = janus_force(origin, right, sign_i=+1, sign_j=+1);
    assert!(f_pp[0] > 0.0, "Masse+ attire Masse+ vers +x");

    let f_mm = janus_force(origin, right, sign_i=-1, sign_j=-1);
    assert!(f_mm[0] > 0.0, "Masse- attire Masse- vers +x");

    // Signes opposés → RÉPULSION (force vers -x)
    let f_pm = janus_force(origin, right, sign_i=+1, sign_j=-1);
    assert!(f_pm[0] < 0.0, "Masse+ repousse Masse- vers -x");

    let f_mp = janus_force(origin, right, sign_i=-1, sign_j=+1);
    assert!(f_mp[0] < 0.0, "Masse- repousse Masse+ vers -x");

    // Magnitude : G*m/r² (softening négligeable à r=1)
    let expected = G * 1.0 / 1.0;
    assert!((f_pp[0] - expected).abs() / expected < 0.01,
        "Magnitude incorrecte");
}
```

---

## 4. CONDITIONS AUX LIMITES PÉRIODIQUES

### Test obligatoire
```rust
#[test]
fn test_periodic_bc() {
    let box_half = 50.0;
    
    // Particule qui sort à droite
    let mut x = 51.0;
    apply_periodic(&mut x, box_half);
    assert!((x - (-49.0)).abs() < 1e-10, "Wrap droite : {}", x);

    // Particule qui sort à gauche  
    let mut x = -51.0;
    apply_periodic(&mut x, box_half);
    assert!((x - 49.0).abs() < 1e-10, "Wrap gauche : {}", x);

    // Particule dans la boîte — ne doit pas bouger
    let mut x = 10.0;
    apply_periodic(&mut x, box_half);
    assert!((x - 10.0).abs() < 1e-10, "Pas de wrap : {}", x);
}
```

---

## 5. INTÉGRATION FRIEDMANN

### Test obligatoire — Solution paramétrique exacte
```rust
#[test]
fn test_friedmann_vs_parametric() {
    // Comparer l'intégrateur RK4 avec la solution exacte
    // a(u) = α²·ch²(u) pour η = 1.045
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let history = integrate_backward(&params, 2.0, 5000);

    // À z=1 (a=0.5), comparer avec solution paramétrique
    // Écart acceptable : < 1%
    let a_numerical = interpolate_a(&history, z=1.0);
    let a_parametric = compute_parametric(eta, z=1.0);
    let error = (a_numerical - a_parametric).abs() / a_parametric;
    assert!(error < 0.01, "Écart Friedmann/paramétrique {:.2}%", error*100.0);
}
```

---

## 6. FORMULE SNIa (mu_janus_for_fit)

### Test obligatoire — Cohérence avec résultat connu
```rust
#[test]
fn test_mu_janus_known_values() {
    // Valeurs de référence : fit Pantheon+ validé
    // η = 1.045, q0 = -0.022, cst = 23.856

    let eta = 1.045;
    let cst = 23.856;

    // À z=0.5, μ doit être ≈ 42.3 (from janus_exact_fit.json)
    let mu = mu_janus_for_fit(0.5, eta, cst);
    assert!((mu - 42.3).abs() < 0.1,
        "μ(z=0.5) attendu ~42.3, obtenu {}", mu);

    // Monotonicité : μ doit croître avec z
    for z in [0.1, 0.5, 1.0, 2.0] {
        let mu_prev = mu_janus_for_fit(z * 0.9, eta, cst);
        let mu_curr = mu_janus_for_fit(z, eta, cst);
        assert!(mu_curr > mu_prev, "μ non monotone à z={}", z);
    }
}
```

---

## 7. COSMOINTERPOLATOR — Synchronisation Friedmann ↔ N-corps

### Contexte
La Tâche 2 (Hubble friction) nécessite de passer a(t) et H(t)
au kernel CUDA à chaque step. Ces valeurs viennent de friedmann.rs
mais les unités de temps sont différentes.

### Convention confirmée dans friedmann.rs (ligne 106)
```rust
let a_dot = params.omega_plus.sqrt();
// H₀ = √Ω₊ en unités adimensionnelles
```
Pour η=1.045 : H₀ = √(1/2.045) = 0.6993

### Test obligatoire avant Tâche 2
```rust
#[test]
fn test_cosmo_interpolator() {
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, 50.0);

    // Passé : a = 1/(1+z_init)
    let (a_start, _) = cosmo.get_params_at_tau(cosmo.tau_start);
    assert!((a_start - 1.0/51.0).abs() < 1e-4);

    // Présent : a = 1.0 exactement
    let (a_end, h_end) = cosmo.get_params_at_tau(cosmo.tau_end);
    assert!((a_end - 1.0).abs() < 1e-4);

    // H₀ = √Ω₊ (convention friedmann.rs ligne 106)
    assert!((h_end - params.omega_plus.sqrt()).abs() < 1e-4);

    // Monotonie
    let tau_mid = cosmo.tau_start + (cosmo.tau_end - cosmo.tau_start) / 2.0;
    let (a_mid, _) = cosmo.get_params_at_tau(tau_mid);
    assert!(a_mid > a_start && a_mid < a_end);
}
```

### Piège connu — Fonctions fantômes
```
Ces fonctions N'EXISTENT PAS dans friedmann.rs :
  integrate_friedmann(eta, t)   ← INEXISTANT
  interpolate_a(&history, t)    ← INEXISTANT

Utiliser CosmoInterpolator::new() et get_params_at_tau()
(code dans janus_roadmap.md Tâche 2)
```

### Piège connu — Choc des unités de temps
```
t_nbody (dt=0.01 arbitraire) ≠ τ cosmologique (H₀·t)

Conversion obligatoire :
  dtau_cosmo = (tau_end - tau_start) / total_nbody_steps

Ne jamais passer t_nbody directement à get_params_at_tau()
```

---

## 8. MÉTHODE DE FORCE N-CORPS

### Test de méthode — Avant toute simulation > 100K particules
```
La méthode de force doit être validée sur un cas physique simple :

TEST OBLIGATOIRE : Effondrement gravitationnel
  - 1000 particules de même signe, distributions sphérique
  - Après 50 steps : les particules doivent se rapprocher du centre
  - Rayon moyen doit diminuer de > 5%

TEST ANTI-PM : Ségrégation sur 100K
  - 50K particules+, 50K particules-
  - Après 200 steps : ségrégation doit augmenter de > 10%
  - Si ségrégation stagne : méthode PM inutilisable pour Janus
  - → Utiliser Barnes-Hut obligatoirement

MÉTHODES VALIDÉES pour Janus :
  ✅ Barnes-Hut CPU (Rust) — Phase 1b validée
  ✅ Barnes-Hut GPU (Rust+CUDA) — validé 0% écart
  ❌ Particle-Mesh (Python+CuPy) — ségrégation nulle, inutilisable
```

---

## 8. PROCÉDURE DE LANCEMENT SIMULATION

### Checklist obligatoire avant tout lancement > 1M particules

```
□ 1. Paramètres testés sur 100K, 50 steps
     → KE/KE₀ < 50 à step 50 ✓
     → Ségrégation augmente entre step 0 et step 50 ✓

□ 2. Seeds synchronisées CPU/GPU si comparaison
     → Écart ségrégation < 5% ✓

□ 3. Estimation temps total calculée
     → steps × temps/step = durée acceptable ✓

□ 4. Espace disque vérifié
     → snapshots NPZ + frames PNG ✓

□ 5. nohup + PID sauvegardé

□ 6. Surveillance steps 20, 50, 100 planifiée
     → Critères d'arrêt définis AVANT le lancement

□ 7. NE PAS LANCER sans validation explicite
```

---

## 9. RÈGLES DE COMPORTEMENT CLAUDE CLI

```
1. Ne jamais lancer une simulation sans instruction explicite.

2. Toujours rapporter step 20 et attendre confirmation
   avant de continuer au-delà de step 50.

3. Si un critère d'arrêt est atteint → arrêter et signaler.
   Ne pas "laisser tourner pour avoir les frames".

4. Tout changement de paramètre (softening, dt, méthode)
   par rapport à la version validée doit être signalé
   et re-validé sur 100K avant application.

5. Ne jamais interpréter un résultat ambigu comme "correct".
   Toujours poser la question explicitement.

6. Toute fonction de mesure physique (ségrégation, énergie,
   COM, distance) doit avoir son test trivial dans le code.
```

---

## HISTORIQUE DES BUGS RENCONTRÉS

| Bug | Cause | Leçon |
|-----|-------|-------|
| Équations accélération incorrectes | Densités locales au lieu de E conservé | Toujours vérifier vs papier source |
| Écart analytique/numérique 0.8 mag | H(z) standard Friedmann mixé avec acc Janus | Cohérence théorique avant implémentation |
| Ségrégation nulle avec PM | Méthode PM lisse interactions courte portée | Valider méthode sur cas physique connu |
| OOM 10M f64 sur 12GB VRAM | Estimation VRAM incorrecte (arbre ×8) | Calculer VRAM avant de lancer |
| COM faux avec conditions périodiques | Moyenne simple ignore wrap | Toujours utiliser minimum image convention |
| Lancement sans instruction | Claude CLI autonome | Checklist obligatoire avant lancement |
