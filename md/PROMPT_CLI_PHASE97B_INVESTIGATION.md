# PROMPT CLI — Phase 9.7-B : Investigation du 11% résiduel TreePM CPU

**Mission** : identifier la cause du 11.25% d'erreur médiane qui persiste après Phase 9.6 (splitting fix + r_s/r_cut canonical) et 9.7-A (grad2 confirmé meilleur que grad4). Localiser le bug latent par tests systématiques, **sans modifier le code de production sans validation**.

**Contexte** :
- Phase 9.5 : 16.61% médiane → bug splitting Tree polynomial vs PM Gaussian identifié
- Phase 9.6 : 11.25% médiane après fix splitting + r_s = r_cut/5
- Phase 9.7-A : grad4 amplifie le bruit, grad2 OK → la cause du 11% est ailleurs
- **Restant** : un quatrième bug à identifier
- **Constraint critique** : pivot Barnes-Hut impossible (résonance octree, pic k=L/8 à 20 884× les voisins, rédhibitoire pour préprint). TreePM est la seule voie.

**Branche** : `feat/treepm-jpp-port`. Commits atomiques par sous-étape.

---

## Règles d'opération

- **Mode autonome** sauf critères d'arrêt §6
- **Pas de modification du code production** sauf si une étape valide explicitement le besoin et propose un fix testable
- **Tous les tests sont diagnostiques** (`#[ignore]` par défaut) jusqu'à validation finale
- **Logs** dans `logs/treepm/phase97b_$(date +%F).log`
- **Référence** : test `test_treepm_combined_vs_pp_direct_precision` (N=1000, L=100, seed=42) reste la mesure principale

---

## Étape 1 — Test de convergence en N_pm (15 min)

Phase 9.6 a déjà mesuré ce plateau, mais sur un seul exécutable. Re-vérifier sur état actuel du code (post-Phase 9.7-A) :

```rust
#[test]
#[ignore]
fn diagnostic_convergence_in_n_pm_post_97a() {
    // Setup identique au test principal mais N_pm varie
    let n = 1000;
    let l = 100.0_f64;
    let seed = 42;
    let softening = 0.05_f64;
    let g_phys = 1.0_f64;
    
    println!("\n=== Convergence en N_pm (post Phase 9.7-A) ===");
    println!("N_pm | dg     | r_s    | r_cut  | median  | P95     | max");
    println!("-----|--------|--------|--------|---------|---------|--------");
    
    for &n_pm in &[32_usize, 64, 128, 256] {
        let dg = l / n_pm as f64;
        let r_s = 1.2 * dg;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        
        // Setup particles (deterministic seed)
        let pos = generate_random_particles(n, l, seed);
        let mass = vec![1.0_f64; n];
        
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let acc_treepm = compute_treepm_total_forces_cpu(
            &pos, &mass, l, n_pm, r_s, r_cut, theta, softening, g_phys,
        );
        
        let (median, p95, max) = compute_error_stats(&acc_pp, &acc_treepm);
        
        println!("{:4} | {:.3}  | {:.3}  | {:.3}  | {:.4}% | {:.4}% | {:.4}%",
                 n_pm, dg, r_s, r_cut, median*100.0, p95*100.0, max*100.0);
    }
}
```

**Interprétation attendue** :

- Si **erreur décroît** avec N_pm (ex. 11% → 5% → 2% → 1%) : bug de résolution, le 11% à N_pm=64 est juste insuffisant
- Si **erreur plateau** (ex. 11.7% → 11.2% → 10.7% → 10.4%, comme déjà observé) : bug **systémique indépendant de la PM résolution** → cause dans la zone de transition PM/Tree ou dans une convention de normalisation

**Critère de fin Étape 1** : tableau comparatif documenté. Verdict "convergent" ou "plateau".

---

## Étape 2 — Décomposition Tree-only vs PM-only vs combiné (30 min)

Si verdict Étape 1 = "plateau", l'étape critique est de séparer la contribution de chaque module pour localiser le 11%.

### 2.1 Tree-only baseline

Calculer les forces avec **uniquement le Tree** (sans cutoff PM, en augmentant `r_cut` jusqu'à ce que toutes les paires soient vues par le Tree). Comparer à PP-direct.

```rust
#[test]
#[ignore]
fn diagnostic_tree_only_vs_pp() {
    let n = 1000;
    let l = 100.0_f64;
    let seed = 42;
    let softening = 0.05_f64;
    let g_phys = 1.0_f64;
    
    let pos = generate_random_particles(n, l, seed);
    let mass = vec![1.0_f64; n];
    
    let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
    
    // Tree avec r_cut très grand (= L/2 pour PBC + minimum image)
    // ET splitting désactivé (= 1.0 partout) pour avoir Newton pur via Tree
    let acc_tree_full = compute_tree_only_forces_no_splitting(
        &pos, &mass, l, /*theta=*/0.5, softening, g_phys,
    );
    
    let (median, p95, max) = compute_error_stats(&acc_pp, &acc_tree_full);
    
    println!("\n=== Tree-only (r_cut=∞, splitting=1) vs PP-direct ===");
    println!("Median: {:.4}%, P95: {:.4}%, Max: {:.4}%",
             median*100.0, p95*100.0, max*100.0);
    
    // Si Tree-only est précis (median < 1%), le bug est dans le PM ou la combinaison
    // Si Tree-only montre 11% médiane, le bug est DANS le tree (Barnes-Hut multipole, opening criterion, ou softening)
}
```

**Implémentation `compute_tree_only_forces_no_splitting`** : si la fonction n'existe pas, créer un wrapper minimal qui :
- Utilise `TreePMTree::build_with_rs_and_g` avec `r_s` quelconque (sera ignoré)
- Modifie temporairement `pairwise_acc_with_split` pour utiliser `splitting = 1.0` (force complète Newton)
- OU : fait directement la sommation Barnes-Hut sans appeler la fonction de splitting

**Critère** :
- Tree-only median < 1% → Tree est correct, bug dans PM ou combinaison
- Tree-only median ≈ 11% → bug **dans le Tree** (à investiguer §3)

### 2.2 PM-only à courte portée

Le diagnostic Phase 9.6 avait montré PM-only median 15.85%. Mais c'était à `r > r_cut` typique (zone de validité PM). Vérifier ce que donne PM-only **partout**, en demandant au PM seul de fournir TOUTE la force (pas juste la partie longue portée).

Le PM seul est intrinsèquement imprécis à courte portée — c'est attendu. Mais quantifier précisément :

```rust
#[test]
#[ignore]
fn diagnostic_pm_only_full_range() {
    // PM seul, calcul sur TOUTES les paires (pas juste longue portée)
    // = équivalent à passer r_s très grand (Gaussian damping inactif)
    // 
    // Si PM-only-fullrange ≈ PP-direct à grande distance et diverge à courte portée,
    // c'est cohérent : on connaît la limitation PM
    //
    // Mesurer : à quelle distance r/dg le PM-only commence à donner ±1% de PP ?
    // C'est la zone de raccord critique avec le Tree
}
```

### 2.3 Vérification de la cohérence de la zone de transition

C'est le test le plus important. Pour chaque paire de particules (i,j), comparer :

```
F_pair_PP(r)        : Newton avec softening Plummer (référence)
F_pair_PM_alone(r)  : ce que donnerait le PM si la paire était isolée
F_pair_Tree(r)      : ce que donne le Tree avec son splitting Springel
F_pair_PM+Tree      : total que devrait reconstituer la force complète
```

L'identité que le code TreePM doit satisfaire :

```
F_pair_PM_alone(r) × splitting_pm(r) + F_pair_Tree(r) ≈ F_pair_PP(r)
```

Mesurer cette quantité pour `r ∈ {0.5·r_s, r_s, 2·r_s, 3·r_s, 5·r_s, 8·r_s}` sur des paires synthétiques (2 particules isolées dans la boîte) :

```rust
#[test]
#[ignore]
fn diagnostic_two_particle_decomposition() {
    let l = 100.0_f64;
    let n_pm = 64;
    let dg = l / n_pm as f64;
    let r_s = 1.2 * dg;
    let r_cut = 6.0 * dg;
    let softening = 0.05;
    let g_phys = 1.0;
    let mass = vec![1.0_f64, 1.0];
    
    println!("\n=== Décomposition force 2-particules ===");
    println!("r/r_s | r       | F_PP      | F_TreePM  | rel_err  | F_tree    | F_pm     ");
    println!("------|---------|-----------|-----------|----------|-----------|----------");
    
    for &factor in &[0.5, 1.0, 2.0, 3.0, 5.0, 8.0] {
        let r = factor * r_s;
        let pos = vec![
            (l/2.0, l/2.0, l/2.0),
            (l/2.0 + r, l/2.0, l/2.0),
        ];
        
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let acc_tpm = compute_treepm_total_forces_cpu(
            &pos, &mass, l, n_pm, r_s, r_cut, 0.5, softening, g_phys,
        );
        
        let f_pp = acc_pp[0].0;     // composante x sur particule 0
        let f_tpm = acc_tpm[0].0;
        let rel_err = ((f_tpm - f_pp) / f_pp).abs();
        
        // Décomposer si possible (instrumenter le code pour exposer F_tree et F_pm séparément)
        // Sinon : F_tree calculé via splitting × F_PP_no_softening
        // F_pm = F_total - F_tree
        
        println!("{:.2}  | {:.3}   | {:.5e} | {:.5e} | {:.4}% | ...", 
                 factor, r, f_pp, f_tpm, rel_err*100.0);
    }
}
```

**Interprétation** :
- Si erreur grande à `r ≈ r_s` mais petite à `r ≪ r_s` et `r ≫ r_s` : zone de raccord problématique
- Si erreur uniforme partout : bug global (V_cell, normalisation FFT)
- Si erreur croît avec `r` : bug dans le PM longue portée
- Si erreur décroît avec `r` : bug dans le Tree courte portée

**Critère de fin Étape 2** : verdict sur la localisation du bug (Tree, PM, ou raccord).

---

## Étape 3 — Investigation ciblée selon verdict Étape 2

### Cas A — Bug dans le Tree (Tree-only > 1% médiane)

Vérifier successivement :

**A.1 Opening criterion Barnes-Hut**
```bash
grep -n "theta\|opening\|cell_size.*r\|s/d" src/treepm/tree_short.rs
```

L'opening criterion standard est `s/d < θ` où `s` est la taille de la cellule et `d` la distance au centre. Vérifier :
- Convention `s` : taille du côté ou demi-côté ?
- `d` : distance au COM ou au centre géométrique ?
- θ utilisé : 0.5 (demandé dans test) ou autre ?

**A.2 Calcul du COM**
- Pour des particules égales (mass=1), COM = barycentre. Test : 4 particules sur un carré, COM doit être au centre.
- Pour une cellule m+/m- mixte, séparer les COM par signe (déjà fait selon code).

**A.3 Barnes-Hut multipole order**
Le code utilise-t-il monopôle seul ou ordre supérieur ? Springel 2005 montre que monopôle suffit pour θ=0.5, mais c'est limite. Mesurer l'erreur en fonction de θ : si θ=0.3 donne 1% et θ=0.5 donne 11%, c'est le monopôle qui est insuffisant.

```rust
#[test]
#[ignore]
fn diagnostic_theta_sensitivity() {
    for &theta in &[0.1_f64, 0.3, 0.5, 0.7] {
        // Run avec ce theta, mesurer median rel err
    }
}
```

### Cas B — Bug dans le PM (PM-only diverge à grande distance)

**B.1 V_cell normalization**

CLI doit lire `pm_grid.rs` et tracer **chaque facteur de normalisation** appliqué à `ρ_grid` et à `Φ` :
- Convention `assign_mass` : dépose `mass` ou `mass / V_cell` ?
- FFT forward : avec ou sans normalisation `1/N` ?
- FFT inverse : avec ou sans normalisation `1/N` ?
- Green's function : `-4πG/k²` ou `-G/k²` ou `-1/k²` ?
- Application au caller : `g_constant = G_phys` ou `g_constant = G_phys / V_cell` ?

Construire un tableau "facteur appliqué × ordre d'application" et vérifier que le produit final satisfait l'équation de Poisson `Φ = -4πG·ρ/k²`.

**B.2 Test analytique avec densité connue**

Test : déposer une distribution `ρ(x) = sin(2π·k·x/L)` avec k=4 (un mode pur, pas de bruit haute fréquence). La solution analytique de Poisson est `Φ(x) = +4πG·sin(2π·k·x/L) × (L/(2π·k))²`. Comparer la solution numérique à cette analytique :

```rust
#[test]
#[ignore]
fn diagnostic_pm_analytical_sinusoid() {
    let n_pm = 64;
    let l = 100.0_f64;
    let k_mode = 4;
    let g_phys = 1.0;
    
    // Construire ρ(x) = sin(2πk·x/L) sur la grille (pas via assign_mass, mais directement)
    let mut rho = vec![0.0_f32; n_pm * n_pm * n_pm];
    let cell_size = l / n_pm as f64;
    for i in 0..n_pm {
        let x = i as f64 * cell_size;
        let rho_val = (2.0 * std::f64::consts::PI * k_mode as f64 * x / l).sin();
        for j in 0..n_pm {
            for k in 0..n_pm {
                rho[i*n_pm*n_pm + j*n_pm + k] = rho_val as f32;
            }
        }
    }
    
    // Résoudre Poisson
    // [...] solve_poisson(...)
    
    // Comparer à analytique : Φ(x) = +sin(2πk·x/L) × (L/(2πk))² × (4πG)
    let expected_amplitude = (l / (2.0 * std::f64::consts::PI * k_mode as f64)).powi(2) 
                             * (4.0 * std::f64::consts::PI * g_phys);
    
    // [Mesurer amplitude effective dans rho après solve_poisson, comparer]
    
    println!("Expected amplitude: {}", expected_amplitude);
    println!("Measured amplitude: {}", measured);
    println!("Ratio: {:.6}", measured / expected_amplitude);
    
    // Si ratio = 1.0 ± 1% : PM correct
    // Si ratio = constante différente de 1 : facteur de normalisation incorrect (lequel ?)
    // Si ratio dépend de N_pm : bug différent (FFT norm probable)
}
```

**B.3 Convention cell-centered vs node-centered**

Vérifier si la position d'évaluation de Φ correspond à la position où la masse est déposée. Si CIC dépose à la cellule `[i,j,k]` mais que le gradient est évalué au coin `(i·dg, j·dg, k·dg)` au lieu du centre `((i+0.5)·dg, ...)`, ça décale tout d'un demi-cellule, ce qui peut donner un biais constant.

### Cas C — Bug à la zone de raccord (Tree OK, PM OK, combinaison fausse)

Le splitting Springel est mathématiquement exact. Si Tree et PM sont individuellement corrects mais ne se combinent pas correctement, c'est probable que :

**C.1 La fonction splitting_pm en réel n'est pas le complément de splitting_tree_springel**

Le code utilise actuellement `splitting_pm_weight = (r/r_cut)^4` polynomial pour le **Tree weighting** (Phase 9.6 docs). Mais le PM lui-même utilise Gaussian damping `exp(-k²·r_s²)` en Fourier. La transformée de Fourier de la Gaussian k-space est une **Gaussian en réel**, pas un polynôme.

Donc :
- PM en réel équivalent à : `splitting_pm_real(r) = 1 - T(r/(2·r_s))`
- Tree utilise : `splitting_tree_springel(r, r_s) = T(r/(2·r_s))`

Vérifier que dans le code, le tree utilise bien `T(r/(2·r_s))` (pas `1 - T(...)` par erreur, ou `1 - polynomial(r/r_cut)` qui était l'ancienne convention). Lire `tree_short.rs::pairwise_acc_with_split` ligne par ligne et vérifier le signe et la convention.

**C.2 Double comptage à la zone de raccord**

Si Tree et PM contribuent tous deux à la même zone (r entre 2·r_s et 6·r_s), on a double comptage. Le splitting Springel est censé éviter ça mathématiquement, mais une erreur d'implémentation peut le casser.

Test : pour une paire à `r = 4·r_s` (zone de raccord typique), mesurer :
- F_total mesuré
- F_PP attendu
- T(r/(2r_s)) = T(2) ≈ 0.046 → Tree fournit ~4.6% de la force, PM fournit ~95.4%
- Si T(r/(2r_s)) appliqué à un endroit en plus du PM Gaussian, on amplifierait

---

## Étape 4 — Test de la PBC (15 min)

Indépendamment, vérifier que la convention PBC est cohérente entre PP-direct et TreePM.

```rust
#[test]
#[ignore]
fn diagnostic_pbc_consistency() {
    // 2 particules : une à (10, 50, 50), l'autre à (90, 50, 50)
    // Distance directe : 80 Mpc
    // Distance minimum image : 20 Mpc (via la frontière)
    // 
    // PP-direct doit utiliser 20 Mpc (minimum image)
    // TreePM doit donner la même réponse via PM (FFT PBC) + Tree (qui doit aussi gérer PBC)
    
    let l = 100.0_f64;
    let pos = vec![(10.0, 50.0, 50.0), (90.0, 50.0, 50.0)];
    let mass = vec![1.0_f64, 1.0];
    
    let acc_pp = pp_direct_forces_newton(&pos, &mass, l, 0.05, 1.0);
    let acc_tpm = compute_treepm_total_forces_cpu(
        &pos, &mass, l, 64, 1.875, 9.375, 0.5, 0.05, 1.0,
    );
    
    println!("PP-direct (min image at r=20):  acc_x[0] = {:.6e}", acc_pp[0].0);
    println!("TreePM (FFT PBC + Tree min image): acc_x[0] = {:.6e}", acc_tpm[0].0);
    
    // Si les deux ne sont pas cohérents (signe ou amplitude), bug dans la PBC du Tree
    // F_PP attendu : +G·m/r² avec r=20 ≈ 1/400 = 2.5e-3
    // Mais avec softening 0.05, c'est ~2.5e-3 (négligeable correction)
}
```

**Critère de fin Étape 4** : PBC cohérente confirmée ou bug PBC localisé.

---

## Étape 5 — Synthèse et recommandation

Compiler dans `logs/treepm/phase97b_synthesis.md` :

```markdown
# Phase 9.7-B — Synthèse investigation 11% résiduel

## Résultats par étape

| Étape | Résultat | Verdict |
|---|---|---|
| 1. Convergence N_pm | [tableau] | convergent / plateau |
| 2. Tree-only | median X% | bug Tree / OK |
| 2. PM-only | median X% | bug PM / OK |
| 2. Décomposition 2-part | [tableau r/r_s] | localisation |
| 3. Investigation ciblée | [résultats] | cause identifiée |
| 4. PBC | cohérent / divergent | OK / bug |

## Cause root identifiée

[Description précise du bug avec ligne de code et valeur incorrecte]

## Fix proposé

[Diff exact à appliquer, OU recommandation de design change]

## Tests de validation

[Comment vérifier que le fix marche]

## Estimation effort fix

[X heures]

## CLI recommendation

GO Phase 9.8 (apply fix) / NO-GO (need more investigation) / DEMAND HUMAN
```

---

## 6. Critères d'arrêt automatique

CLI s'arrête et demande humain UNIQUEMENT si :

1. **Étape 2 révèle 3 sources d'erreur simultanées** (Tree, PM, et raccord tous biaisés) — investigation devient combinatoire, l'humain doit prioriser
2. **Aucune des hypothèses A/B/C ne reproduit la signature d'erreur observée** — le bug est ailleurs (concurrency, FFT library bug, etc.) et nécessite expertise spécifique
3. **Un fix proposé en Étape 3/5 modifierait l'API publique du module TreePM** (signature changée pour des callers externes) — décision architecturale
4. **Le test analytique B.2 (sinusoid) donne ratio=1.0 ± 1%** mais le test combiné garde 11% — paradoxe nécessitant analyse humaine

Tout autre cas : continuer en autonomie, documenter, proposer fix dans la synthèse.

---

## Estimation de durée

| Étape | Durée |
|---|---|
| 1. Convergence N_pm | 15 min |
| 2. Tree-only / PM-only / décomposition | 1h |
| 3. Investigation ciblée | 1-2h selon cas |
| 4. PBC | 15 min |
| 5. Synthèse | 30 min |
| **TOTAL** | **3-4h** |

Si CLI dépasse 5h sans verdict clair : critère §6.2, demander humain.

---

## Référence physique

**Identité fondamentale TreePM** :

```
F_total(r) = F_PP(r)                     (force complète Newton souhaitée)

F_total(r) = F_PM(r) + F_Tree(r)         (somme des deux modules)

F_PM(r) = -G·m/r² × [1 - T(r/(2r_s))]    (Gaussian damping en Fourier <-> 1-T en réel)

F_Tree(r) = -G·m/r² × T(r/(2r_s))        (convention Springel 2005 / PhotoNs)

Donc:  F_PM + F_Tree = -G·m/r² × [1 - T + T] = -G·m/r²  ✓ exact
```

Si cette identité n'est pas exactement respectée par le code, c'est la source du 11%.

**Rappel valeurs T(x)** :
- T(0) = 1.0
- T(1) ≈ 0.5724
- T(2) ≈ 0.0460
- T(3) ≈ 4.40e-4

À `r = 2·r_s`, T(1) ≈ 0.5724 : Tree fournit 57.24% de la force, PM 42.76%.
À `r = 4·r_s`, T(2) ≈ 0.0460 : Tree fournit 4.60% de la force, PM 95.40%.

---

**Fin du prompt. Bonne route, CLI.**
