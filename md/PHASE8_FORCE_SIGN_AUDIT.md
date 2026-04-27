# Phase 8 — Audit du signe des forces Janus

## Contexte

Le snapshot z=1.14 du test Phase 7 Q5 révèle une physique ΛCDM au lieu de 
Janus :
- corr(δ+, δ-) = +0.58 (devrait être < 0)
- r(k) large = +0.999 (devrait être < 0)
- m+ forment des halos δ=74 (comportement ΛCDM typique)

**Le run a été arrêté.** On doit identifier pourquoi la répulsion 
cross-species ne fonctionne pas.

## Tâche — Audit SANS modification

Inspecter en lecture seule tous les endroits où le signe des forces 
cross-species est déterminé.

### Question 1 — Interaction factors

L'audit précédent (Q4) a identifié ce code dans `compute_forces_bvh` :

```cuda
// m+ → particule (lignes 746-751)
double interaction = (my_sign > 0) ? 1.0 : -c_ratio_sq;
double f = interaction * mass_plus * inv_rp3;

// m- → particule (lignes 764-769)
double interaction = (my_sign < 0) ? 1.0 : -1.0;
double f = interaction * mass_minus * inv_rm3;
```

Vérifier :
1. Que ces lignes sont bien présentes **inchangées** dans le code actuel
2. La valeur actuelle de `c_ratio_sq` (constante ou dynamique ?)
3. Si le code de lancement passe la bonne valeur de `c_ratio_sq`

**Rapporter** :
- Valeur de `c_ratio_sq` utilisée au lancement (chercher dans 
  `janus_adaptive_zoom.rs` comment elle est initialisée et mise à jour)
- Comment elle évolue avec z si dynamique
- La valeur effective à z=10 (step 0) et à z=1.14 (step 2500+)

### Question 2 — Type de `signs`

Vérifier le type exact de la variable `signs` :

```bash
grep -n "signs:" src/nbody_gpu.rs | head -10
grep -n "signs.*Vec<" src/nbody_gpu.rs | head -10
grep -n "__restrict__.*signs" src/nbody_gpu.rs
```

Rapporter :
- Type Rust (i32 ? i8 ? u8 ?)
- Type CUDA dans le kernel (int* ? char* ?)
- Cohérence des deux

Si `signs` est signé côté Rust mais non-signé côté kernel CUDA, 
`signs=-1` devient `signs=255` ou similaire, et la comparaison 
`(my_sign > 0)` retourne toujours `true`.

### Question 3 — Injection du signe dans la force

Vérifier l'assemblage final de l'accélération. Dans le kernel :

```cuda
// Combien de fois apparaît "acc +=" vs "acc -=" ?
// La force répulsive doit avoir son signe correctement propagé
```

Extraire les **lignes complètes** qui calculent `ax/ay/az` (pas juste le 
facteur `interaction`). Pour chaque accumulation de force, confirmer 
que le signe est cohérent avec la physique Janus.

### Question 4 — Test VSL

Si `c_ratio_sq` évolue :

```bash
grep -n "c_ratio_sq" src/bin/janus_adaptive_zoom.rs
grep -n "c_ratio_sq" src/nbody_gpu.rs
```

Rapporter :
- Fonction qui calcule `c_ratio_sq(z)`
- Valeur à z_init et z_final du test
- Si elle peut devenir négative ou > 1 à un moment

### Question 5 — Sanity test : forcer la répulsion

Ne PAS modifier le code, mais **proposer** un test de sanité :

Si on lance le test avec `c_ratio_sq = 1.0` **forcé** (pas de VSL), 
est-ce que la répulsion Janus réapparaît ?

Ne pas exécuter ce test tant qu'on n'a pas analysé les Q1-Q4.

## Rapport attendu

Un document structuré répondant aux 5 questions avec code EXACT (pas 
de paraphrase). Idéalement avec les numéros de ligne précis.

## Ne RIEN modifier

L'audit est en lecture seule. Si un bug évident apparaît (ex: `signs` 
mal typé), le signaler sans patcher. On décidera ensemble.

