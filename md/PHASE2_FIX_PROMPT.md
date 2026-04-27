# Phase 2 — Fix périodicité niveau 2 (COM merge)

## Contexte
Le fix niveau 1 (distances dans les kernels de force) a réduit la déviation 
v_rms de 87% → 50%, mais le critère GO n'est pas atteint. Le diagnostic de 
CLI est correct : les COM de l'arbre BVH sont calculés sans minimum image 
dans `reduce_com_bvh` (lignes 625-660).

Conséquence : quand un nœud fusionne deux enfants contenant des particules 
aux bords opposés, le COM apparent est au centre de la box au lieu d'être 
"à travers le bord". Ces COMs fantômes attirent les particules vers le 
centre artificiellement.

## Objectif Phase 2
Corriger le merge des COM par l'approche "unfolding" : calculer le COM 
fusionné en tenant compte de la périodicité, puis wrapper le résultat dans 
la box.

## Étape 1 — Audit préalable du kernel reduce_com_bvh

Avant toute modification, rapporter :

1. Le code exact des lignes 625-660 (kernel `reduce_com_bvh` ou équivalent)
2. La structure exacte de `node_data` : à quels indices sont stockés 
   `com_plus_xyz`, `com_minus_xyz`, `mass_plus`, `mass_minus`, `bbox`, etc. ?
3. Comment les enfants gauche/droite sont fusionnés : arbre bottom-up ?
4. Si `box_size` est déjà disponible dans le scope du kernel (paramètre ou 
   constante globale)

Me remonter ce rapport d'abord. Ne pas patcher avant retour.

## Étape 2 — Modification proposée (après validation du rapport)

Pour chaque COM fusionné (m+, m-, ou les deux), appliquer l'approche 
unfolding :

```cuda
// Exemple pour com_plus : fusion des enfants gauche et droit
// m_left, m_right sont les masses totales m+ des enfants
double total_mp = m_left + m_right;

if (total_mp > 0.0) {
    // Référence : COM gauche
    double com_left_x = node_data[left_base + 4];
    double com_left_y = node_data[left_base + 5];
    double com_left_z = node_data[left_base + 6];
    
    // Droite, avec minimum image depuis la gauche
    double com_right_x = node_data[right_base + 4];
    double com_right_y = node_data[right_base + 5];
    double com_right_z = node_data[right_base + 6];
    
    double dx = minimum_image(com_right_x - com_left_x, box_size, box_half);
    double dy = minimum_image(com_right_y - com_left_y, box_size, box_half);
    double dz = minimum_image(com_right_z - com_left_z, box_size, box_half);
    
    // Poids (ne pas diviser par zéro si une branche est vide)
    double w_right = m_right / total_mp;
    
    // COM fusionné (dans le référentiel "déplié" de gauche)
    double com_merged_x = com_left_x + w_right * dx;
    double com_merged_y = com_left_y + w_right * dy;
    double com_merged_z = com_left_z + w_right * dz;
    
    // Wrap back dans [-L/2, +L/2]
    if (com_merged_x >  box_half) com_merged_x -= box_size;
    if (com_merged_x < -box_half) com_merged_x += box_size;
    if (com_merged_y >  box_half) com_merged_y -= box_size;
    if (com_merged_y < -box_half) com_merged_y += box_size;
    if (com_merged_z >  box_half) com_merged_z -= box_size;
    if (com_merged_z < -box_half) com_merged_z += box_size;
    
    node_data[my_base + 4] = com_merged_x;
    node_data[my_base + 5] = com_merged_y;
    node_data[my_base + 6] = com_merged_z;
}
```

Faire la même chose pour `com_minus` (indices différents).

## Étape 3 — Cas dégénérés à gérer

- Si `m_left == 0` ou `m_right == 0` (un enfant vide) : prendre directement 
  le COM de l'enfant non-vide, pas d'unfolding nécessaire.
- Si `total_mp == 0` (nœud sans m+) : mettre COM à (0,0,0) ou laisser valeur 
  indéterminée mais s'assurer que la masse est à 0 (sera filtré par masse=0 
  ailleurs).

## Étape 4 — Passer box_size au kernel reduce_com_bvh

Si ce n'est pas déjà fait, ajouter `double box_size` à la signature du 
kernel, et ajouter `box_size` à l'appel `.launch()` correspondant.

## Étape 5 — Compilation

```bash
cargo build --release --features cuda --bin janus_adaptive_zoom
```

## Étape 6 — Test empirique identique à Phase 1

Relancer exactement le même test coquilles radiales :

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 50 --l-box 100 --z-init 10.0 --z-final 9.8 \
  --snap-interval 10 --steps-check 999 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir /app/output/test_periodic_fix_lvl2 \
  --run-label test_periodic_fix_lvl2
```

Puis même script Python d'analyse radiale.

**Critère GO** :
- `|<v_r>|` < 50 km/s dans toutes les coquilles
- `v_rms` varie de moins de 20% entre coquilles
- Pas de pattern monotone infall→outflow

**Critère NO-GO** :
- Pattern similaire à Phase 1 → il reste un autre bug ailleurs
- NaN ou instabilité → régression introduite par le fix

## Étape 7 — Rapport

Me fournir :
1. Rapport audit Étape 1 (avant patch)
2. Diff des modifications
3. Build log
4. Tableau coquilles radiales
5. Verdict GO/NO-GO

## Étape 8 — Si GO

Test supplémentaire de régression : relancer 50 steps sur la config initiale 
pour vérifier que le kernel produit toujours des forces cohérentes (pas de 
division par zéro, pas de NaN dans un halo dense).

---

## Notes

- **Ne pas toucher** à autre chose que le merge des COM
- **Ne pas** tenter d'optimiser pour l'instant
- Le `minimum_image()` helper défini en Phase 1 est réutilisé ici
- Si un cas dégénéré produit NaN : me le signaler sans le patcher
