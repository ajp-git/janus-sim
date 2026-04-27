# Phase 3 — Diagnostic discriminant

## Contexte
Fix Phase 1 a réduit la déviation v_rms de 87% → 50%.
Fix Phase 2 (unfolding COMs) a produit des checksums différents mais 
statistiques identiques à Phase 1. La dynamique n'est pas modifiée.

On doit identifier pourquoi avant d'aller plus loin.

## Observation clé du rapport Phase 2

Les m+ et m- ont les MÊMES vitesses radiales :

```
[0,10]   <v_r> m+ = -1195   <v_r> m- = -1413   km/s
[10,20]  <v_r> m+ = -1259   <v_r> m- = -1184   km/s
```

Dans un Janus sain, m+ (5%) devrait être repoussé par m- (95%). Les m+ 
devraient se comporter très différemment des m-. Ici ils tombent ensemble.

## Étape A — Comparaison absolue Phase 1 vs Phase 2

Me fournir, pour les deux phases, le tableau COMPLET avec v_rms absolu :

```
Shell      N       v_rms_P1    v_rms_P2    <v_r>_P1   <v_r>_P2
[0,10]
[10,20]
[20,30]
[30,40]
[40,50]
```

Si v_rms_P2 < v_rms_P1 → le patch niveau 2 améliore quelque chose.
Si v_rms_P2 ≈ v_rms_P1 → le patch niveau 2 est totalement neutralisé.

## Étape B — Vérifier que le code unfolding est réellement exécuté

Ajouter dans le kernel `reduce_com` des compteurs atomiques (ou printf 
conditionnel) pour mesurer :

1. Combien de nœuds sont construits au total pendant un step ?
2. Pour combien de nœuds la formule unfolding produit un résultat 
   DIFFÉRENT du COM naïf ?
3. Pour combien de nœuds la condition `|com_right - com_left| > box_half` 
   est vraie (cas où unfolding joue) ?

Si réponse 3 = 0 → aucun nœud ne traverse les bords → unfolding inutile 
→ le bug est ailleurs.
Si réponse 3 > 0 mais petite → unfolding modifie peu de nœuds 
→ effet dynamique faible attendu.
Si réponse 3 grande → unfolding devrait avoir un effet → chercher pourquoi 
il n'en a pas.

## Étape C — Test décisif : gravité pure (μ=0)

Modifier temporairement le run pour avoir UNIQUEMENT des m+ (pas de m-).

Méthode simple : forcer `mu=0` dans la ligne de commande si supporté, ou 
patcher temporairement le code IC pour générer 100% m+.

Lancer le même test coquilles radiales :

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 50 --l-box 100 --z-init 10.0 --z-final 9.8 \
  --snap-interval 10 --steps-check 999 \
  --h0 69.9 --mu 0.0001 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir /app/output/test_pure_mplus \
  --run-label pure_mplus
```

(μ=0.0001 au lieu de 0 pour éviter division par zéro, génère 99.99% m+)

**Interprétation** :
- Si `<v_r>` ≈ 0 dans toutes les coquilles → la gravité pure m+ est 
  périodique OK, et le bug Phase 2 était vraiment dans la répulsion 
  cross-sign ou ailleurs.
- Si `<v_r>` < 0 partout → même la gravité pure n'est pas périodique, 
  il y a encore un chemin de calcul qui ignore box_size.

## Étape D — Rapport

Me fournir :
1. Tableau Étape A (comparaison absolue P1 vs P2)
2. Comptage Étape B (combien de nœuds traversent les bords)
3. Tableau coquilles Étape C (simulation μ ≈ 0)
4. Diagnostic proposé par CLI

## Important

- Ne pas appliquer d'autre fix pour l'instant
- Si un test fait apparaître un bug évident (ex: COM jamais lu), le 
  signaler mais ne pas corriger
- Si la compilation échoue : remonter le log sans tenter de corriger

