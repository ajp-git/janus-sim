# Mode autonome v2 — AJP absent à nouveau

## Situation

AJP est absent. Phase 8 audit a conclu "code de force correct" mais 
certains indicateurs restent suspects :
- r(k) large = +0.9992 à z=1.14 (ne décroît pas avec le temps)
- Corr = +0.2 stable entre z=10 et z=1.14 (devrait évoluer si Janus fonctionne)

AJP a rejeté la proposition "tester μ=1" parce que ça changerait le modèle 
sans prouver que le code fonctionne pour μ=19.

## Mission

Exécuter Phase 9 — analyse approfondie du snapshot existant sans nouveau run.
Le but : trancher si le code Janus fonctionne à μ=19, ou s'il y a un bug plus 
profond.

## Règles inviolables (rappel)

1. **PAS de run de production** (> 3h). Phase 9 est de l'analyse Python pure.
2. **PAS de patch au code source**. Analyse en lecture seule.
3. **PAS de test μ=1** ni aucune autre variation de paramètres physiques. 
   AJP a explicitement rejeté cette voie.
4. **PAS de conclusion "GO production"** même si résultats paraissent bons.
5. **PAS de git push** vers un remote.
6. Tests courts (< 30 min) sur snapshot existant uniquement.

## Mission Phase 9 — 5 tâches

Voir `PHASE9_DEEPER_ANALYSIS.md` pour les détails techniques. Résumé :

### Tâche 1 — r(k) sur plage large de k

À 128³ et 256³, calculer r(k) pour k = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0 h/Mpc.

### Tâche 2 — Évolution temporelle de r(k)

À 128³, calculer r(k) à z=10, z=5, z=2, z=1.14 pour k = 0.05, 0.1, 0.5, 1.0 h/Mpc.

Utiliser les snapshots existants du run test_phase7_q5_z03 :
- snap_00000 (z=10)
- snap à z≈5
- snap à z≈2
- snap_02720 (z=1.14)

### Tâche 3 — Séparation spatiale m+ / m-

À 128³, calculer var(δ+), var(δ-), ratio, δ_max, percentile 99.

### Tâche 4 — Ségrégation par cellule

À 128³, fraction de cellules dans les 4 quadrants (signes de δ+ et δ-).

### Tâche 5 — Profil radial autour du pic m-

Trouver la cellule avec δ_minus max, tracer δ+(r) et δ-(r) en profil radial.

## Script Python à écrire

Si pas encore existant, créer `scripts/phase9_deep_analysis.py` qui :
1. Charge les snapshots nécessaires
2. Construit les grilles de densité à 128³ et 256³
3. Calcule les 5 séries de métriques
4. Génère un rapport markdown avec tableaux complétés

Le script doit produire des fichiers :
- `/app/output/phase9_deep_analysis.md` (rapport texte)
- `/app/output/phase9_rk_vs_k.png` (graphe tâche 1)
- `/app/output/phase9_rk_evolution.png` (graphe tâche 2)
- `/app/output/phase9_radial_profile.png` (graphe tâche 5)

## Verdict à rédiger

À la fin du rapport, inclure un verdict structuré :

```
=== VERDICT PHASE 9 ===

Critère 1 — r(k) décroît avec k :     ✅ / ❌
Critère 2 — r(k) décroît avec t :     ✅ / ❌
Critère 3 — var(δ-)/var(δ+) > 1 :     ✅ / ❌
Critère 4 — Ségrégation spatiale :    ✅ / ❌
Critère 5 — Creux m+ autour pic m- : ✅ / ❌

Score : N/5 critères Janus satisfaits

Interprétation :
  - 5/5 → code Janus fonctionne correctement à μ=19
  - 3-4/5 → signal faible mais présent, ambigu
  - 0-2/5 → code NE simule PAS Janus, bug plus profond
```

**Ne pas ajouter "tout est OK, GO production" même si score = 5/5.** 
La décision GO revient à AJP.

## Détection des tentatives de dérive

Si pendant l'analyse tu penses :
- "Tester μ=1 éclaircirait les choses" → NON, c'est explicitement interdit
- "On pourrait patcher rapidement X pour voir" → NON, pas de patch
- "Je vais lancer un test de 1h pour comparer" → NON, pas de run
- "L'analyse pointe vers un bug, je corrige" → NON, document le bug et stop

Dans chacun de ces cas : **arrête, écris ce que tu voulais faire dans le log, 
et attends AJP**.

## Si la Phase 9 se termine proprement

Écrire dans le log :
- Heure de fin
- Score X/5 obtenu
- Résumé objectif (pas d'interprétation rassurante ou alarmiste)
- Liste des fichiers produits

Ne pas enchaîner sur une autre phase. Attendre AJP.

## Si tu rencontres un problème

- Erreur dans le script Python : essayer de corriger le script, pas le code source
- Snapshot manquant : noter dans le log, skipper cette tâche, continuer les autres
- Résultat inattendu : documenter, ne pas spéculer dans le verdict
- Comportement de CLI qui semble bizarre : stop, attendre AJP

## Durée estimée

Phase 9 est de l'analyse Python sur snapshots existants. 30-60 minutes 
maximum. Ne pas passer plus de 2h. Si après 2h tu n'as pas fini, écrire 
dans le log où tu en es et attendre AJP.

