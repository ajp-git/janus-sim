# Mode autonome — AJP est absent pendant quelques heures

## Principes

AJP est absent. Tu (CLI) dois avancer seul, mais dans un périmètre précis.

### Règles inviolables

1. **NE PAS lancer de run de production longue durée** (> 3h). Pas de v7, 
   pas de μ-scan, pas de run >z=0.5 avec N>3M.
2. **NE PAS toucher au preprint** ni à aucun document destiné à JPP.
3. **NE PAS supprimer de fichiers** (snapshots, logs, CSV). Accumuler même 
   si ça prend de la place.
4. **NE PAS git push** vers un remote. Commits locaux OK.
5. **Chaque modification de code Rust/CUDA** doit être dans une branche 
   git séparée (par exemple `cli-autonomous-phase8`), jamais sur `main`.
6. **Si un doute apparaît** → **s'arrêter et écrire un rapport**. Ne pas 
   faire de choix important seul.

### Ressources autorisées

- Tests courts (< 30 min) sur petites boîtes (L ≤ 100 Mpc, N ≤ 500K)
- Compilations, tests unitaires
- Analyses Python sur snapshots existants
- Audits de code en lecture seule

### Communication

Tout ce que tu fais doit être documenté dans :
```
/app/output/AUTONOMOUS_LOG.md
```

Structure : un horodatage + ce que tu as fait + verdict + fichiers produits.
AJP lira ce fichier à son retour.

## Mission principale — Résoudre le bug Janus

Ordre de priorité :

### Étape 1 — Audit Phase 8 (FORCE SIGN AUDIT)

Suivre le plan `PHASE8_FORCE_SIGN_AUDIT.md` qui a été transmis précédemment. 
Répondre aux 5 questions **sans modifier de code**.

**Livrables** :
- `/app/output/audit_phase8_force_sign.md` avec les 5 réponses
- Lignes de code exactes citées avec numéros

Ne pas passer à l'étape 2 tant que l'audit est incomplet.

### Étape 2 — Identifier le bug le plus probable

À partir de l'audit, identifier la cause la plus probable parmi :
- Hypothèse 1 : `c_ratio_sq` dynamique dévie (VSL casse la répulsion)
- Hypothèse 2 : type `signs` mal interprété (u8 vs i32)
- Hypothèse 3 : logique `interaction = ...` fausse
- Hypothèse 4 : propagation du signe de la force incorrecte

Écrire le diagnostic dans le log autonome avec **niveau de confiance** 
(certain / probable / spéculatif).

### Étape 3 — Test discriminant SANS modification de code

Pour chaque hypothèse avec confiance ≥ probable, proposer un test qui la 
discriminerait **sans modifier le code source**.

Exemples de tests discriminants possibles :

**Pour Hypothèse 1 (VSL)** : chercher dans le code un flag ou paramètre 
CLI permettant de désactiver VSL et mettre `c_ratio_sq = 1.0`. Si un tel 
flag existe → lancer un test court (L=50, 20 steps). Si pas de flag, 
passer à autre chose.

**Pour Hypothèse 2 (type signs)** : écrire un kernel CUDA minimal qui lit 
`signs[0]` et `signs[N/2]` et imprime leurs valeurs via printf. Vérifier 
que m+ = +1 et m- = -1 dans le kernel, pas 255 ou 0.

**Pour Hypothèse 3 (logique)** : écrire un script Python qui lit un 
snapshot et calcule la force sur une particule test depuis une particule 
cible de signe connu. Comparer avec ce que dit le kernel.

Lancer les tests qui ne nécessitent pas de modification de code.

### Étape 4 — Proposition de fix (NE PAS APPLIQUER)

Sur la base de l'étape 3, proposer un fix précis avec :
- Fichier à modifier
- Lignes exactes
- Code nouveau (avant/après)
- Tests de validation à faire après le fix

**Écrire la proposition dans le log, ne pas appliquer le patch.**

AJP lira la proposition à son retour et validera ou non.

### Étape 5 — Si Hypothèse 1 (VSL) confirmée avec haute confiance

**ET** qu'il existe un moyen de désactiver VSL via paramètre CLI sans 
modifier de code, **alors** tu peux :

- Lancer un test **court** (L=100, N=500K, z=10→5, ~15 min) avec VSL désactivé
- Mesurer corr(δ+,δ-) sur le snapshot final
- Si corr < 0 → hypothèse confirmée, documenter
- Si corr > 0 → hypothèse infirmée, chercher ailleurs

**Si** VSL ne peut être désactivé sans modifier le code source → ne pas 
modifier, écrire la recommandation dans le log et s'arrêter là.

## Garde-fous supplémentaires

### Détection de boucle improductive

Si après 3 tentatives successives sans progrès (même test répété, même 
diagnostic, pas d'info nouvelle), **s'arrêter** et écrire dans le log :

> "Je tourne en rond. Besoin de l'input d'AJP pour sortir de cette boucle."

### Détection de précipitation

Si tu te surprends à conclure "GO" ou "bug résolu" sans tests 
multiples et indépendants, **demande-toi** : est-ce que je vérifierais 
cela si AJP ou Claude me le demandait ? Si non, continuer à chercher.

### Arbre de décision en cas d'imprévu

**Si erreur de compilation** : noter, ne pas patcher si pas évident, 
s'arrêter et attendre AJP.

**Si test donne résultat ambigu** : noter les deux interprétations, ne 
pas trancher unilatéralement.

**Si hypothèse Phase 8 infirmée** : ne pas chercher une 5ème hypothèse 
seul. Écrire dans le log "Phase 8 audit complet, aucune hypothèse 
confirmée, besoin AJP" et s'arrêter.

## Récapitulatif

Tu as l'autonomie de :
- Faire l'audit complet Phase 8
- Lancer des tests courts non destructifs
- Analyser les snapshots existants
- Proposer des fixes (pas les appliquer)
- Écrire des rapports détaillés

Tu n'as PAS l'autonomie de :
- Modifier du code de production
- Lancer des runs > 3h
- Prendre des décisions scientifiques de fond
- Conclure "le bug est résolu" sans preuves multiples

À la fin de la période autonome, AJP trouvera :
- `/app/output/AUTONOMOUS_LOG.md` (journal)
- `/app/output/audit_phase8_force_sign.md` (audit Phase 8)
- Résultats des tests courts effectués
- Proposition(s) de fix documentée(s)

