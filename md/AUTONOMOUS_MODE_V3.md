# Mode autonome v3 — AJP absent 6h en voiture

## Contexte

AJP part en voiture, sera absent 6h environ. Pas d'accès téléphone ni 
messagerie pendant le trajet.

Le test Phase 13 (offset Morton octree) est en cours (~2h restantes). 
Selon le résultat, CLI doit lancer OU NE PAS lancer la production v7b 
10M jusqu'à z=0.

## Règles inviolables

1. **Suivre strictement le critère GO/NO-GO** défini ci-dessous. Pas 
   d'interprétation créative, pas de "presque-GO", pas de "ça devrait 
   aller".

2. **Si ambigu → NO-GO.** En cas de doute, on ne lance pas la 
   production. AJP préfère attendre son retour que de gaspiller 30h.

3. **Pas de nouveaux fixes code source** pendant son absence. 
   Uniquement lancement ou non-lancement de la production avec le code 
   tel qu'il est maintenant (avec le patch Phase 13 appliqué).

4. **Documenter tout** dans `/app/output/AUTONOMOUS_LOG_V3.md` (append).

## Mission

### Étape 1 — Attendre la fin du test Phase 13

Le run `test_phase13_octree` doit atteindre z=2. Attendre qu'il se 
termine ou que le kill-switch s'active.

### Étape 2 — Calculer le spectre angulaire sur snap final

Utiliser `validate_ics.py` (ou script Phase 9) sur le dernier snapshot 
du run Phase 13.

Extraire :
- max_axes(m+), max_axes(m-), max_axes(total)
- Angle du pic principal m-

### Étape 3 — Application stricte du critère GO/NO-GO

**GO production** si ET seulement si TOUS ces critères sont vrais :

| Critère | Seuil |
|---------|-------|
| max_axes(m-) à z=2 | < 3.0 |
| max_axes(total) à z=2 | < 3.0 |
| Angle pic m- | pas dans [85°, 95°] ni [265°, 275°] ni [-5°, 5°] ni [175°, 185°] |
| Aucun NaN dans le run | — |
| v_rms final | < 2000 km/s |

**NO-GO production** si au moins UN critère échoue. Aucune exception.

### Étape 4 si GO — Lancer la production v7b

Nettoyer le dossier précédent si existant :

```bash
# Vérifier que janus_adaptive_v7b n'existe pas OU est vide
ls /app/output/janus_adaptive_v7b 2>/dev/null
# Si existe, le renommer pour archive :
[ -d /app/output/janus_adaptive_v7b ] && mv /app/output/janus_adaptive_v7b /app/output/janus_adaptive_v7b_old_$(date +%s)
```

Lancer :

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 215 --l-box 500 --z-init 10.0 --z-final 0.0 \
  --snap-interval 20 --steps-check 50 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir /app/output/janus_adaptive_v7b \
  --run-label v7b_all_fixes \
  2>&1 | tee /app/output/janus_adaptive_v7b/run.log
```

Activer le kill-switch habituel dans un tmux :

```bash
tmux new -d -s janus_monitor '
while true; do
  last=$(tail -1 /app/output/janus_adaptive_v7b/time_series.csv 2>/dev/null)
  if [ -z "$last" ]; then sleep 120; continue; fi
  step=$(echo $last | cut -d, -f1)
  z=$(echo $last | cut -d, -f3)
  rho_max=$(echo $last | cut -d, -f8)
  v_rms=$(echo $last | cut -d, -f9)
  
  awk -v v="$v_rms" "BEGIN { if (v+0 > 20000) exit 1 }"
  if [ $? -ne 0 ]; then
    echo "CRITICAL v_rms=$v_rms at step=$step z=$z" > /app/output/janus_adaptive_v7b/KILLED.txt
    docker stop $(docker ps -q --filter name=janus)
    break
  fi
  
  awk -v r="$rho_max" "BEGIN { if (r+0 > 1e17) exit 1 }"
  if [ $? -ne 0 ]; then
    echo "CRITICAL rho_max=$rho_max at step=$step z=$z" > /app/output/janus_adaptive_v7b/KILLED.txt
    docker stop $(docker ps -q --filter name=janus)
    break
  fi
  
  echo "$(date +%H:%M) step=$step z=$z rho=$rho_max v=$v_rms" >> /app/output/janus_adaptive_v7b/monitor.log
  sleep 1800
done
'
```

### Étape 5 si NO-GO — Ne rien lancer, documenter

Écrire dans le log :

```
=== PHASE 13 RESULT: NO-GO ===
Critères échoués :
- [liste des critères qui n'ont pas passé avec valeurs]
Production NON lancée. En attente d'AJP.
```

Puis **s'arrêter** et ne rien faire d'autre.

### Étape 6 — Checkpoints pendant la production (si lancée)

Si la production tourne, toutes les 90 minutes, appender dans le log :

```
=== CHECKPOINT à {HH:MM} ===
Step : {step}
z : {z}
v_rms : {v_rms} km/s
ρ_max : {rho_max}
split_max : {split_max}
NaN présent : {oui/non via grep}
Spectre angulaire récent (si snapshot disponible) : max_axes(m-) = ?
```

**Si à un checkpoint v_rms > 10 000 km/s**, stopper immédiatement le 
run (même si kill-switch ne s'est pas déclenché). Documenter et 
attendre AJP.

**Si à un checkpoint max_axes(m-) > 10** (contamination grille de 
retour), stopper et documenter.

## Décisions NON autorisées sans AJP

- Modifier le code source
- Relancer un run si kill-switch s'active
- Changer les paramètres de la production (N, z_final, etc.)
- Tenter un nouveau fix
- Conclure "GO" si un critère est à la limite (ex: max_axes=3.01)

## Arbre de décision résumé

```
Test Phase 13 fini ?
├── Oui
│   ├── Tous critères GO ? 
│   │   ├── Oui → Lancer production v7b + kill-switch + checkpoints
│   │   └── Non → Ne rien lancer, documenter, attendre AJP
│   └── Run crashé ou NaN ? → Ne rien lancer, documenter
└── Non → Attendre fin Phase 13
```

## Résumé à écrire à la fin

Dans `AUTONOMOUS_LOG_V3.md`, écrire un résumé final :

```
=== RÉSUMÉ AUTONOME ===
- Heure de fin Phase 13 : {HH:MM}
- Résultat Phase 13 : GO / NO-GO
- Production v7b lancée : OUI / NON
- Si OUI : step atteint à l'heure {HH:MM AJP_return} = ?
- Kill-switch déclenché : OUI / NON
- Anomalies détectées : [liste]
- Fichiers produits : [liste]
```

Ce résumé doit être facile à lire en 30 secondes au retour d'AJP.

