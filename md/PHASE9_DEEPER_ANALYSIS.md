# Phase 9 — Analyse plus approfondie avant décision

## Contexte

CLI a fait un bon audit (Phase 8) et identifié que corr(δ+,δ-) à 64³ était 
pollué par le shot noise. À 256³, corr = +0.04. Il propose μ=1 comme 
prochaine étape.

**Avant de tester μ=1, on doit valider si la physique Janus fonctionne 
correctement à μ=19**, parce que si on change μ et qu'on trouve 
anti-corrélation, on ne saura pas si c'était un bug Janus ou juste μ 
trop extrême.

Deux points restent suspects :
- r(k) large = +0.9992 à z=1.14 (devrait décroître avec le temps)
- Corr = +0.2 stable entre z=10 et z=1.14 (décroissance attendue si 
  Janus fonctionne)

## Tâches à effectuer sur snap z=1.14 (snap_02720)

### Tâche 1 — r(k) sur TOUTE la plage k disponible

À résolution 128³ et 256³, tracer r(k) pour les bandes :

```
k [h/Mpc]    : 0.01  0.02  0.05  0.1   0.2   0.5   1.0   2.0
r(k) 128³    :  ?     ?     ?     ?     ?     ?     ?     ?
r(k) 256³    :  ?     ?     ?     ?     ?     ?     ?     ?
```

**Attendu Janus** : r(k) décroît avec k (grandes échelles → petites 
échelles). r(k→0) = +1 (ICs partagées), r(k→large) → 0 ou négatif.

**Rejeté** : r(k) ≈ +1 partout (ΛCDM-like).

### Tâche 2 — Évolution de r(k) dans le temps

À 128³, tracer r(k) à z=10, z=5, z=2, z=1.14 sur les mêmes bandes :

```
z     r(k=0.05)  r(k=0.1)  r(k=0.5)  r(k=1.0)
10    ~1        ~1        ~1        ~1     (attendu)
5     ?         ?         ?         ?
2     ?         ?         ?         ?
1.14  ?         ?         ?         ?
```

**Attendu Janus** : r(k) décroît avec le temps à petites échelles (k>0.1).

### Tâche 3 — Séparation spatiale m+ / m-

À 128³, calculer :

```
var(δ+)      = ?
var(δ-)      = ?
var(δ-)/var(δ+) = ?    (Janus : > 1 car m- plus contrastés)

Max overdensity :
  δ_max(m+) = ?
  δ_max(m-) = ?         (Janus : δ_max- > δ_max+)
  
Percentile 99 :
  δ_99(m+) = ?
  δ_99(m-) = ?
```

### Tâche 4 — Ségrégation par cellule

À 128³ :

```
Fraction cellules où δ+ > 0 ET δ- > 0   = ?   (co-localisation)
Fraction cellules où δ+ > 0 ET δ- < 0   = ?   (m+ en vide m-)
Fraction cellules où δ+ < 0 ET δ- > 0   = ?   (m- en vide m+)
Fraction cellules où δ+ < 0 ET δ- < 0   = ?   (sous-densité double)
```

**Attendu Janus** : fraction "(+, -) ou (-, +)" > fraction "(+, +) ou (-, -)".
**ΛCDM** : fraction "(+, +)" élevée (co-localisation dominante).

### Tâche 5 — Profil radial autour du pic m-

Trouver la cellule où δ- est maximum à 128³. Tracer le profil radial de 
δ+ et δ- autour de ce point :

```
r [Mpc]     : 5    10   20   50   100
δ-(r)       : ?    ?    ?    ?    ?
δ+(r)       : ?    ?    ?    ?    ?
```

**Attendu Janus** : autour d'un pic m-, δ+ devrait être **en creux** 
(négatif à petit r, les m+ ont été chassés du conglomérat m-).

## Format rapport

Un document `/app/output/phase9_deep_analysis.md` avec les 5 tableaux 
complétés, plus un verdict final :

**GO Janus** si :
- r(k) décroît avec k ET avec t
- var(δ-)/var(δ+) > 1
- Fraction ségrégation > 0.5
- Profil radial montre creux m+ autour pics m-

**NO-GO Janus** si :
- r(k) ≈ +1 partout
- var(δ-)/var(δ+) < 1
- Co-localisation dominante
- Profil radial montre m+ et m- suivre le même pattern

## Après Phase 9

**Si GO Janus** : le code fonctionne, corr=+0.2 à 64³ était bien du 
shot noise. On peut lancer la production v7 à μ=19.

**Si NO-GO Janus** : il y a un bug plus profond. Ne pas tester μ=1 
aveuglément, revenir sur l'audit du kernel de force avec un test 
"2 particules" ultra-minimal (force répulsive ou attractive, test 
direct).

## NE PAS lancer μ=1 avant ce rapport

μ=1 est une voie de contournement qui pourrait masquer le bug principal. 
On doit d'abord savoir si Janus fonctionne à μ=19 ou pas.

