# Posture théorique de janus-sim

## Modèle implémenté

janus-sim implémente le **Modèle Cosmologique Janus** (JCM) tel
que formulé par J.-P. Petit et G. D'Agostini dans leurs
publications :

- Petit (2014), Mod. Phys. Lett. A 29, 1450182
- D'Agostini & Petit (2018), Astrophys. Space Sci. 363, 139
- Petit, D'Agostini & Debergh (2018), Progress in Physics 14
- Petit, Margnat & Zejli (2024), Eur. Phys. J. C 84, 1226

## Posture du présent travail

Ce travail prend le modèle Janus **comme cadre phénoménologique
pour investigation numérique**. Il ne préjuge pas de :

1. la cohérence mathématique fondamentale de la formulation
   bimétrique (en particulier vis-à-vis des identités de
   Bianchi)
2. la compatibilité avec les contraintes cosmologiques CMB
   (qui n'ont pas été simulées ici)
3. l'unicité de l'interprétation physique du modèle parmi
   les alternatives bimétriques

Le code documente **explicitement et de façon transparente** les
conventions d'implémentation propres à la présente version :

- raccord cosmologique à z = 4.51 (C0-discontinu, non Petit-natif)
- convention VSL phénoménologique c²(z)/c²₀ = (1+z)^δ
- choix des constantes α² = 0.1815 et τ₀ = 22.71 Gyr
- μ = 19 dérivé de la platitude Ωb(1+μ) = 1

Ces conventions sont **discutables et révisables**. Le présent
travail revendique seulement la cohérence numérique \emph{de cette
implémentation} et la viabilité phénoménologique \emph{de ces
choix}.

## Pour aller plus loin

Les critiques publiques du modèle Janus (compatibilité Bianchi,
universalité du couplage, etc.) sont référencées et discutées
dans le paper accompagnateur (§1, §7.3). Le présent code
n'apporte pas de réponse à ces critiques — il fournit un
outil numérique permettant d'évaluer empiriquement la
phénoménologie du modèle.
