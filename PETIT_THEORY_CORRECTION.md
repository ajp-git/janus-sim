# Correction théorique fondamentale — retour aux papers de Petit

J'ai lu les 8 papers originaux de Petit (2014 MPLA, 2018 Progress in Physics CMB, 2019 PTEP consistency, 2024 CITV). Voici ce qu'ils disent réellement, et ce que ça implique pour nos simulations.

---

## 1. Le rapport de densité η — on a tout faux

Petit ne prédit PAS η ≈ 1. Il postule **ρ⁻ >> ρ⁺**, avec μ = ρ⁻/ρ⁺.

- Simulations DESY 1992 (Descamp) : **μ = 64** pour obtenir la structure lacunaire
- Valeur "conservatrice" du livre : **μ = 8**
- Justification physique : le temps de Jeans est t_J ∝ 1/√(4πGρ), donc avec ρ⁻ >> ρ⁺, les masses négatives s'effondrent **en premier**, créent les vides, et compriment les masses positives en parois

Nos η = 0.87–1.00 donnent deux instabilités simultanées → ce n'est pas le mécanisme de Petit.

---

## 2. La longueur d'écrantage λ — elle n'existe pas dans la théorie

Dans aucun des 8 papers, Petit ne mentionne un paramètre λ de Yukawa. La loi cross-espèce est une **anti-Newton pure en 1/r²** :

```
Δφ⁺ = −Δφ⁻ = 4πG(δρ⁺ + δρ⁻)
d²x⁺/dt² = −∂φ⁺/∂x
d²x⁻/dt² = +∂φ⁺/∂x   ← signe opposé
```

Toutes nos tours de trichotomie sur λ optimisaient un paramètre sans existence théorique.

---

## 3. La structure prédite — des vides, pas des filaments

Petit prédit :
- Masses **négatives** → **conglomérats sphéroïdaux** au centre des grands vides cosmiques
- Masses **positives** → **parois planes** (walls) et filaments aux intersections
- Morphologie globale : bulles de savon jointives, amas de galaxies aux jonctions de 4 bulles

La métrique pertinente n'est pas P global, c'est la **corrélation vide ↔ domination m⁻**.

---

## 4. Les conditions initiales — homogène aléatoire + asymétrie ρ

Petit utilise des ICs **uniformes aléatoires**, avec juste le rapport μ = ρ⁻/ρ⁺ >> 1. Pas de Zel'dovich, pas de corrélation δ⁻ = ±δ⁺. Le mécanisme est entièrement piloté par l'instabilité de Jeans asymétrique.

---

## Ce qu'il faut tester maintenant

**Run de référence théorique Petit pur :**

- μ = 8 (ρ⁻/ρ⁺ = 8) — valeur minimale du livre
- Loi anti-Newton 1/r² pure — λ = 0 (désactiver tout écrantage Yukawa)
- ICs : positions aléatoires uniformes, deux populations, rapport de nombre N⁻/N⁺ = 8
- Boîte : 500 Mpc, N_total = 2M (1.78M négatifs + 222K positifs)
- Intégrateur : leapfrog standard, pas de modification

**Métriques à mesurer :**
1. Fraction de vides (cellules > 95% m⁻) en fonction du temps
2. Fraction de parois (cellules > 95% m⁺)
3. Taille caractéristique des conglomérats négatifs (rayon moyen des régions m⁻ pures)
4. P local 32³ comme avant pour comparaison

**Question physique précise :**
Est-ce que μ = 8, loi 1/r² pure, ICs aléatoires reproduit la structure lacunaire (voids sphéroïdaux + parois) décrite par Petit dans ses simulations 2D ?

Si oui → on a enfin la run de référence théoriquement justifiée.
Si non → on documente pourquoi et on ajuste μ vers 64.
