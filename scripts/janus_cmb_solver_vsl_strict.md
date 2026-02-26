# JANUS VSL CMB SOLVER — VERSION STRICTE (CORRIGÉE)

## 🎯 Objectif scientifique

Tester falsifiablement si le modèle Janus avec :
- η primordial constant
- VSL dynamique : `dc/dz = +c/2 * x_e(z)/(1+z)`
- transition uniquement via recombinaison (Saha)
- aucune constante cosmologique, aucune matière noire

peut reproduire simultanément :
- r_d ≈ 147 Mpc
- ℓ₁ ≈ 220
- ℓ₂/ℓ₁ ≈ 2.44

---

## ⚠️ CORRECTIONS vs spec initial o3

**Correction 1 — signe de dc/dz :**
Le spec o3 écrit `dc/dz = +1/2 * x_e/(1+z) * c`. C'est correct mais à vérifier :
- d(ln c)/d(ln a) = −½ x_e  (Petit VSL)
- d(ln a)/dz = −1/(1+z)
- Donc dc/dz = **+c/2 · x_e/(1+z)**  ✓ (c croît vers le passé)

**Correction 2 — H(z) primordial :**
Le spec o3 propose `H² = 8πG/3 |ρ₊(1−η)c²|` avec fallback `H=H₀(1+z)`.
**Problème : pour η>1, E = ρ₊(1−η)c² < 0 toujours → le fallback est toujours actif.**
η_primordial ne joue alors aucun rôle. H(z)=H₀(1+z) doit être posé explicitement
comme hypothèse du régime VSL, pas comme fallback d'une formule cassée.

**Solution :** poser H(z) = H₀(1+z) directement dans le régime VSL (z > z_drag),
et H(z) = Janus Friedmann 2014 pour z < z_drag. C'est honnête et non-circulaire.

---

## 1️⃣ PARAMÈTRES FIXES (AUCUN FIT)

```python
H0      = 76.0          # km/s/Mpc — contraint par Pantheon+
Omega_b = 0.0493        # fraction baryonique
T0      = 2.725         # K, CMB aujourd'hui
c0      = 299792.458    # km/s
Ei      = 13.6          # eV, énergie ionisation H

# Constantes SI standards : G, kB, me, mp, h_planck, sigma_T

# Balayage η primordial — AUCUN FIT, grille exhaustive uniquement
eta_grid = np.linspace(1.05, 5.0, 40)   # 40 points uniformes
```

Aucune normalisation ultérieure. Aucun paramètre ajusté post-hoc.

---

## 2️⃣ DOMAINE D'INTÉGRATION

```
z_max = 10000   (x_e = 1 partout au-dessus, Saha inutile)
z_min = 0
```

Intégrer **du passé vers le présent** : z_max → 0.
Interdiction d'intégrer 0 → z_max.

---

## 3️⃣ IONISATION — SAHA EXACTE

À chaque z :

```
T(z)  = T0 * (1+z)
nb(z) = (3*H0²/8πG) * Omega_b * (1+z)³ / mp   [en m⁻³, tout en SI]
```

Résoudre numériquement l'équation de Saha :

```
x_e² / (1 - x_e) = S(z)

où S(z) = (1/nb) * (2π me kB T / h²)^(3/2) * exp(-Ei / kB T)
```

Solution analytique exacte : `x_e = (−S + √(S²+4S)) / 2`

Contrainte stricte : `0 ≤ x_e ≤ 1`

**Vérification attendue :**
- z=800 : x_e ≈ 0.000
- z=1100 : x_e ≈ 0.004
- z=1300 : x_e ≈ 0.167
- z=1400 : x_e ≈ 0.561
- z=1500 : x_e ≈ 0.918

---

## 4️⃣ ÉVOLUTION DE c(z) — ÉQUATION DIFFÉRENTIELLE STRICTE

```
dc/dz = +c(z)/2 * x_e(z) / (1+z)
```

Condition initiale : `c(z_max) = c0`

Intégrer de z_max → 0 (sens décroissant en z).

**Ne jamais imposer** `c ∝ (1+z)^0.5` directement.
Cette loi doit émerger naturellement quand x_e ≈ 1.

**Vérification :** pour z >> z_drag (x_e ≈ 1) :
`d(ln c)/d(ln a) ≈ −0.5` → c ∝ a^(−0.5) ✓

---

## 5️⃣ H(z) — STRUCTURE HONNÊTE

**Régime VSL (z > z_drag) :**
```
H(z) = H0 * (1+z)
```
C'est une hypothèse explicite du régime VSL de Petit (1988/2018), pas un fallback.

**Régime Janus z < z_drag :**
```
H²(z) = H0² * [Omega_+ * (1+z)³ + E_term]
```
avec équations Friedmann Janus 2014. Ce régime peut avoir H²<0 pour z > 2.59 —
c'est documenté et attendu (zone orpheline).

**Ne pas bricoler** une transition continue entre les deux régimes.
Poser z_drag comme frontière explicite, défini en étape 6.

---

## 6️⃣ DÉFINITION DE z_drag

```
z_drag = z tel que x_e(z_drag) = 0.5
```

Résoudre numériquement. **Pas de valeur importée ΛCDM.**

**Résultat attendu avec Saha + H0=76 :**
z_drag ≈ 1387 (différent de 1060 ΛCDM car H(z) différent n'affecte pas Saha).

Note : Saha dépend de T(z) et nb(z), pas de H(z). z_drag est donc robuste.

---

## 7️⃣ VITESSE DU SON

```
R(z)   = (3 * Omega_b / (4 * Omega_gamma)) / (1+z)
          avec Omega_gamma = 2.47e-5 / h² (h = H0/100)
c_s(z) = c(z) / sqrt(3 * (1 + R(z)))
```

---

## 8️⃣ RAYON ACOUSTIQUE

```
r_d = ∫_{z_drag}^{z_max} c_s(z) / H(z)  dz
```

Avec c(z) dynamique (intégré en étape 4) et H(z) = H0*(1+z).

**Résultat attendu :** r_d >> 147 Mpc si la fenêtre d'intégration est large.
C'est le résultat honnête à documenter, pas à cacher.

---

## 9️⃣ DISTANCE DE DIAMÈTRE ANGULAIRE

```
D_M = ∫_0^{z_drag} c(z) / H(z)  dz
```

---

## 🔟 PREMIER PIC CMB

```
ℓ₁ = π * D_M / r_d
```

Aucune normalisation supplémentaire.

---

## 1️⃣1️⃣ VALIDATION STRICTE

Le modèle est viable si toutes les conditions suivantes sont satisfaites :

```
|r_d  − 147| / 147  < 0.10
|ℓ₁   − 220| / 220  < 0.10
```

Sinon → **falsification documentée**.

Un échec est un résultat scientifique. Le documenter proprement.

---

## 1️⃣2️⃣ OUTPUTS OBLIGATOIRES

Le code doit produire :

1. **Plot x_e(z)** sur z ∈ [800, 2000] avec marqueur z_drag
2. **Plot c(z)/c0** sur z ∈ [0, 2000] — vérifier que c ∝ (1+z)^0.5 pour z >> z_drag
3. **Plot intégrande c_s/H** sur z ∈ [z_drag − 200, z_drag + 500]
4. **Tableau de résultats** : z_drag, r_d, D_M, ℓ₁ vs cibles

Sauvegarder en `janus_cmb_results.png`.

---

## 1️⃣3️⃣ CHECKLIST DEBUG OBLIGATOIRE

- [ ] Intégration z_max → 0 (sens décroissant)
- [ ] c(z_max) = c0 comme CI
- [ ] x_e résolu par Saha exacte à chaque pas, pas interpolé grossièrement
- [ ] z_drag déterminé par x_e = 0.5, jamais importé
- [ ] H(z) = H0*(1+z) explicitement dans le régime VSL
- [ ] r_d intégré de z_drag vers z_max (sens croissant en z)
- [ ] Aucun paramètre ajusté pour "faire marcher" le résultat

---

## 🧠 Contexte pour le modèle

Ce calcul est une **vérification de falsifiabilité**, pas un ajustement.

Résultat connu a priori (calcul préliminaire) :
- r_d sera >> 147 Mpc avec cette équation de transition
- car x_e passe de 1 à 0 sur Δz ≈ 600, rendant la fenêtre d'intégration trop large

Le test détermine si l'équation `dc/dz = +c/2 * x_e/(1+z)` est suffisante,
ou si un mécanisme de transition plus abrupt est nécessaire.

**C'est une question posée au modèle, pas un résultat ajusté.**

---

## 1️⃣4️⃣ BALAYAGE η — OUTPUT SUPPLÉMENTAIRE

Pour chaque η ∈ linspace(1.05, 5.0, 40) :
- recalculer r_d, D_M, ℓ₁ indépendamment
- noter : Saha et dc/dz ne dépendent pas directement de η
- η intervient uniquement dans H(z) régime Janus z < z_drag

Produire `janus_eta_scan.png` avec :
- **Gauche** : r_d(η) avec ligne horizontale cible 147 Mpc
- **Droite** : ℓ₁(η) avec ligne horizontale cible 220

Si aucune valeur η ∈ [1.05, 5] ne satisfait les critères → falsification documentée.
Si une valeur satisfait → η_VSL est une prédiction du modèle.

---

## 1️⃣5️⃣ TEST COMPLÉMENTAIRE — EXPOSANT n DE H(z)

### Objectif
Tester si `H(z) = H₀(1+z)^n` avec `1 < n ≤ 2` peut reproduire r_d ≈ 147 Mpc et ℓ₁ ≈ 220.

### Paramètres
```python
n_grid = np.linspace(1.0, 2.0, 21)   # 21 points
# Tout le reste inchangé : H0=76, Omega_b=0.0493, Saha exact, dc/dz identique
```

### Analyse théorique préalable
```
c_s/H ∝ c(z)^0.5 / (1+z)^n = (1+z)^(0.5 - n)
```
- n=1.0 → intégrande ∝ (1+z)^(-0.5) → divergence forte
- n=1.5 → intégrande ∝ (1+z)^(-1.0) → logarithmique → r_d fini mais grand
- n=2.0 → intégrande ∝ (1+z)^(-1.5) → convergence forte → r_d petit

### Calculs identiques aux étapes 7-9
```
r_d = ∫_{z_drag}^{z_max} c_s(z) / (H0*(1+z)^n) dz
D_M = ∫_0^{z_drag} c(z) / (H0*(1+z)^n) dz
ℓ₁  = π * D_M / r_d
```

### Outputs supplémentaires : `janus_n_scan.png`
- **Gauche** : r_d(n) avec ligne cible 147 Mpc
- **Droite** : ℓ₁(n) avec ligne cible 220
- Marquer le n critique où r_d = 147 (s'il existe)

### Ce que ce test dira
| Résultat | Interprétation |
|----------|----------------|
| ∃ n ∈ [1,2] tel que r_d ≈ 147 | Régime primordial doit être plus rapide que Milne pur |
| Aucun n ne fonctionne | Physique acoustique à repenser entièrement |
| n ≈ 1.5 fonctionne | Compatible avec domination radiation (ΛCDM n=3/2) |

### Note de falsification
Si n ≈ 1.5 donne r_d ≈ 147, ce n'est **pas** une validation de Janus —
c'est une contrainte : le régime primordial Janus doit se comporter comme
une domination radiation standard, ce qui nécessite une justification théorique
dans les équations VSL de Petit 1988/1995.
