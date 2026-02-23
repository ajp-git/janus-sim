# Analyse Linéaire Janus — Théorie des Perturbations à Deux Fluides
# Source : ChatGPT o3 — 23 février 2026
# Pour : Jean-Pierre Petit / dossier Janus

---

## 1. Système linéarisé à deux composantes

Densités moyennes : ρ̄₊, ρ̄₋
Contrastes de densité : δ₊, δ₋
Couplage gravitationnel croisé : α (répulsion Janus)

Équations du mouvement en régime linéaire (espace de Fourier, mode k) :

    δ̈₊ + 2H δ̇₊ = 4πG (ρ̄₊ δ₊ − α ρ̄₋ δ₋)
    δ̈₋ + 2H δ̇₋ = 4πG (ρ̄₋ δ₋ − α ρ̄₊ δ₊)

Le signe − devant α encode la répulsion croisée Janus.

---

## 2. Forme matricielle

D = (δ₊, δ₋)ᵀ

    D̈ + 2H Ḋ = 4πG M D

avec la matrice de couplage :

         ┌  ρ̄₊    −α ρ̄₋ ┐
    M =  │               │
         └ −α ρ̄₊   ρ̄₋  ┘

---

## 3. Valeurs propres de M

det(M − λI) = 0 donne :

    λ± = (ρ̄₊ + ρ̄₋)/2 ± (1/2) √[(ρ̄₊ − ρ̄₋)² + 4α² ρ̄₊ ρ̄₋]

La croissance gravitationnelle existe si au moins une valeur propre est positive.
λ₊ > 0 toujours.
Le signe de λ₋ détermine le comportement du second mode.

---

## 4. Cas symétrique — condition critique

Pour ρ̄₊ = ρ̄₋ = ρ̄ :

    λ₊ = ρ̄ (1 + α)    → toujours instable si α > −1
    λ₋ = ρ̄ (1 − α)    → signe dépend de α

### CONDITION EXACTE

    α < 1  →  λ₋ > 0  →  les deux modes croissent  →  filaments possibles
    α = 1  →  λ₋ = 0  →  mode neutre  →  croissance partiellement bloquée
    α > 1  →  λ₋ < 0  →  mode oscillatoire  →  suppression des structures

---

## 5. Application au modèle Janus (η=1.045)

Dans notre simulation : ρ̄₋/ρ̄₊ ≈ 0.69/0.31 ≈ 2.23

Le code implémente α = 1 (répulsion symétrique) :
    interaction = if sign_i == sign_j { 1.0 } else { -1.0 }

Avec β = 1 (anti-corrélation parfaite δ₋ = −δ₊) :

    G_eff = G × (1 − α × β × ρ̄₋/ρ̄₊)
          = G × (1 − 1 × 1 × 2.23)
          = −1.23 G

**G_eff < 0 → oscillations type plasma gravitationnel**

Conclusion : avec les paramètres actuels, le modèle Janus est
**exactement au cas critique α = 1**, voire en régime suppressif
pour le mode λ₋ compte tenu de l'asymétrie ρ̄₋ > ρ̄₊.

---

## 6. Tableau de synthèse

| α     | Dynamique                                  | Filaments |
|-------|--------------------------------------------|-----------|
| α < 1 | Croissance normale des deux modes          | ✅ Possibles |
| α = 1 | Croissance marginale (cas Janus actuel)    | ⚠️ Fragiles |
| α > 1 | Oscillations + suppression des contrastes  | ❌ Supprimés |

---

## 7. Test minimal discriminant

### Principe : mode unique anisotrope

Initialiser δ(x) = A cos(kx · x) — un seul mode plan, direction x.

Implémentation Rust :
```rust
let kx = 2.0 * PI / box_size;
let amplitude = 0.02 * box_size;
for p in particles.iter_mut() {
    p.pos.x += amplitude * (kx * p.pos.x).sin();
}
```

### Deux runs identiques

- **Run A** : gravité attractive pure (pas de Janus, α=0)
- **Run B** : Janus complet (α=1)

### Métrique discriminante

```rust
fn compute_sigma(particles: &[Particle]) -> (f64, f64, f64) {
    let n = particles.len() as f64;
    let sx = particles.iter().map(|p| p.pos.x * p.pos.x).sum::<f64>() / n;
    let sy = particles.iter().map(|p| p.pos.y * p.pos.y).sum::<f64>() / n;
    let sz = particles.iter().map(|p| p.pos.z * p.pos.z).sum::<f64>() / n;
    (sx.sqrt(), sy.sqrt(), sz.sqrt())
}

let anisotropy = sigma_x / ((sigma_y + sigma_z) * 0.5);
```

### Interprétation

| anisotropy(t) | Diagnostic |
|---------------|------------|
| >> 1 et croissant | Croissance anisotrope OK → problème numérique (ICs/boîte) |
| ≈ 1 constant | Isotropisation → Janus supprime la croissance anisotrope |
| décroissant | Suppression dynamique → α > 1 effectif |

### Paramètres du test

- N = 4M (suffisant)
- box = 400 Mpc
- θ = 1.5
- dt = 0.005
- Durée : 3-4h max

---

## 8. Questions ouvertes pour JPP

1. **Quelle est la valeur exacte de α dans le modèle Janus de Petit ?**
   Le code implémente α=1 mais le papier EPJC 2024 donne-t-il une valeur différente ?

2. **L'expansion cosmique (H≠0) peut-elle sauver la croissance pour α=1 ?**
   ChatGPT propose de dériver la fréquence des oscillations et le temps
   caractéristique de suppression — à vérifier.

3. **Le facteur η=1.045 modifie-t-il α effectivement ?**
   η intervient dans l'équation de Friedmann mais pas directement dans le
   couplage gravitationnel croisé — à clarifier avec JPP.

4. **P(k) Janus exact :**
   Si λ₋ < 0, aucun P(k) ne donnera de toile cosmique.
   Si λ₋ ≥ 0, P(k) ΛCDM approximé suffit.
   Le test mode unique répond à cette question en 1 jour.

---

## 9. Prochaines étapes (ChatGPT propose)

- Dériver la fréquence exacte des oscillations pour α > 1
- Relier au temps caractéristique de suppression
- Analyser si l'expansion cosmique H(t) peut maintenir la croissance
  même pour α ≈ 1 (amortissement Hubble vs suppression Janus)

---

## Source

ChatGPT o3, session 23 février 2026
Dans le cadre du projet Janus — simulation cosmologique N-corps GPU
Contact : Jean-Pierre Petit (EPJC 84:1226, 2024)

---

## 10. Rôle de l'expansion cosmique (H≠0) pour α≈1

### Résultat fondamental

Pour α=1 exactement, l'équation du mode λ₋ devient :

    δ̈ + 2H δ̇ = 0  (terme gravitationnel nul)

Solution générale en univers matière dominée (a ∝ t^(2/3)) :

    δ(t) = C₁ + C₂ t^(-1/3)

**Un mode constant + un mode décroissant. Aucune croissance.**

### Pour α = 1−ε (légèrement inférieur)

    δ ∝ a^p    avec    p ≈ (3/5) ε

| ε = 1−α | Exposant p | Croissance |
|---------|------------|------------|
| 1.0     | 0.6        | ΛCDM normal |
| 0.1     | 0.06       | Quasi nulle |
| 0.01    | 0.006      | Gelée |

**Si α = 0.99 → δ ∝ a^0.006 → croissance quasi nulle même avec expansion.**

### Pour α > 1

    δ̈ + 2H δ̇ + ω² δ = 0  (oscillateur amorti)

Oscillations décroissantes. Pas de croissance structurelle.

### Conclusion

**Non. L'expansion cosmique ne peut pas sauver la croissance si α ≈ 1.**

L'expansion ralentit une croissance existante mais ne peut pas
en créer une si G_eff ≈ 0.

---

## 11. Implication critique pour le code Janus actuel

Le code implémente :
```rust
let interaction = if sign_i == sign_j { 1.0 } else { -1.0 };
```

→ α = 1 **strictement**

**Conclusion : le mode λ₋ est exactement neutre dans notre simulation.**

La ségrégation observée (S_max=0.694) provient uniquement du mode λ₊
(attraction intra-signe). Les filaments nécessitent le mode λ₋ pour
développer la croissance anisotrope.

**Ce n'est pas un problème numérique. C'est une propriété du modèle Janus.**

---

## 12. Questions pour JPP

1. **α=1 est-il exact dans le modèle de Petit, ou y a-t-il une asymétrie ?**
   Le papier EPJC 2024 donne-t-il une valeur de α différente de 1 ?

2. **L'asymétrie ρ₋/ρ₊ ≈ 2.23 (η=1.045) change-t-elle α effectif ?**
   Si le couplage croisé n'est pas symétrique en densité, α_eff ≠ 1.

3. **Janus prédit-il une toile cosmique ?**
   Si α=1 strictement → pas de croissance linéaire anisotrope →
   pas de filaments naturels. Comment JPP explique-t-il les grandes
   structures observées dans son modèle ?

---

## 13. Test discriminant final

Run A : attraction pure (α=0) → filaments attendus
Run B : Janus α=1 → pas de filaments attendus (théoriquement)

Si Run B confirme l'absence de filaments → résultat théorique validé
expérimentalement → papier avec JPP.

Si Run B montre des filaments malgré α=1 → notre analyse linéaire
est incomplète (effets non-linéaires, asymétrie ρ).


---

## 14. Effet de l'asymétrie de densité ρ₊ ≠ ρ₋ (ChatGPT o3)

### Résultat spectaculaire

Avec la matrice générale et α=1 quelconque ρ₊, ρ₋ :

    (1−r)² + 4r = (1+r)²  (identité algébrique exacte)

Donc :

    λ₊ = ρ̄₊ (1+r)    (r = ρ₋/ρ₊ = 2.23 pour η=1.045)
    λ₋ = 0            EXACTEMENT, indépendant de r

**L'asymétrie de densité ne change rien au cas critique α=1.**
λ₋=0 est garanti algébriquement quelle que soit la valeur de r.

### Mais pour α = 1−ε (légèrement < 1)

    λ₋ ≈ (ρ̄₊ ρ̄₋)/(ρ̄₊ + ρ̄₋) × 2ε

Avec η=1.045 : ρ̄₊=0.31, ρ̄₋=0.69 :

    préfacteur = (0.31 × 0.69)/(0.31 + 0.69) = 0.214

    λ₋ ≈ 0.214 × 2ε = 0.43ε

Petit mais positif — la croissance réapparaît dès ε > 0.

### Tableau de synthèse

| Situation | λ₋ | Croissance |
|-----------|-----|------------|
| α=1, ρ₊=ρ₋ | 0 (critique) | Fragile |
| α=1, ρ₊≠ρ₋ | 0 (toujours!) | Idem |
| α<1, ρ₊≠ρ₋ | 0.43ε > 0 | Restaurée |
| α>1 | < 0 | Oscillations |

### Implication code

Le code actuel implémente α=1 strict :
```rust
let interaction = if sign_i == sign_j { 1.0 } else { -1.0 };
```
→ λ₋ = 0 garanti, indépendant de η ou de ρ̄₋/ρ̄₊.

### Test proposé (ChatGPT)

| Run | ρ₋/ρ₊ | α   | λ₋ attendu |
|-----|--------|-----|------------|
| A   | 1.00   | 0   | ρ̄ (ΛCDM pur) |
| B   | 2.23   | 1.0 | 0 (Janus actuel) |
| C   | 2.23   | 0.95| 0.43×0.05 = 0.022 |
| D   | 2.23   | 0.90| 0.43×0.10 = 0.043 |

### Question ouverte cruciale

**α=1 est-il une contrainte dure du modèle de Petit, ou une liberté ?**

Si α est fixé à 1 par la théorie → pas de filaments possibles en régime linéaire.
Si α peut varier → ε=0.05-0.10 suffit à restaurer la croissance.

À soumettre à JPP avant de modifier le code.


---

## 15. ρ₊ ≠ ρ₋ viole-t-il un principe fondamental de Janus ? (ChatGPT o3)

### Distinction cruciale

**Symétrie des équations** (niveau fondamental) ≠ **Symétrie des solutions**

Un Lagrangien symétrique n'impose pas que l'univers reste symétrique.
Exemple classique : brisure spontanée de symétrie dans un potentiel mexicain.

### Conclusion conceptuelle

    ρ₊ ≠ ρ₋ ne viole pas nécessairement Janus.

Cela viole une symétrie de solution, pas une symétrie des équations.

### Point critique non générique

Le point (ρ₊=ρ₋, α=1) est un point critique :
- instable
- hypersensible aux perturbations
- non générique

Un univers réaliste ne resterait jamais exactement dessus.
La moindre fluctuation statistique entraîne ρ₊ ≠ ρ₋ localement.

### Implication physique élégante

Si (ρ₊=ρ₋, α=1) → suppression structurelle → alors la formation de
structure exige une brisure spontanée de symétrie.

**C'est physiquement très élégant et cohérent avec la cosmologie standard.**

### Implication pour le code

Tester une asymétrie 5-10% ne trahit pas Janus — cela teste si le
modèle est coincé sur un point critique non générique.

### Questions suivantes (ChatGPT propose)

1. La dynamique non-linéaire produit-elle spontanément cette asymétrie ?
2. L'inflation pourrait-elle la générer dans le modèle Janus ?
3. Combien d'asymétrie minimale pour rétablir la croissance observable ?


---

## 15. Asymétrie minimale pour croissance observable (ChatGPT o3)

### Fonction de croissance pour α = 1−ε, ρ₊ ≠ ρ₋

    δ ∝ a^p    avec    p ≈ (3/5) × [2r/(1+r)²] × ε

La fonction f(r) = 2r/(1+r)² est maximale en r=1 : f(1) = 0.5

### Condition pour p ≥ 0.5 (croissance "observable")

    0.5 × ε ≥ 0.83  →  ε ≥ 1.66  →  IMPOSSIBLE

**Conclusion : aucune asymétrie de densité ne peut restaurer une croissance
rapide si α est trop proche de 1.**

### Valeur critique de α

    ε = 1−α ≳ 0.3 à 0.5
    →  α ≲ 0.5 à 0.7

L'asymétrie ρ₊ ≠ ρ₋ ne compense qu'un facteur 0.5 au maximum.

### Tableau récapitulatif final

| α    | ε=1−α | p (croissance) | Filaments |
|------|-------|----------------|-----------|
| 1.0  | 0     | 0              | ❌ |
| 0.95 | 0.05  | ~0.03          | ❌ quasi nulle |
| 0.7  | 0.3   | ~0.18          | ⚠️ lente |
| 0.5  | 0.5   | ~0.30          | ⚠️ marginale |
| 0.0  | 1.0   | 0.6 (ΛCDM)     | ✅ |

### Recommandation pratique

Mesurer α effectif depuis les sorties numériques (test mode unique) AVANT
de modifier les densités. Si α_eff ≥ 0.8 → l'asymétrie ne changera rien.


---

## 16. Régime non-linéaire — peut-il rattraper une croissance lente ? (ChatGPT o3)

### Temps pour atteindre la non-linéarité (δ~1)

Depuis δᵢ ~ 10⁻⁵ :

    a^p ~ 10⁵  →  a ~ 10^(5/p)

| p    | a requis | Réaliste ? |
|------|----------|------------|
| 1.0  | 10⁵      | ✅ ΛCDM normal |
| 0.3  | 10^16    | ❌ |
| 0.1  | 10^50    | ❌ jamais |

### Conclusion ferme

**Si p ≲ 0.2, le régime non-linéaire ne sauvera pas la filamentation.**

Le régime non-linéaire est multiplicatif, pas générateur.
Il amplifie une croissance existante, ne la crée pas.

### Pourquoi on observe S_max=0.694 malgré α=1

La ségrégation observée provient du **mode λ₊** (toujours instable) :
- λ₊ = ρ̄₊(1+r) > 0 → attraction intra-signe → clusters de masses+ et masses−
- Ce mode crée le blob sphérique et la ségrégation globale S(t)

Le mode λ₋ = 0 (neutre) contrôle la **croissance anisotrope** → filaments.
Ces deux phénomènes sont indépendants.

**On peut avoir S_max élevé ET pas de filaments — c'est exactement ce qu'on observe.**

### Synthèse finale — table de décision

| α    | p    | Ségrégation S | Filaments |
|------|------|---------------|-----------|
| 1.0  | 0    | ✅ (mode λ₊)  | ❌        |
| 0.7  | ~0.18| ✅            | ⚠️        |
| 0.5  | ~0.30| ✅            | ⚠️        |
| 0.0  | 0.6  | ✅            | ✅        |

### Prochaine étape recommandée

Mesurer p expérimentalement via le test mode unique anisotrope.
Comparer Run A (α=0) vs Run B (α=1) → déduit α effectif réel.


---

## 17. Spectre non-gaussien — peut-il contourner le blocage ? (ChatGPT o3)

### Résultat

Même avec δᵢ ~ 10⁻² (non-gaussien fort) et p=0.1 :

    a^0.1 = 100  →  a = 10²⁰  →  impossible

Si δᵢ ~ 1 (déjà non-linéaire) et G_eff ≈ 0 :
- Les structures injectées ne s'amplifient pas
- Elles se diluent avec l'expansion
- Artefact initial, pas croissance physique

### Conclusion

**Un spectre non gaussien ne compense pas un G_eff faible.**

La toile cosmique est une propriété de la dynamique gravitationnelle,
pas du spectre initial. Le spectre déclenche ce que la dynamique permet.

### Synthèse globale de l'analyse linéaire Janus

**Ce qui crée la ségrégation S(t) :** mode λ₊ = ρ̄₊(1+r) > 0 toujours
→ attraction intra-signe → blob sphérique → S_max observable ✅

**Ce qui créerait les filaments :** mode λ₋ = ρ̄(1−α)
→ α=1 dans le code → λ₋=0 → croissance anisotrope nulle ❌

**Ce qui ne peut pas aider :**
- Asymétrie ρ₊ ≠ ρ₋ (λ₋=0 algébriquement pour α=1)
- Expansion cosmique H(t) (amortit mais ne crée pas)
- Spectre non gaussien (symptôme, pas cause)
- Régime non-linéaire (multiplicatif, pas générateur)

### La seule solution : modifier α

α ≲ 0.5-0.7 pour p ≥ 0.3 (croissance marginale)
α = 0 (ΛCDM pur) pour p = 0.6 (filaments garantis)

### Question fondamentale pour JPP

α=1 dans le code Janus n'est pas un choix numérique — c'est la physique.
La répulsion croisée m+↔m− est symétrique par construction.

**Si α=1 est une contrainte dure du modèle de Petit :**
→ Le modèle Janus ne peut pas produire de filaments cosmologiques
  par instabilité gravitationnelle linéaire
→ Les grandes structures observées ont une autre origine dans Janus
→ C'est un résultat théorique nouveau à soumettre à JPP


---

## 18. α(k) dépendant de l'échelle — piste élégante (ChatGPT o3)

### Principe

    α(k) = 1 − ε × k²/(k² + kc²)

- k → 0 (grandes échelles) : α ≈ 1 → symétrie Janus préservée
- k ~ k_struct : α ≈ 1−ε → gravité effective restaurée → filaments

### Condition pour filamentation

    1 − α(k_struct) ≥ 0.3  →  ε ≥ 0.3

### Facteur de croissance résultant

    p(k) ∝ [1 − α(k)] = ε × k²/(k² + kc²)

Aux échelles structurelles (k ~ kc) : p ≈ ε/2 ≈ 0.15-0.20 — marginal mais possible.

### Cohérence physique avec Janus

Oui si le couplage inter-secteur est médié par un champ massif :
- Symétrie exacte en IR (grandes échelles)
- Brisure partielle en UV (petites échelles)
- Analogue aux théories de gravité Yukawa / médiateur massif

### Implémentation numérique possible

En espace réel, α(k) → opérateur différentiel :

    F(r) = interaction × [1 − ε × exp(−r/rc)]

où rc = 2π/kc est l'échelle de transition physique.

Dans le kernel CUDA :
```rust
let r = dist.sqrt();
let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
let interaction = if sign_i == sign_j { 1.0 } else { -alpha_r };
```

### Paramètres à tester

    ε = 0.3 à 0.5
    rc = box/10 à box/3  (quelques dizaines de Mpc)

### Question fondamentale pour JPP

Le modèle de Petit (EPJC 2024) permet-il un α(k) ou impose-t-il α=1 à toutes échelles ?
Un médiateur massif ou terme ∇² dans l'action bimétrique ?
Si oui → α(k) est une extension naturelle du modèle.
Si non → α=1 strict → pas de filaments par instabilité linéaire.


---

## 19. Stabilité complète avec α(k) (ChatGPT o3)

### Valeurs propres mode par mode

    λ±(k) = (ρ̄₊+ρ̄₋)/2 ± (1/2)√[(ρ̄₊−ρ̄₋)² + 4α(k)²ρ̄₊ρ̄₋]

### Stabilité globale

    det M(k) = ρ̄₊ρ̄₋(1 − α(k)²)

Si |α(k)| ≤ 1 pour tout k → det > 0 → pas d'instabilité pathologique.

### Spectre de stabilité

| α(k)    | Dynamique               |
|---------|-------------------------|
| < 1     | Croissance              |
| = 1     | Neutre                  |
| > 1     | Oscillations amorties   |

### Avec α(k) = 1 − ε k²/(k²+kc²)

- k → 0 : α → 1, λ₋ → 0 (grandes échelles neutres)
- k >> kc : α → 1−ε, λ₋ → ρ̄ε (petites échelles croissent)

**Instabilité sélective en échelle** — exactement ce qu'on veut.

### Condition pour toile cosmique

1. Bande en k où 1−α(k) ≳ 0.3
2. Cette bande couvre les échelles 1-20 Mpc (ou équivalent boîte)
3. Pas de région avec α(k) > 1

### Implémentation

α(k) en espace réel = potentiel Yukawa :

    V(r) ∝ (1/r)(1 − exp(−kc·r))

```rust
let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
let interaction = if sign_i == sign_j { 1.0 } else { -alpha_r };
```

Compatible Barnes-Hut — pas besoin de PM.

### Valeur de kc à estimer

kc doit correspondre aux échelles structurelles de la boîte.
Pour box=400 Mpc : kc ~ 2π/40 Mpc⁻¹ (mode de turnover).


---

## 20. Compatibilité avec la relativité effective Janus (ChatGPT o3)

### Trois interprétations de α(k)

**A — Médiateur massif (compatible ✅)**
Interaction +/− médiée par un champ de masse m :
    propagateur ~ 1/(k²+m²)  →  α(k) ~ k²/(k²+m²)
Parfaitement cohérent avec une théorie relativiste.

**B — Interaction non locale effective (compatible ✅)**
Opérateur α(□) dans une théorie effective covariante.
Courant en gravité modifiée.

**C — Modification phénoménologique sans Lagrangien (❌)**
Perd la cohérence relativiste.

### Conditions de compatibilité

1. Causalité : α(k) issu d'un opérateur local/quasi-local
2. Invariance de jauge : cohérence dynamique des deux secteurs
3. Pas de mode fantôme : |α(k)| ≤ 1 pour tout k

### Conclusion

    Un α(k) de type Yukawa est compatible avec la relativité effective Janus
    s'il correspond à un médiateur massif ou opérateur covariant.

    Ce n'est pas une rustine — c'est une extension naturelle du modèle.

### Interprétation physique

- Symétrie Janus exacte à grande échelle (α→1 en IR)
- Interactions inter-secteurs à portée finie (médiateur massif)
- Neutralité cosmique globale + gravité locale restaurée

### Prochaines étapes possibles

1. Dériver le Lagrangien minimal produisant cet α(k)
2. Contraintes observationnelles (CMB, LSS) sur kc et ε
3. Masse du médiateur correspondant à kc de la simulation :
   m ~ ℏkc/c ~ ℏ/(rc)


---

## 21. Validation par Gemini 2.0 (lecture de janus_linear_analysis.md)

Gemini confirme indépendamment les conclusions de ChatGPT après lecture du document.

### Points de convergence

**Blob inévitable avec α=1 :**
- Mode λ₊ → ségrégation (blob) ✅
- Mode λ₋ = 0 → croissance anisotrope gelée ❌
- 32M particules + boîte plus grande = blob plus grand, pas de filaments

**Plan 5 jours convergent :**

| Jour | Action |
|------|--------|
| 1 | Test mode unique anisotrope (validation expérimentale α=1) |
| 2 | Implémentation α(k) Yukawa dans kernel CUDA |
| 3 | ICs Zel'dovich + P(k) filtré |
| 4-5 | Run 32M, box=400 Mpc, θ=1.0 |

**Paramètres cibles (Gemini) :**

| Paramètre | Actuel | Cible |
|-----------|--------|-------|
| α | 1.0 strict | α(k) Yukawa ε=0.3 |
| ICs | Aléatoire uniforme | Zel'dovich + P(k) |
| Box | 100 unités | 400-500 Mpc |
| N | 8M | 16-32M |

**Code Yukawa proposé (Gemini) :**
```rust
let r_c = 40.0;      // Échelle de transition (Mpc)
let epsilon = 0.3;   // Force de la brisure
let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
let interaction = if sign_i == sign_j { 1.0 } else { -alpha_r };
```

### Option alternative (si α=1 intouchable)

Brisure spontanée de symétrie de densité : ρ₋/ρ₊ >> 1.
Gemini note honnêtement : l'analyse linéaire prédit λ₋=0 même avec asymétrie.
Seuls les effets non-linéaires à courte portée pourraient aider.
→ Faible probabilité, mais testable rapidement.

### Consensus final 5 IA (Grok, ChatGPT, Gemini, DeepSeek + analyse linéaire)

**α(k) Yukawa + ICs Zel'dovich + box 400 Mpc = seule voie viable pour les filaments.**


---

## 22. Validation par DeepSeek R1 (lecture de janus_linear_analysis.md)

DeepSeek confirme indépendamment l'ensemble de l'analyse après lecture du document.

### Points clés confirmés

- λ₋ = 0 algébriquement pour α=1, quelle que soit l'asymétrie ρ₊/ρ₋
- L'asymétrie ρ₋/ρ₊=2.23 (η=1.045) ne modifie pas α dans les équations de perturbation
- η intervient dans Friedmann mais pas dans le couplage croisé
- Le régime non-linéaire ne peut pas créer de filaments depuis λ₋=0
- α(k) Yukawa est compatible avec un médiateur massif spin-2 ou scalaire

### Point nouveau (DeepSeek)

**ε=0.1 (α=0.9) insuffisant** — p~0.06, trop lent même sur run long.
Nécessite ε ≥ 0.3 pour p ≥ 0.2 (minimum marginal avec non-linéaire).

### Recommandations DeepSeek

1. Test mode unique anisotrope → baseline expérimental
2. Test α=0.9 court (500K, 500 steps) → confirme p trop faible
3. Implémenter α(r) Yukawa (ε=0.3, rc=40 Mpc)
4. Dériver le Lagrangien correspondant pour cohérence avec l'action Janus
5. Run complet 16-32M avec ICs Zel'dovich + box 400 Mpc

### Consensus FINAL — 5 IA (Grok, ChatGPT, Gemini, DeepSeek + analyse linéaire)

**Conclusion unanime :**
Le modèle Janus α=1 ne peut pas produire de filaments cosmologiques
par instabilité gravitationnelle linéaire. Ce n'est pas un artefact numérique.

**Solution unanime :**
α(k) Yukawa + ICs Zel'dovich + box 400 Mpc + N=16-32M

**Question pour JPP :**
α=1 est-il une contrainte fondamentale, ou une extension α(k) est-elle
permise par les principes fondateurs de Janus ?

