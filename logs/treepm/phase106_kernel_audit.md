---
name: Phase 10.6 GPU Tree kernel audit
description: Audit ligne par ligne de forces_treepm_short_range_janus, identifie 2 bugs concurrents (r_s hardcodé r_cut/3 + erfc seul au lieu de Springel T(x))
type: project
---

# Audit ligne par ligne — kernel `forces_treepm_short_range_janus`

**Fichier** : `src/nbody_gpu_twopass.rs:2015-2153`
**État** : post-revert Phase 10.5 (kernel utilise erfcf seul)
**Branche** : `feat/treepm-jpp-port`
**Date** : 2026-04-29

## 1. Signature kernel

```c
extern "C" __global__ void forces_treepm_short_range_janus(
    const float* pos_all,                    // [n×3] toutes positions
    const signed char* signs_all,            // [n] signs ±1
    const float* node_pos,                   // BVH cells (7×f32 par node)
    const float* node_mass,                  // BVH cell masses
    const int* left, right, node_types,      // BVH structure
    float* acc,                              // [n×3] sortie accélération
    int n_all_signed,                        // |n_all|, signe = tree_sign
    float theta_soft_packed,                 // upper16: θ×1000, lower16: ε×1000
    float rcut_boxhalf_packed,               // upper16: r_cut, lower16: box_half
    float cross_packed                       // upper16: cmp×1000, lower16: cpm×1000
)
```

**Aucun paramètre `r_s`** — c'est la racine du problème (voir §6).

## 2. Décompactage paramètres (lignes 2026-2036)

```c
int packed_cross = __float_as_int(cross_packed);
float cross_minus_plus = (float)((packed_cross >> 16) & 0xFFFF) / 1000.0f;
float cross_plus_minus = (float)(packed_cross & 0xFFFF) / 1000.0f;

int packed_ts = __float_as_int(theta_soft_packed);
float theta = (float)((packed_ts >> 16) & 0xFFFF) / 1000.0f;
float softening = (float)(packed_ts & 0xFFFF) / 1000.0f;

int packed_rb = __float_as_int(rcut_boxhalf_packed);
float r_cut = (float)((packed_rb >> 16) & 0xFFFF);
float box_half = (float)(packed_rb & 0xFFFF);
```

Limitation cudarc 12-args → packing en `f32`. Les valeurs ×1000 perdent 3 décimales.
À θ=0.5, soft=0.05, cross=1.0, perte minime mais worth noting.

**`r_cut` est passé sans×1000** (limite 65535 Mpc).

## 3. **HARDCODE r_s — ligne 2038** ⚠️

```c
float r_s = r_cut / 3.0f;
```

**C'est le bug #1.** Le kernel calcule SON r_s à partir de r_cut, ignorant
totalement la convention du host.

| Convention | r_cut/r_s | Source |
|---|---|---|
| Kernel hardcoded | **3** | ligne 2038 |
| PhotoNs-GPU canonical | **5** | Wang & Meng 2021 |
| Springel/GADGET-2 typique | ~4.5 | MNRAS 364, 1105 |
| Test setup | 5 | `r_s = 1.2·dg, r_cut = 6·dg` |
| Host A.5a | 5 | identique test |
| Host CPU TreePM | 5 | `treepm_force.rs:r_s = r_cut/5` |

**Conséquence numérique** : avec `r_cut = 9.375` (test), kernel utilise
`r_s_kernel = 3.125` au lieu du `r_s_host = 1.875` attendu. Ratio 5/3 = 1.667.

## 4. sign_factor Janus (lignes 2055-2062)

```c
float sign_factor;
if (my_sign == tree_sign) {
    sign_factor = 1.0f;                   // Newton self-attraction
} else if (my_sign > 0) {
    sign_factor = -cross_plus_minus;      // m+ feels m- via repulsive cross
} else {
    sign_factor = -cross_minus_plus;      // m- feels m+ via repulsive cross
}
```

✅ **Convention Janus correcte**, matches `JanusCoupling.factor_for(...)` :
- m+/m+ : +1.0
- m+/m− : −φ·repulsion_scale (répulsif)
- m−/m+ : −c̄²/φ·repulsion_scale (répulsif, asymétrique)
- m−/m− : +1.0

A.5c PASS (3/3 cas Janus) confirme que ce bloc est correct.

## 5. Boucle BVH descente

### 5.1 MIC pour test ouverture cellule (lignes 2091-2099)

```c
float dx = cx-px, dy = cy-py, dz = cz-pz;
if (dx > box_half) dx -= 2.0f * box_half;   // MIC
if (dx < -box_half) dx += 2.0f * box_half;
... (idem dy, dz)
float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;
float r = sqrtf(r2);
```

✅ MIC appliqué proprement. Phase 10A.1 fix CPU avait reproduit ceci.

### 5.2 Skip cellule au-delà de r_cut (lignes 2101-2103)

```c
float closest_dist = r - hs;
bool cell_beyond_rcut = (closest_dist > r_cut);
if (__all_sync(0xFFFFFFFF, cell_beyond_rcut)) continue;
```

Optimisation correcte : si TOUTE la cellule est à distance > r_cut, skip.

### 5.3 Critère ouverture Barnes-Hut (ligne 2105)

```c
bool should_approx = (nt == 1) || ((4.0f*hs*hs) < (theta2*r2));
```

Standard BH : si feuille (nt==1) ou si `(2·hs)/r < θ`, approximation multipôle.

### 5.4 **Calcul force avec splitting (lignes 2118-2128)** ⚠️

```c
float ddx = comx-px, ...                    // MIC à nouveau sur COM
float rp2 = ddx*ddx + ddy*ddy + ddz*ddz + eps2;     // Plummer softening
float rp = sqrtf(rp2);
float erfc_arg = rp / (2.0f * r_s);                  // x = r/(2·r_s_kernel)
float erfc_factor = erfcf(erfc_arg);                 // ⚠️ erfc seul, pas T(x)

if (erfc_factor > 1e-6f) {
    float irp3 = 1.0f / (rp * rp2);                  // 1/r³
    float f = sign_factor * m * irp3 * erfc_factor;  // F_short = sign × m·erfc(x)/r³ × r̂
    ax += f*ddx; ay += f*ddy; az += f*ddz;
}
```

**C'est le bug #2.** Le kernel applique `erfc(x)` seul, alors que la convention
Springel pour PM `exp(-k²·r_s²)` nécessite `T(x) = erfc(x) + (2x/√π)·exp(-x²)`.

**Démonstration** : si PM applique `φ_long_k = -4πG/k² × ρ × exp(-k²·r_s²)`,
alors en réel `φ_long(r) = -G/r × erf(r/(2r_s))`. Donc :

```
F_long(r) = -∇φ_long = (G/r²) × [erf(x) - (2x/√π)·exp(-x²)]
F_short(r) = F_PP - F_long = (G/r²) × [erfc(x) + (2x/√π)·exp(-x²)]
                           = F_PP × T(x)                        ← formule correcte
```

Avec **erfc(x) seul**, on a F_short + F_long ≠ F_PP. Le terme `(2x/√π)·exp(-x²)`
manque dans le Tree, créant un déficit.

## 6. Vérification analytique pré-fix vs mesure

Convention test : `r_s_host = 1.875, r_cut = 9.375, x_host = r/(2·r_s_host)`.
Kernel : `r_s_kernel = r_cut/3 = 3.125, x_kernel = r/(2·r_s_kernel) = 0.6·x_host`.

Tree GPU pré-fix retourne : `F_short_GPU = F_PP × erfc(0.6·x_host)`
Test attend : `F_Tree_expected = F_PP × T(x_host)`

Donc : `ratio_Tree = erfc(0.6·x_host) / T(x_host)`

| r/r_s_host | x_host | 0.6·x_host | erfc(0.6·x_host) | T(x_host) | ratio prédit | ratio mesuré |
|---|---|---|---|---|---|---|
| 0.5 | 0.25 | 0.15 | 0.832 | 0.989 | **0.842** | 0.833 ✓ |
| 1.0 | 0.50 | 0.30 | 0.671 | 0.919 | **0.730** | 0.716 ✓ |
| 1.5 | 0.75 | 0.45 | 0.524 | 0.771 | **0.680** | 0.658 ✓ |
| 2.0 | 1.00 | 0.60 | 0.396 | 0.572 | **0.692** | 0.658 ✓ |
| 3.0 | 1.50 | 0.90 | 0.203 | 0.213 | **0.956** | 0.871 |

✅ **Match excellent** (l'écart à r/r_s=3 vient des fillers + multipôle BVH).

## 7. Vérification analytique post-fix Springel (Phase 10.5)

Avec `t_factor = T(x_kernel) = T(0.6·x_host)` :

`ratio_Tree = T(0.6·x_host) / T(x_host)`

| r/r_s_host | x_host | 0.6·x_host | T(0.6·x_host) | T(x_host) | ratio prédit | ratio mesuré post-fix |
|---|---|---|---|---|---|---|
| 0.5 | 0.25 | 0.15 | 0.998 | 0.989 | 1.009 | 1.007 ✓ |
| 1.0 | 0.50 | 0.30 | 0.973 | 0.919 | 1.059 | 1.143 |
| 1.5 | 0.75 | 0.45 | 0.937 | 0.771 | 1.215 | 1.209 ✓ |
| 2.0 | 1.00 | 0.60 | 0.872 | 0.572 | 1.524 | 1.492 ✓ |
| 3.0 | 1.50 | 0.90 | 0.654 | 0.213 | 3.071 | 2.940 ✓ |

✅ **Match excellent**. Le fix Springel a corrigé le bug #2 mais le bug #1
(r_s mismatch) reste — d'où la sur-correction qui croît avec r/r_s.

## 8. Vérification Total (cohérence PM/Tree)

Pré-fix : `F_total_GPU = F_PM + F_Tree = F_PP × (1 - T(x_host)) + F_PP × erfc(0.6·x_host)`

Donc : `ratio_Total = 1 - T(x_host) + erfc(0.6·x_host)`

| r/r_s | ratio prédit | ratio mesuré |
|---|---|---|
| 0.5 | 0.843 | 0.835 ✓ |
| 1.0 | 0.752 | 0.738 ✓ |
| 1.5 | 0.753 | 0.734 ✓ |
| 2.0 | 0.824 | 0.801 ✓ |
| 3.0 | 0.990 | 0.971 ✓ |

✅ **Match parfait**. Le PM est correct (ratio_PM ≈ 0.99) et le Tree GPU est
exactement `F_PP × erfc(r/(2·r_cut/3))`.

## 9. Comparaison avec CPU `pairwise_acc_with_split`

| Aspect | CPU | GPU Janus | Match ? |
|---|---|---|---|
| splitting func | `splitting_tree_springel(r, r_s)` = T(r/(2r_s)) | `erfcf(r/(2·r_cut/3))` | ❌ DIVERGENT |
| arg passé | `r, r_s` (deux paramètres) | `r, r_cut` (r_s dérivé) | ❌ DIVERGENT |
| sign_factor Janus | JanusCoupling | identique | ✅ |
| MIC périodique | oui | oui | ✅ |
| Softening | Plummer ε² | Plummer ε² | ✅ |
| r_cut cutoff | r > r_cut → skip | identique | ✅ |

**Deux divergences fondamentales** : la convention de splitting (formule erfc vs T)
ET le scale (r_cut/3 vs r_s_host).

## 10. Vérification cohérence PM ↔ Tree (côté host `step_treepm_gpu_cosmo`)

Lignes 5260-5400 :
- `step_treepm_gpu_cosmo(.., r_cut, r_s, ..)` accepte r_cut ET r_s en paramètres
- PM : `solve_device(.., r_s)` → utilise r_s_host pour smoothing `exp(-k²·r_s_host²)`
- Tree : `compute_short_range_forces_janus(r_cut, ..)` → kernel utilise r_s_kernel = r_cut/3

**Donc PM et Tree utilisent des r_s DIFFÉRENTS dans le pipeline production**,
et ceci dès l'origine (pas un bug Phase 10.5 spécifique).

## 11. Cohérence interne du pipeline historique `step_treepm_gpu` (non-Janus, ligne 5077)

Ligne 5154 : `let r_s = r_cut / 3.0;` — host calcule lui aussi r_cut/3 et le
passe à cuFFT. Donc dans `step_treepm_gpu` (non-Janus, pré-Phase 10) :
- PM utilise r_s = r_cut/3 (host)
- Tree utilise r_s = r_cut/3 (kernel hardcodé)
- ✅ COHÉRENT entre PM et Tree (même mauvaise valeur des deux côtés)

Mais Tree applique erfc(x) seul → bug #2 toujours présent. Décomposition
mathématique imparfaite (sum ≠ F_PP).

**`step_treepm_gpu_cosmo` (Phase 10A.4) introduit l'incohérence supplémentaire**
en passant `r_s` librement au PM, qui peut différer de `r_cut/3`.

## 12. Vérification PM seule (`compute_pm_only_janus`)

Lignes 4798-4902 :
- Accepte `r_s` paramètre
- `solve_device(.., r_s)` : applique smoothing avec r_s passé
- `cic_gather_janus` : gradient 2-point centré, pas de facteur extra
- ratio_PM ≈ 0.99 mesuré → PM correct par rapport à `F_PP × (1 - T(x))`
  avec x = r/(2·r_s_host)

✅ **PM est correct vis-à-vis du test reference**. Le test mesure PM contre
`F_PP × (1 - T(x_host))`, et obtient 0.99. PM utilise le bon r_s (parce qu'il
le reçoit en paramètre).

## 13. Convention de référence à appliquer

Pour cohérence Springel et compatibilité PhotoNs/CPU validé :

```
F_short(r) = G·m·m'/r² × T(r/(2·r_s))     (Tree, Springel)
F_long(r)  = G·m·m'/r² × [erf(x) - (2x/√π)·exp(-x²)]    (PM real-space)
F_short + F_long = G·m·m'/r²              (= F_PP, exact)

avec x = r/(2·r_s)
r_s = même valeur des deux côtés
```

Le test `test_phase105_decomp` utilise exactement cette convention avec
`r_s = 1.2·dg = 1.875`. C'est cohérent avec PhotoNs-GPU et CPU
`splitting_tree_springel`.

## 14. Récapitulatif

| Bug | Localisation | Description | Impact mesuré |
|---|---|---|---|
| #1 | nbody_gpu_twopass.rs:2038 et 1882 | `r_s = r_cut/3` hardcodé dans kernel(s) | x_kernel = 0.6·x_host (factor 5/3 mismatch) |
| #2 | nbody_gpu_twopass.rs:2122 et 1966 | `erfcf(x)` au lieu de `T(x)` | terme (2x/√π)·exp(-x²) absent |

**Pour avoir Tree = F_PP × T(x_host), il faut fixer LES DEUX :**
1. Passer `r_s` en paramètre au kernel (au lieu de le dériver de r_cut)
2. Calculer `t_factor = erfcf(x) + (2x/√π)·exp(-x²)` au lieu d'`erfcf(x)` seul

Phase 10.5 a fixé le bug #2 mais le bug #1 a augmenté l'erreur visible
(au lieu de la réduire), parce que avec erfc seul + r_s_kernel trop grand,
les deux erreurs se partiellement compensaient.
