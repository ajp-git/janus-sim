# Optimisation BH GPU — Suite (Post-Morton)
# Instructions pour Claude CLI — Février 2026

---

## ÉTAT ACTUEL (mesuré)

| Opt | Description      | Temps/step 2M | Speedup cumulé | Statut     |
|-----|-----------------|---------------|----------------|------------|
| 0   | KDK baseline    | 7810 ms       | 1.0×           | ✅ référence |
| 1   | DKD intégrateur | 3868 ms       | 2.0×           | ✅ DONE     |
| 2   | Morton sorting  | 2662 ms       | 2.9×           | ✅ DONE     |

8M estimé actuel : 19.5h → objectif : maximiser, horizon 16M en 12h

---

## RÈGLES ABSOLUES

```
1. Valider sur 500K avant de tester sur 2M
2. Comparer S(t) et KE/KE₀ vs run référence après chaque optimisation
3. S'ARRÊTER après chaque optimisation, rapporter, attendre instruction
4. Ne pas modifier src/ sans backup
5. Ne JAMAIS : docker stop $(docker ps -q) ou docker system prune
6. Mettre à jour janus_roadmap.md après chaque étape validée
7. Implémenter TOUTES les optimisations — ne pas s'arrêter si objectif atteint
```

---

## OPTIMISATION 3 — Asymmetric θ (5 lignes, spécifique Janus)

### Principe (Gemini — physiquement motivé)

Dans Janus, la force cross (masses+/masses−) est **répulsive** et crée
des voids à grande échelle → force naturellement basse fréquence.
La force self (même signe) est attractive → structures fines, filaments.

Utiliser deux θ différents dans le même kernel :
- θ_self  = 0.5 (attractif)  → haute précision pour les filaments
- θ_cross = 1.0 (répulsif)   → approximation grossière suffisante pour les voids

Divise le nombre de nœuds visités pour les interactions croisées par ~4.

### Modification dans compute_forces_simple (kernel CUDA)

Remplacer :
```cuda
double s_over_r = (2.0 * half_size) / r;
if (node_type == 1 || s_over_r < theta) {
```

Par :
```cuda
// Asymmetric theta : cross-interactions use larger opening angle
double node_mass_same  = (my_sign > 0) ? mass_plus  : mass_minus;
double node_mass_cross = (my_sign > 0) ? mass_minus : mass_plus;
double theta_eff = (node_mass_cross > node_mass_same * 0.1) ? theta * 2.0 : theta;
double s_over_r = (2.0 * half_size) / r;
if (node_type == 1 || s_over_r < theta_eff) {
```

### Validation

```
Test 500K, 200 steps vs référence Morton+DKD :
  ✓ S(200) identique ±5%
  ✓ KE/KE₀ identique ±1%
  ✓ Temps/step 2M : cible < 1800 ms
```

**STOP — rapporter, attendre instruction.**

---

## OPTIMISATION 4 — GPU tree build (Karras 2012)

### Principe

Remplacer `LinearOctree::build()` CPU par pipeline GPU complet.
**Zéro transfert CPU↔GPU dans la boucle force.**

Bottleneck actuel : CPU tree build O(N log N) + 2× PCIe ~500 MB/step.

### Pipeline GPU

```
a) Kernel Morton GPU  : clés 30-bit par particule      O(N)
b) CUB RadixSort GPU  : trier par clé Morton           O(N)  ← GPU, pas rayon CPU
c) Kernel Karras      : construire BVH depuis clés      O(N)
d) Kernel réduction   : calculer COMs +/- en parallèle  O(N log N)
```

Note : le Morton actuel (opt2) est sur CPU (rayon). Opt4 le remplace
par un Morton + sort entièrement GPU — gain supplémentaire sur opt2.

### Structure BVH GPU

```cuda
struct BVHNode {
    float4 com_plus;     // x,y,z,mass masses+
    float4 com_minus;    // x,y,z,mass masses-
    float  half_size;
    int    left, right;  // -1 = feuille
    int    particle_idx;
};
```

### Validation

```
✓ S(t) identique ±5% vs référence
✓ Temps/step 2M : cible < 500 ms
✓ Temps/step 8M : cible < 3000 ms → 6000 steps ≈ 5h
```

**STOP — rapporter, attendre instruction.**

---

## OPTIMISATION 5 — Incremental tree updates (Grok)

### Principe

dt=0.005 → déplacement typique < 0.1% box par step.
Mettre à jour uniquement les nœuds dont les particules ont bougé.
Fallback full rebuild toutes les 10 steps.

### Implémentation

```cuda
__global__ void flag_moved_particles(
    const double* pos_new, const double* pos_old,
    uint8_t* moved_flags, int n, double threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double dx = pos_new[i*3]   - pos_old[i*3];
    double dy = pos_new[i*3+1] - pos_old[i*3+1];
    double dz = pos_new[i*3+2] - pos_old[i*3+2];
    moved_flags[i] = (dx*dx + dy*dy + dz*dz > threshold*threshold) ? 1 : 0;
}
```

Seuil : 0.5 × softening. Utiliser CUB scan pour compter les flagués.

### Validation

```
✓ S(t) identique ±5% vs full rebuild
✓ Temps/step 2M : cible < 200 ms
```

**STOP — rapporter, attendre instruction.**

---

## OPTIMISATION 6 — Force Freezing (Gemini — post-ségrégation)

### Principe

Une fois S > 0.05, la force cross (répulsive) change lentement.
Recalculer la force cross 1 step sur 2, réutiliser le step précédent
(pondéré par le facteur d'échelle a).

```
Step pair   : Force_self + Force_cross  (calcul complet)
Step impair : Force_self + Force_cross_prev × (a_prev/a_curr)²
```

**ATTENTION : activer uniquement si S > 0.05 (ségrégation établie)**

### Validation

```
✓ S(t) identique ±5% vs calcul complet
✓ KE/KE₀ identique ±2%
✓ Activer/désactiver automatiquement selon S
```

**STOP — rapporter, attendre instruction.**

---

## OPTIMISATION 7 — Async multi-stream (Grok)

### Principe

3 CUDA streams en overlap :
```
Stream 1 : build tree (t+1)     ← pendant que Stream 2 calcule forces
Stream 2 : compute forces (t)   ← pendant que Stream 3 fait kick/drift
Stream 3 : kick/drift (t-1)
```

Gain estimé : 1.2-2×.

### Validation

```
✓ S(t) identique ±2%
✓ Profil Nsight : overlap > 60%
```

**STOP — rapporter, attendre instruction.**

---

## MISE À JOUR ROADMAP après chaque optimisation

Ajouter dans `janus_roadmap.md` :

```markdown
## Optimisations BH GPU

| Opt | Description           | Temps/step 2M | Speedup cumulé | Statut |
|-----|-----------------------|---------------|----------------|--------|
| 0   | KDK baseline          | 7810 ms       | 1.0×           | ✅     |
| 1   | DKD intégrateur       | 3868 ms       | 2.0×           | ✅     |
| 2   | Morton sorting (CPU)  | 2662 ms       | 2.9×           | ✅     |
| 3   | Asymmetric θ          | ? ms          | ?×             | 🔄     |
| 4   | GPU tree build        | ? ms          | ?×             | ⬜     |
| 5   | Incremental updates   | ? ms          | ?×             | ⬜     |
| 6   | Force Freezing        | ? ms          | ?×             | ⬜     |
| 7   | Async streams         | ? ms          | ?×             | ⬜     |

Horizon : 16M particules < 12h overnight
```

---

## FORMAT RAPPORT

```
OPTIMISATION X — [SUCCÈS/ÉCHEC]
Temps/step 2M avant : X ms
Temps/step 2M après : X ms ([speedup]×)
Speedup cumulé      : X× (vs KDK 7810 ms)
8M estimé           : Xh
Validation S(t)     : [PASS/FAIL] écart X%
Validation KE/KE₀   : [PASS/FAIL] écart X%
Roadmap mise à jour : [OUI/NON]
```

---

## OBJECTIF

| Config        | Temps/step | 6000 steps | Particules max 12h |
|---------------|------------|------------|-------------------|
| Actuel opt2   | 2662 ms    | 19.5h      | ~5M               |
| Après opt3+4  | < 500 ms   | < 3.5h     | ~30M              |
| Après opt3→7  | < 150 ms   | < 15 min   | > 50M             |

Horizon final : 16M particules en 12h overnight.
