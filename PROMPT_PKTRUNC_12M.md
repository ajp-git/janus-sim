# MISSION : Run 12M production — ICs Zel'dovich P(k) tronqué
# Objectif : filaments/blobs multiples présentables, pas de frontière planaire
# Lis ce fichier en entier avant toute action.

---

## CONTEXTE

Le run 12M précédent montrait une frontière planaire (ségrégation sur un seul axe Z).
Cause : les ICs uniformes aléatoires + ordre février créent une brisure de symétrie
sur un seul mode → un seul plan de ségrégation.

Solution : ICs Zel'dovich avec P(k) **tronqué aux grandes échelles**.
En supprimant les modes k < k_min (λ > 150 Mpc), on force plusieurs structures
à émerger à l'échelle 30-100 Mpc → résultat visuellement riche et présentable.

Bugs critiques immuables :
- FIX-015 : box = n_side × 2.15 Mpc
- FIX-016 : dtau_per_dt = tau_range / (TOTAL_STEPS × DT)

---

## ÉTAPE 1 — Stopper le run actuel

```bash
docker stop $(docker ps -q --filter "name=janus-sim")
```

Vérifier que le GPU est libéré :
```bash
nvidia-smi
```

---

## ÉTAPE 2 — Implémenter P(k) tronqué dans les ICs Zel'dovich

### Principe

Le spectre de puissance standard ΛCDM P(k) ∝ k^n_s × T²(k) donne trop de
puissance aux grandes échelles (petits k). Pour une boîte de 492 Mpc, le mode
fondamental λ=492 Mpc domine et crée une seule structure → frontière planaire.

En tronquant P(k) à k < k_min, on supprime ces modes dominants et on laisse
les modes intermédiaires (λ=30-100 Mpc) créer plusieurs structures.

### Implémentation

```rust
// Dans la génération du champ δ(k) via FFT :

let k_min = 2.0 * PI / 150.0;  // supprime λ > 150 Mpc
let k_max = 2.0 * PI / 15.0;   // supprime λ < 15 Mpc (bruit)

for kx in ... {
    for ky in ... {
        for kz in ... {
            let k = (kx*kx + ky*ky + kz*kz).sqrt();
            
            // Fenêtre spectrale : anneau dans l'espace k
            let window = if k < k_min || k > k_max {
                0.0  // mode supprimé
            } else {
                1.0  // mode conservé
            };
            
            // Amplitude P(k) standard × fenêtre
            let pk = power_spectrum(k) * window;
            delta_k[idx] = pk.sqrt() * random_complex_gaussian();
        }
    }
}
```

### Paramètres de troncature recommandés

```
k_min = 2π / 150 Mpc  → supprime les modes λ > 150 Mpc
k_max = 2π / 15 Mpc   → supprime les modes λ < 15 Mpc

Résultat attendu : 3-8 structures indépendantes dans la boîte 492 Mpc
Échelle typique des structures : 30-100 Mpc
```

### Attribution des signes (density-based + ordre physique)

```rust
// IMPORTANT : ne PAS utiliser l'ordre "positifs d'abord" (biais Z)
// Utiliser l'attribution basée sur le signe de δ(x) local :
//   δ(x) > 0 → particule POSITIVE (surdensité)
//   δ(x) < 0 → particule NÉGATIVE (sous-densité)
// Puis mélanger les indices aléatoirement pour éviter tout biais spatial
```

---

## ÉTAPE 3 — Test de validation 500K (~20 min)

Avant le run 12M, valider que les ICs P(k) tronqué donnent plusieurs structures.

```
N         = 500_000
Box       = 172 Mpc
k_min     = 2π / 60 Mpc   (adapté à la boîte 172 Mpc)
k_max     = 2π / 6 Mpc
Softening = 0.65 Mpc
Steps     = 2000
dtau_per_dt = tau_range / (2000 × 0.01)
Snapshots = NON
Output    : output/pktrunc_500k_test/
```

### Image à générer après le test 500K

Générer une image de densité au step 0 ET au step 2000 :
- Vérifier visuellement que step 0 montre plusieurs structures (pas un seul blob)
- Vérifier que step 2000 montre une ségrégation multi-structures

Uploader les deux images avant de lancer le 12M.

Critères PASS :
```
Step 0    : ≥ 3 structures distinctes visibles dans la boîte
Step 2000 : Seg_max > 0.05  ET  KE/KE₀ < 20
Morphologie : plusieurs blobs/filaments, PAS une frontière planaire
```

---

## ÉTAPE 4 — Run production 12M + snapshots

Si étape 3 PASS, lancer immédiatement sans attendre retour humain.

```
N               = 12_000_000
Box             = 492 Mpc
k_min           = 2π / 150 Mpc  (supprime λ > 150 Mpc)
k_max           = 2π / 15 Mpc   (supprime λ < 15 Mpc)
Softening       = 0.65 Mpc
θ               = 0.7
dt              = 0.01
z_init          = 5.0
Steps           = 20000
ICs             = Zel'dovich density-based, signes mélangés aléatoirement
Virialization   = virialize_sampled(n=80000)
dtau_per_dt     = tau_range / (20000 × 0.01)  ← FIX-016 IMPÉRATIF
SNAPSHOT_INTERVAL = 20   → 1000 snapshots → ~192 GB
Output          = /mnt/T2/janus-sim/output/production_pktrunc_12m/
```

### Format snapshot binaire (identique au run précédent)
```
snap_XXXXXX.bin
Header  : [n_particles: u64, step: u64, reserved: u64]
Data    : n_particles × [x: f32, y: f32, z: f32, sign: f32]
Taille  : ~192 MB/snapshot
```

---

## CHECKLIST AVANT LANCEMENT 12M

```
□ nvidia-smi → GPU propre (0 MB utilisé)
□ df -h /mnt/T2/ → > 200 GB disponibles
□ Image step 0 vérifiée → ≥ 3 structures visibles
□ k_min et k_max corrects dans le code
□ Signes density-based MÉLANGÉS (pas ordonnés par Z)
□ dtau_per_dt = tau_range / (20000 × 0.01)  ← vérifier dans le code
□ virialize_sampled(n=80000)
□ SNAPSHOT_INTERVAL = 20
□ KE/KE₀ step 5 < 1.05
□ git commit -m "feat: P(k) tronqué k_min=2pi/150 k_max=2pi/15"
□ Container ID sauvegardé dans RUNS.md
```

---

## MILESTONES À SURVEILLER

```
Step 5     : KE/KE₀ < 1.05  (stable)
Step 100   : KE/KE₀ < 0.95  (friction Hubble active)
Step ~3000 : onset ségrégation (Seg > 0.05)
Step ~4500 : pic ségrégation (Seg > 0.2)
Step 20000 : run complet z=0
```

---

## DOCUMENTATION

Après lancement, ajouter dans RUNS.md :
```
### Run: production_pktrunc_12m
Date: 2026-03-05
N=12M, Box=492 Mpc, Steps=20000
ICs: Zel'dovich density-based, k_min=2π/150, k_max=2π/15
Signes: density-based mélangés (pas de biais Z)
dtau_per_dt: tau_range / (20000 × 0.01)
Snapshots: toutes les 20 steps → 1000 fichiers
Container: [ID]
ETA: ~35h
```

---

## RÈGLES ABSOLUES

```
JAMAIS  : docker stop $(docker ps -q)  sans filtre --filter name=
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
TOUJOURS : vérifier image step 0 avant run 12M
JAMAIS  : lancer 12M sans validation 500K préalable
```

---

## NOTE IMAGES

Les snapshots .bin seront traités par script Python externe.
CLI ne génère pas d'images — sauvegarder uniquement les .bin et time_series.csv.
Exception : images de validation à l'étape 3 (step 0 et step 2000 du 500K).
