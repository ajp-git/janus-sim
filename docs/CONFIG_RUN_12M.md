# Configuration Complète — Simulation Janus 12M Particules

## Contexte : Modèle Cosmologique Janus

Le modèle Janus est une cosmologie bimétrique proposée par **Jean-Pierre Petit & Gilles D'Agostini** où l'univers contient deux populations de masse :
- **Masses positives (+)** : matière ordinaire
- **Masses négatives (−)** : "matière sombre" à masse négative

### Règles d'Interaction Gravitationnelle Janus
```
+/+ → Attraction (Newton classique)
−/− → Attraction (Newton classique)
+/− → Répulsion mutuelle (anti-Newton)
```

Cette règle élimine le "runaway paradox" et produit naturellement une accélération cosmique sans constante cosmologique Λ.

---

## Objectif Actuel

**Voir des FILAMENTS cosmiques se former** par ségrégation gravitationnelle entre masses + et −.

Actuellement, nous observons une **instabilité dipolaire** (mode k=1) où les + et − se séparent en deux hémisphères. Nous voulons voir si des structures plus complexes (filaments, nœuds) peuvent émerger.

---

## Configuration du Run Actuel (12M)

### Paramètres Physiques

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **N_total** | 12,008,989 | Nombre total de particules (229³) |
| **N_+** | ~5.87M | Masses positives |
| **N_−** | ~6.14M | Masses négatives |
| **η = N_−/N_+** | 1.045 | Ratio de densité (paramètre unique du modèle) |
| **L_BOX** | 492 Mpc | Taille de la boîte (conditions périodiques) |
| **z_init** | 5.0 | Redshift initial |
| **z_final** | 0.0 | Redshift final (aujourd'hui) |

### Paramètres Numériques

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **N_GRID** | 229 | Grille pour ICs Zel'dovich (229³) |
| **DT** | 0.01 | Pas de temps (unités Hubble) |
| **TOTAL_STEPS** | 20,000 | Nombre total d'itérations |
| **θ (Barnes-Hut)** | 0.7 | Paramètre d'ouverture de l'arbre |
| **ε (softening)** | 0.65 Mpc | Adoucissement gravitationnel |
| **Intégrateur** | DKD Leapfrog | Drift-Kick-Drift symplectique |

### Troncature du Spectre P(k)

```
P(k) ∝ k^0.96 / (1 + (k/k0)^4)

Avec fenêtre de troncature :
  k_min = 2π/200 Mpc⁻¹  →  λ_max = 200 Mpc (supprime grandes échelles)
  k_max = 2π/8 Mpc⁻¹    →  λ_min = 8 Mpc   (supprime petites échelles)
  k0 = 0.02 Mpc⁻¹       →  Échelle de coupure du spectre
```

**But de la troncature** : Supprimer le mode k=1 (λ = 492 Mpc = L_box) pour éviter l'instabilité dipolaire globale et favoriser les structures multi-échelles.

---

## Conditions Initiales : Zel'dovich avec Attribution par Densité

### Génération des ICs

1. **Grille régulière** : 229³ = 12M positions sur grille cartésienne
2. **Spectre P(k) tronqué** : Génération des modes de Fourier δ_k avec :
   ```
   δ_k = Normal(0, σ_k) × exp(iφ)
   σ_k = sqrt(P(k)) × amplitude × D(z_init)
   amplitude = 0.01
   D(z=5) = a_init = 1/6 ≈ 0.167
   ```

3. **Champ de déplacement Zel'dovich** :
   ```
   ψ_k = -i k δ_k / k²
   ψ(x) = IFFT(ψ_k)

   Position finale : x = x_grille + ψ(x) × scale
   Vitesse finale  : v = ψ(x) × vel_scale

   scale = 0.3 × spacing / max(|ψ|)
   vel_scale = √(1+z_init) × scale
   ```

4. **Attribution des signes basée sur la densité** :
   - Calculer le champ de densité δ(x) = IFFT(δ_k)
   - Trier les particules par δ décroissant
   - Les N_+ = N/(1+η) particules dans les régions les plus denses → signe +
   - Les N_- = N×η/(1+η) particules restantes → signe −

   **Conséquence** : Les + sont initialement dans les surdensités, les − dans les sous-densités.

5. **Mélange aléatoire** : Les indices sont mélangés (shuffle) pour éviter les biais de layout mémoire.

### Vérification Anti-Corrélation

```
Corrélation(index, z_position) ≈ 0  ✓
```
Confirme que le mélange a éliminé les corrélations spatiales artificielles.

---

## Virialization Janus

### Problème avec la virialization standard

La virialization classique utilise :
```
2 KE + PE_total = 0  →  KE_cible = -PE_total/2
```

Mais dans Janus avec η ≈ 1 :
- Les paires +/− répulsives dominent → PE_total > 0
- KE_cible = -PE_total/2 < 0  →  **Impossible !**

### Solution : PE_binding (paires attractives seulement)

```rust
// Calculer PE uniquement pour les paires qui s'attirent
PE_binding = PE(+/+) + PE(−/−)   // Toujours < 0

// Virialization avec PE_binding
KE_cible = -PE_binding / 2

// Facteur d'échelle des vitesses
α = sqrt(KE_cible / KE_actuelle)
v_new = v × α
```

**α typique** : 4-5 pour η = 1.045

---

## Expansion Cosmique Couplée

### Équations de Friedmann Janus

Deux facteurs d'échelle couplés a(t) et ā(t) :

```
(ȧ/a)² = (8πG/3) × ρ₀/a³        // Secteur +
(ā̇/ā)² = (8πG/3) × |ρ̄₀|/ā³      // Secteur −

Conservation : ρc²a³ + ρ̄c̄²ā³ = E = const
```

Avec η = 1.045 :
- Ω_+ = 1/(1+η) = 0.489
- Ω_- = η/(1+η) = 0.511
- E = Ω_+ - Ω_- = -0.022 < 0 → Accélération cosmique

### Paramètre de décélération

```
q₀ = -0.022  (proche de zéro, univers quasi-plat)
```

### Implémentation dans le N-body

```rust
// À chaque pas de temps :
tau += dtau_per_step;
let (a, H) = cosmo.get_params_at_tau(tau);

// Friction de Hubble sur les vitesses
v_new = v × exp(-H × dt × dtau_per_dt)

// Pas DKD avec friction
drift(dt/2);
kick(dt, a, H);  // Forces + friction Hubble
drift(dt/2);
```

---

## État Actuel du Run

### Progression

| Métrique | Valeur |
|----------|--------|
| Steps complétés | 2030 / 20000 (10%) |
| z actuel | 3.37 |
| KE/KE₀ | 1.68 |
| Ségrégation | 0.318 |
| Temps/step | ~11.4 s |
| ETA restant | ~57h |

### Observations à step 2030

- **KE augmente** : KE/KE₀ = 1.68 (normal avec virialization Janus)
- **Ségrégation croît** : Seg = 0.318 (vs 0.007 initial)
- **Mode dominant** : k ≈ k_box malgré la troncature

---

## Problème Identifié : Instabilité Dipolaire

### Analyse Fourier du champ de polarisation

```
P(x) = (ρ+ - ρ-) / (ρ+ + ρ-)

Spectre P(k) montre :
  k_peak ≈ k_box = 2π/492 = 0.0128 Mpc⁻¹

Le mode fondamental k=1 domine malgré la troncature !
```

### Test L = 2L (984 Mpc)

Nous avons testé avec une boîte 2× plus grande :

| Métrique | L=492 Mpc | L=984 Mpc |
|----------|-----------|-----------|
| k_box | 0.0128 | 0.0064 |
| k_peak observé | 0.0114 | 0.0057 |
| λ_peak | 550 Mpc | 1100 Mpc |

**Conclusion** : k_peak ≈ k_box dans les DEUX cas. L'instabilité s'adapte à la taille de la boîte.

### Interprétation

Le mode k=1 domine car :
1. **Taux de croissance** : γ(k) ∝ 1/k (modes longs croissent plus vite)
2. **Conditions périodiques** : Sélectionnent le mode fondamental
3. **Pas d'échelle intrinsèque** : Janus n'a pas de longueur caractéristique naturelle

---

## Ce qu'il faudrait pour voir des filaments

### Options possibles

1. **Supprimer explicitement les modes k ≤ 2** :
   ```
   k_min = 3 × 2π/L_box
   ```
   Forcer les structures à échelle 1/3 de la boîte ou moins.

2. **Ajouter une physique à courte portée** :
   - Longueur d'écrantage (type Yukawa) : F ∝ exp(-r/λ)/r²
   - λ_écran ~ 10-50 Mpc pour stabiliser les grandes échelles

3. **Boîte beaucoup plus grande** :
   - L = 2000+ Mpc avec N > 100M particules
   - Pour que λ_physique << L_box

4. **ICs avec structures pré-existantes** :
   - Injecter des perturbations à échelles multiples
   - Spectre P(k) avec plusieurs pics

---

## Format des Snapshots

### Structure binaire

```
Header (24 bytes):
  u64 n_particles
  u64 step_number
  u64 reserved (0)

Body (n × 16 bytes):
  f32 x, y, z, sign   // pour chaque particule
  sign = +1.0 ou -1.0
```

### Lecture Python

```python
import struct
import numpy as np

def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)
    sign = data[:, 3]
    pos_plus = data[sign > 0, :3]
    pos_minus = data[sign < 0, :3]
    return n, step, pos_plus, pos_minus
```

---

## Hardware

| Composant | Spécification |
|-----------|---------------|
| GPU | NVIDIA RTX 3060 12GB |
| CPU | x86_64 Linux |
| RAM | 32 GB |
| Stockage | NVMe sur /mnt/T2 |

---

## Références

1. **Petit, Margnat & Zejli (2024)** — EPJC 84:1226 — Modèle actuel
2. **D'Agostini & Petit (2018)** — Astrophys. Space Sci. 363:139 — Formule μ(z)
3. **Petit & D'Agostini (2014)** — Astrophys. Space Sci. 354:611 — Équations Friedmann
4. **Scolnic et al. (2022)** — ApJ 938:113 — Données Pantheon+

---

## Commandes

```bash
# Compiler
docker compose run --rm dev cargo build --release --features cuda --bin production_pktrunc_12m

# Lancer
docker compose run --rm dev cargo run --release --features cuda --bin production_pktrunc_12m

# Vérifier progression
tail -20 output/production_pktrunc_12m_v2/time_series.csv

# Rendre un snapshot
python3 scripts/render_frame_V5.py output/.../snapshots/snap_001000.bin /tmp/frame.png
```
