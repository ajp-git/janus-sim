# Janus Cosmological Model — N-body Simulation

Validation numérique du modèle cosmologique bimétrique Janus de Jean-Pierre Petit.

## Description

Ce projet implémente une simulation N-corps GPU pour valider les prédictions du modèle Janus, qui remplace la matière noire et l'énergie noire par des masses négatives interagissant selon des règles d'interaction spécifiques.

### Règles d'interaction Janus (limite Newtonienne)
- **Masse+ attire Masse+** : gravité Newtonienne classique
- **Masse- attire Masse-** : symétrie (attraction)
- **Masse+ et Masse- se repoussent** : anti-gravité (élimine le runaway)

### Paramètre libre unique
- **η = |ρ̄₀|/ρ₀** : ratio des densités négative/positive
- H₀ = 70 km/s/Mpc (cohérent avec Janus)

## Résultats Phase 1a — Fit Pantheon+

Ajustement sur 1701 supernovae Ia du catalogue Pantheon+ (Scolnic et al. 2022) :

| Paramètre | Valeur |
|-----------|--------|
| η optimal | **1.045** |
| χ²/dof | **0.914** |
| q₀ (décélération) | -0.022 |

Le modèle Janus reproduit les observations SNIa avec **1 seul paramètre libre** (vs 2 pour ΛCDM : Ωm, ΩΛ).

## Références

1. **Petit, J.-P., Margnat, S. & Zejli, H.** (2024). *The Janus Cosmological Model*. Eur. Phys. J. C 84, 1226. [DOI:10.1140/epjc/s10052-024-13589-8](https://doi.org/10.1140/epjc/s10052-024-13589-8)

2. **D'Agostini, G. & Petit, J.-P.** (2018). *Constraints on Janus Cosmological model from recent observations of supernovae type Ia*. Astrophys. Space Sci. 363, 139.

3. **Petit, J.-P. & D'Agostini, G.** (2014). *Negative mass hypothesis in cosmology and the nature of dark energy*. Astrophys. Space Sci. 354, 611.

4. **Zejli, H.** (2024). *The Janus Cosmological Model — Technical Book*. 233 pages. [januscosmologicalmodel.com](https://januscosmologicalmodel.com)

## Prérequis Hardware

- **GPU** : NVIDIA avec CUDA (RTX 3060 12GB ou supérieur recommandé)
- **RAM** : 32 GB minimum
- **Stockage** : SSD recommandé pour les snapshots

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/[user]/janus-sim.git
cd janus-sim

# Vérifier CUDA
nvidia-smi

# Builder avec Docker
docker compose build
```

## Utilisation

### Phase 1a — Fit Friedmann + Pantheon+

```bash
docker compose run --rm friedmann
```

### Phase 1b/c — Simulation N-corps GPU

```bash
# Test rapide (100K particules, 100 steps)
docker compose run --rm dev cargo run --release --features cuda --bin nbody_overnight -- \
  --n 100000 --eta 1.045 --dt 0.01 --steps 100 --output /app/output/test

# Production (500K particules, 10000 steps)
docker compose run --rm dev cargo run --release --features cuda --bin nbody_overnight -- \
  --n 500000 --eta 1.045 --dt 0.01 --steps 10000 --output /app/output/run
```

### Paramètres

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--n` | Nombre total de particules | 100000 |
| `--eta` | Ratio ρ̄/ρ | 1.045 |
| `--dt` | Pas de temps | 0.01 |
| `--steps` | Nombre de steps | 1000 |
| `--output` | Répertoire de sortie | output/ |

## Structure du projet

```
janus-sim/
├── src/
│   ├── lib.rs           # Constantes et règles Janus
│   ├── friedmann.rs     # Intégration FLRW couplées
│   ├── nbody.rs         # N-corps CPU (Barnes-Hut)
│   ├── nbody_gpu.rs     # N-corps GPU (CUDA)
│   └── analysis.rs      # Fitting χ²
├── scripts/             # Visualisation Python
├── data/                # Données Pantheon+ (non incluses)
└── output/              # Résultats (non inclus)
```

## Validations

Le code suit les règles de validation strictes définies dans `VALIDATION_RULES.md` :

- Test trivial obligatoire pour chaque fonction physique
- Conservation d'énergie (KE/KE₀ < 50)
- Ségrégation croissante attendue
- Conditions initiales virialisées (2KE + PE_bind = 0)

## License

MIT

## Contact

- Jean-Pierre Petit : jean-pierre.petit@manaty.net
- Hicham Zejli : hicham.zejli@manaty.net
