# janus-sim

**N-body cosmological simulation of the Janus Cosmological Model
(Petit & D'Agostini) with pure bimetric expansion**

[![Language](https://img.shields.io/badge/language-Rust%20%2B%20CUDA-orange)]()
[![Status](https://img.shields.io/badge/status-research-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

`janus-sim` is a GPU-accelerated cosmological N-body simulator built
from scratch to numerically validate the Janus Cosmological Model
(JCM, [Petit & D'Agostini](https://link.springer.com/article/10.1007/s10509-018-3365-3)).
It implements a TreePM bimetric solver in Rust/CUDA that follows
two species of opposite gravitational mass ($m^+$ and $m^-$)
under pure bimetric expansion from $z=99$ to $z=0$.

The reference run **V-6 PROD** (presented in the accompanying paper)
covers $N = 2.2 \times 10^7$ particles in a $L = 250$ Mpc box on
a single NVIDIA RTX 3060 (12 GB VRAM), in 6.35 wall hours.

### Key results

- **First N-body confirmation** of Janus phenomenology with bimetric
  expansion calibrated on Pantheon+ supernovae ($\eta = 1.045$,
  $\chi^2/\mathrm{dof} = 0.45$)
- **Anti-Newton peak** signature observed in $\simeq 57\%$ of the 1671-halo
  catalog with bootstrap-valid profiles, growing with mass
  ($40\% \to 65\%$ from $M \sim 10^{12}\,M_\odot$ to $> 10^{13.5}\,M_\odot$)
- **Central $m^-$ cavity** observed in $10/10$ archetype halos
  (median $\mu_{\text{shell}_0} = 1.59$, bootstrap $N_b = 1000$)
- **Anti-correlation** $\mathrm{Corr}(\delta^+, \delta^-) = -0.371 \pm 0.001$
  on $16^3$ grid at $z=0$, with marked scale dependence
- **Cosmological compatibility**: $\sigma_{R8}(m^+, z=0) = 0.845$,
  $t_0 = 15.99$ Gyr

See [the paper](docs/janus_paper_c.pdf) for full results.

---

## Architecture

```
janus-sim/
├── src/
│   ├── treepm/          # Bimetric TreePM solver (Tree + PM/FFT)
│   ├── expansion/       # Janus parametric H(a) (Petit 2014 eq.15)
│   ├── ic/              # Zeldovich anti-correlated IC (Phase N)
│   ├── vsl/             # VSL convention c²(z) = (1+z)^δ
│   ├── io/              # Snapshot binary I/O
│   └── diagnostics/     # Energy drift, σ_R, P(k), correlations
├── analysis/            # Python post-processing (FoF, μ_local, bootstrap)
├── config/              # YAML simulation parameters
├── scripts/             # Run launchers, watcher utilities
├── docs/                # Paper preprint, figures, technical notes
└── tests/               # Unit and integration tests
```

---

## Quick start

### Prerequisites

- Linux x86_64 (tested on Ubuntu 22.04 / Linux Mint 21)
- NVIDIA GPU with CUDA capability ≥ 7.0 and ≥ 8 GB VRAM
  (RTX 3060 12 GB recommended for $N = 2.2 \times 10^7$ particles)
- Rust toolchain ≥ 1.75 (`rustup install stable`)
- CUDA toolkit ≥ 12.0
- Python 3.10+ for post-processing (NumPy, SciPy, Matplotlib)

### Build

```bash
git clone https://github.com/ajp-git/janus-sim.git
cd janus-sim
cargo build --release --features=cuda
```

### Run the reference V-6 simulation

```bash
./target/release/janus-sim run --config config/v6_production.yaml
```

This reproduces the canonical run:
$N = 2.2 \times 10^7$, $\mu = 19$, $\eta = 1.045$, $L = 250$ Mpc,
$z = 99 \to 0$, 651 snapshots, $\sim 6$h wall time on RTX 3060.

Snapshots are written to
`output/v6_production_N22M_janus_expansion/snapshots/`
(approximately 319 GB total).

### Reproduce key analyses

```bash
# Anti-correlation Corr(δ+, δ-) on 16³ grid + Monte Carlo convergence
python analysis/m3_grid_analysis.py \
    --snapshot output/v6_production_N22M_janus_expansion/snapshots/snapshot_003247_z-0.0002.bin

# Halo catalog (FoF b=0.5, N_min=32)
python analysis/fof_halos.py \
    --snapshot output/v6_production_N22M_janus_expansion/snapshots/snapshot_003247_z-0.0002.bin \
    --b 0.5 --nmin 32

# Bootstrap profiles μ_local(r/R_FoF) on 10 archetype halos
python analysis/bootstrap_halos.py \
    --catalog output/v6_production_N22M_janus_expansion/diagnostics/catalog_z0_b05.csv \
    --halos 0,1,5,7,28,39,54,210,1041,2257 \
    --n-bootstrap 1000
```

---

## Theoretical framework

The simulator implements the bimetric Janus model as formulated by
[Petit (2014)](https://www.worldscientific.com/doi/abs/10.1142/S0217732314501827)
(eq. 12, 15, 22). In the Newtonian limit relevant for $z \leq 99$,
the coupled Poisson system reduces to:

$$\nabla^2 \Phi^+ = 4\pi G (\rho^+ - \rho^-)$$
$$\nabla^2 \Phi^- = -\nabla^2 \Phi^+ = 4\pi G (\rho^- - \rho^+)$$

producing intra-species attraction and inter-species repulsion.

The expansion follows Petit's parametric solution
$a^+(\mu_p) = \alpha^2 \cosh^2(\mu_p)$ with two implementation
constants:

- $\alpha^2 = 0.1815$ (fixes the Janus transition at $z_J \simeq 4.51$)
- $\tau_0 = 22.71$ Gyr (calibrated for $H_0 = 71.62$ km/s/Mpc)

Below $z_J$, the expansion is matter-dominated; above, the
parametric Janus solution applies. The matching is $C^0$
discontinuous on a single integration step, with no detectable
artifact in the cosmological diagnostics.

A phenomenological VSL convention
$c^2(z)/c_0^2 = (1+z)^{\delta}$ with $\delta = (\eta-1)/\eta$
is applied for numerical stability of the bimetric expansion.

See `docs/janus_paper_c.pdf` for full mathematical details and
discussion of theoretical conventions.

---

## Reproducibility

Every result in the companion paper is reproducible from this repository:

- **Initial conditions**: deterministic, seed = 42, Zeldovich
  approximation with CAMB-like $P(k)$ calibrated on Planck 2020
- **Simulation parameters**: stored in `config/v6_production.yaml`
- **Analysis scripts**: pure Python with seeded RNG for Monte Carlo
- **Figures**: scripts in `docs/figures/` regenerate all paper
  figures from snapshot data

The author is happy to share the full snapshot dataset
($\sim 319$ GB) on request (planned Zenodo archival).

---

## Status and roadmap

**Current** — Reference V-6 run completed and analyzed; paper
preprint in preparation.

**Phase 2** (planned) — Multi-zoom simulations with baryonic
physics (SPH + cooling), $\Lambda$CDM control run for
falsifiability testing, predictive lensing pipeline for cluster
comparisons.

---

## Citation

If you use this code or data in your research, please cite the
companion paper (preprint forthcoming on arXiv):

```bibtex
@misc{Pares2026Janus,
  author       = {Parès, Alain Jean},
  title        = {Première validation N-corps complète du modèle
                  cosmologique Janus avec expansion bimétrique :
                  ségrégation, cavité m⁻ universelle dans les halos
                  massifs et compatibilité observationnelle à 22
                  millions de particules},
  year         = {2026},
  howpublished = {GitHub: \url{https://github.com/ajp-git/janus-sim}},
  note         = {Preprint in preparation}
}
```

---

## Author

**Alain Jean Parès**
Independent researcher, Chaumes-en-Brie, France
📧 [ajp@p997.net](mailto:ajp@p997.net)

---

## Acknowledgments

This work would not have been possible without the foundational work
of Jean-Pierre Petit and Gilles D'Agostini on the Janus Cosmological
Model. The Rust and CUDA open-source ecosystems provided the
technical foundation for the simulator. Pantheon+ supernova data
(Brout et al. 2022) was used for cosmological calibration.

---

## License

MIT License — see [LICENSE](LICENSE) for details. The code is freely
available for academic and personal use; please cite the companion
paper when using results in publications.

---

## Disclaimer

This repository implements the Janus Cosmological Model as a
phenomenological framework for numerical investigation. The
theoretical foundations of JCM are the subject of ongoing debate in
the literature. This work makes no claim regarding the mathematical
consistency of the underlying bimetric formulation and focuses
exclusively on the numerical phenomenology of the model as
implemented in this code, with all conventions transparently
documented.
