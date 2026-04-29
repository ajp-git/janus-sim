# Audit 04 — GPU capabilities

**Date** : 2026-04-29

## Hardware

```
NVIDIA GeForce RTX 3060
VRAM: 12288 MiB (12 GB)
Compute capability: 8.6 (Ampere)
Driver: 550.163.01
```

## CUDA toolkit

```
nvcc --version
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
nvcc location: /usr/bin/nvcc
```

## Capacités attendues RTX 3060 (Ampere sm_86)

| Spec | Valeur | Source |
|---|---|---|
| **Compute capability** | 8.6 | nvidia-smi confirmé |
| **VRAM** | 12 GB (≈11 GB utilisable après safety margin) | nvidia-smi |
| **SMs** | 28 | Ampere sm_86 |
| **CUDA cores** | 28 × 128 = 3584 | |
| **Shared memory par SM** | 100 KB max (configurable depuis 48 KB default) | Ampere |
| **Registers par thread** | 255 max | Ampere |
| **L1 cache par SM** | 128 KB | Ampere |
| **TFlops SP** | ~13 (réel ~10-11 en charge) | NVIDIA spec |
| **TFlops DP** | ~0.2 (×65 plus lent que SP) | NVIDIA spec (consumer Ampere) |
| **Memory bandwidth** | ~360 GB/s (192-bit GDDR6 @ 15 Gbps) | NVIDIA spec |

## Implications PhotoNs-GPU

**Critique** : RTX 3060 SP/DP ratio = 65×. Les optims PhotoNs reposent sur kernels P2P en SP avec table T[4] (interpolation Taylor 4 termes) qui satisfait err < 1e-6. C'est compatible avec le plan §3.0 :

| Module | Précision | Compatible RTX 3060 |
|---|---|---|
| Kernel P2P | SP | ✅ optimal |
| Table T(x), E(x) | SP | ✅ |
| Tree COM, multipoles | DP | ✅ (calcul host CPU ou peu de threads GPU DP) |
| PM solver (FFT) | SP | ✅ cuFFT SP largement suffisant |
| Drift/Kick global | DP pos / SP vel | ✅ mixed precision |
| Diagnostics (P(k), σ8) | DP | ✅ host CPU |

## Mémoire prédictive pour TreePM Janus

Pour N=1M particules, N_pm=256³ :

| Buffer | Taille |
|---|---|
| Particles SoA (pos×3 DP, vel×3 SP, acc×3 SP, mass SP, sign i8) = 8B + 4B + 4B + 4B + 0.125B = 20.125B per particle | × 1M = ~20 MB |
| BVH internal nodes (n-1) | 2N-1 nodes × 12 doubles = ~190 MB |
| ρ_+, ρ_- grids 256³ × DP | 2 × 128 MB = 256 MB |
| Φ_+, Φ_- grids 256³ × DP | 2 × 128 MB = 256 MB |
| FFT workspace (cuFFT R2C 256³) | ~256 MB |
| pm_forces buffer | ~12 MB |
| Total ~1 GB pour 1M | ✅ tient largement dans 12 GB |

Pour N=10M, N_pm=512³ :

| Buffer | Taille |
|---|---|
| Particles SoA × 10M | ~200 MB |
| BVH internal | ~1.9 GB |
| ρ_+, ρ_- grids 512³ × DP | 2 × 1024 MB = 2 GB |
| Φ_+, Φ_- grids 512³ × DP | 2 × 1024 MB = 2 GB |
| FFT workspace | ~2 GB |
| Total ~8 GB | ✅ tient mais marge faible |

## Conclusion Phase 1.4

Hardware et toolkit OK pour TreePM JPP :
- ✅ sm_86, CUDA 12.0, 12 GB VRAM
- ✅ Cible 1M particules : ~1 GB VRAM, marge 11×
- ✅ Cible 10M particules : ~8 GB VRAM, marge 1.5× (à surveiller)
- ✅ SP/DP mix de PhotoNs cohérent avec consumer Ampere

Aucun blocage matériel pour le port TreePM Janus.
