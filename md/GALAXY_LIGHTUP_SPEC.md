# CLUSTER LIGHTUP — Spécification technique
## Visualisation 2.5D isométrique 4K — Formation d'un amas Janus
### Version 2.0 — Avril 2026

---

## CONTEXTE

Les objets identifiés dans le run baryonique 10M sont des **amas de galaxies**
(M ~ 10¹⁵ M☉, R_half ~ 3 Mpc), pas des galaxies individuelles.
m_particule = 5×10¹¹ M☉ = masse Voie Lactée.

Chaque "allumage" = une population stellaire entière (~10¹¹ M☉) qui s'effondre.

**L'histoire visuelle en 4 actes :**
```
z=4.0  → nuage homogène rouge/bleu — univers primordial
z=2.3  → début séparation, m- commence à reculer
z=1.55 → PREMIERS FLASHS blancs — premières populations stellaires Janus
z=1.0  → cœur blanc dense, coquille bleue m- visible
z=0.44 → amas brillant, vide m- établi, courbes rotation plates
```

C'est la prédiction visuelle centrale de Petit :
m− expulsé des zones denses → vide bleu creusé en temps réel.

---

## CENTRE DE L'AMAS

Cluster #1 depuis cluster_catalog.csv :
```
center = (-5.329, 11.171, -39.571) Mpc
R_half = 3.63 Mpc
M*     = 2.24×10¹⁵ M☉
```

Rayon d'extraction : **25 Mpc**
(inclut l'amas + le vide m- autour — essentiel pour l'histoire visuelle)

---

## FORMAT BINAIRE GBIN v2

### Structure fichier `cluster_ZZZZ_SSSSS.gbin`

```
HEADER (40 bytes) :
  [0..4]   u32 magic = 0x4742494E  ("GBIN")
  [4..8]   f32 z
  [8..12]  f32 t_gyr
  [12..16] u32 step
  [16..20] u32 n_plus        ← TOUTES les m+ dans le rayon 25 Mpc
  [20..24] u32 n_minus       ← TOUTES les m- dans le rayon 25 Mpc
  [24..28] u32 n_protostars  ← proto-étoiles ce step
  [28..32] u32 n_new_stars   ← nouvelles proto-étoiles ce step
  [32..44] f32 cx, cy, cz    ← centre amas (Mpc)
  [44..48] f32 radius        ← rayon extraction (Mpc)

PARTICULES m+ (20 bytes × n_plus) :
  f32 x, y, z         (12 bytes) — coordonnées Mpc
  f32 overdensity     (4 bytes)  — densité locale / densité moyenne
  u8  is_protostar    (1 byte)   — 1 si proto-étoile ce step
  u8  is_new_star     (1 byte)   — 1 si PREMIÈRE apparition
  u8[2] padding       (2 bytes)

PARTICULES m- (12 bytes × n_minus) :
  f32 x, y, z         (12 bytes) — coordonnées Mpc
  → TOUTES (pas d'échantillonnage) — vide visible
```

### Taille estimée
```
Rayon 25 Mpc dans boîte 500 Mpc → fraction (25/500)³ = 1.25e-4
N_total = 10M → N_dans_rayon ~ 1 250 000 particules

  m+ : 1 250 000 × 20 bytes = 25 MB/fichier
  m- : 1 250 000 × 12 bytes = 15 MB/fichier
  Total : ~40 MB/fichier × 800 snapshots = ~32 GB

→ Vérifier df -h /mnt/T2 avant lancement
  Si espace < 50 GB → radius = 15 Mpc (~7 GB total)
```

---

## OUTIL 1 — src/bin/cluster_extractor.rs

### Arguments CLI
```bash
./cluster_extractor \
  --snap-dir    output/janus_baryonic_calibrated/snapshots/ \
  --cluster-csv output/janus_baryonic_calibrated/cluster_analysis/cluster_catalog.csv \
  --cluster-id  0 \
  --out-dir     output/cluster_lightup/gbin/ \
  --radius      25.0 \
  --sf-threshold 5.0 \
  --n-neighbors  32 \
  --box-size    500.0
```

### Algorithme Rust

```rust
use rayon::prelude::*;
use kiddo::KdTree;

fn main() {
    // 1. Lire cluster_catalog.csv → center du cluster #1
    let center = read_cluster_center(cluster_csv, cluster_id);
    println!("Cluster center: ({:.1}, {:.1}, {:.1}) Mpc",
             center.x, center.y, center.z);

    // 2. Lister tous les snapshots
    let snaps = list_snapshots_sorted(snap_dir);
    println!("Processing {} snapshots...", snaps.len());

    // 3. Traiter en parallèle (CPU)
    snaps.par_iter().for_each(|snap_path| {
        process_snapshot(snap_path, &center, &config);
    });

    // 4. Post-processing séquentiel : marquer is_new_star
    //    Comparer snapshot N avec snapshot N-1
    //    is_new_star = protostar maintenant ET pas au step précédent
    mark_first_appearances_sequential(&out_dir);

    println!("Done. {} GBIN files written.", snaps.len());
}

fn process_snapshot(snap_path: &str, center: &Vec3, cfg: &Config) {
    let (particles, z, step) = read_jsnp(snap_path);

    // Filtrer dans la sphère (conditions périodiques)
    let mut plus_sphere:  Vec<Particle> = Vec::new();
    let mut minus_sphere: Vec<Particle> = Vec::new();

    for p in &particles {
        let dx = periodic_dist(p.x - center.x, cfg.box_size);
        let dy = periodic_dist(p.y - center.y, cfg.box_size);
        let dz = periodic_dist(p.z - center.z, cfg.box_size);

        if dx*dx + dy*dy + dz*dz < cfg.radius * cfg.radius {
            if p.sign > 0 { plus_sphere.push(p.clone()); }
            else          { minus_sphere.push(p.clone()); }
        }
    }

    // Densité moyenne globale m+
    let n_plus_global = particles.iter().filter(|p| p.sign > 0).count();
    let rho_mean = n_plus_global as f64 / cfg.box_size.powi(3);

    // KD-tree sur m+ de la sphère
    let kdtree = build_kdtree(&plus_sphere);

    // Calculer overdensity + div_v pour chaque m+
    let enhanced: Vec<ParticlePlus> = plus_sphere
        .par_iter()
        .map(|p| {
            let neighbors   = kdtree.nearest_n(&p.pos_arr(), cfg.n_neighbors);
            let rho_local   = sph_kernel_density(&neighbors, p.pos());
            let overdensity = (rho_local / rho_mean) as f32;
            let div_v       = velocity_divergence(&neighbors, p.pos());

            ParticlePlus {
                x: p.x as f32,
                y: p.y as f32,
                z: p.z as f32,
                overdensity,
                is_protostar: (overdensity > cfg.sf_threshold as f32
                               && div_v < 0.0) as u8,
                is_new_star: 0,  // rempli en post
                _pad: [0; 2],
            }
        })
        .collect();

    let out_name = format!("cluster_z{:.3}_s{:05}.gbin", z, step);
    write_gbin(&cfg.out_dir.join(out_name),
               z, step, &enhanced, &minus_sphere,
               center, cfg.radius);
}
```

### Performance
```
800 snapshots × ~1.25M particules/snapshot
→ Extraction : 4-6 heures (CPU rayon, 16 threads)
→ GPU libre pour la simulation en parallèle
→ Sortie : ~32 GB de fichiers GBIN
```

---

## OUTIL 2 — analysis/cluster_lightup.py

### Lecture GBIN ultra-rapide (numpy)

```python
import numpy as np
import struct
from pathlib import Path

def read_gbin(path: Path) -> dict:
    with open(path, 'rb') as f:
        # Header 48 bytes
        magic, z, t, step = struct.unpack('<IfII', f.read(16))
        n_plus, n_minus, n_proto, n_new = struct.unpack('<IIII', f.read(16))
        cx, cy, cz, radius = struct.unpack('<ffff', f.read(16))

        # m+ — dtype structuré numpy
        dtype_plus = np.dtype([
            ('x',  '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('od', '<f4'),
            ('is_star', 'u1'), ('is_new', 'u1'),
            ('pad', 'u1', 2)
        ])
        plus = np.frombuffer(f.read(n_plus * 20), dtype=dtype_plus).copy()

        # m- — xyz seulement
        dtype_minus = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4')
        ])
        minus = np.frombuffer(f.read(n_minus * 12), dtype=dtype_minus).copy()

    return dict(z=z, t=t, step=step,
                plus=plus, minus=minus,
                n_proto=n_proto, n_new=n_new)

# Performance : lecture 800 fichiers ~32 GB → < 5 minutes
```

### Rendu 4K isométrique

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, os

# ─── Paramètres vidéo ────────────────────────────────────────
WIDTH, HEIGHT = 3840, 2160
DPI           = 200
FPS           = 24

# ─── Centre et rayon ─────────────────────────────────────────
CENTER = np.array([-5.329, 11.171, -39.571])  # Mpc
RADIUS = 25.0                                   # Mpc

# ─── Angle isométrique FIXE ──────────────────────────────────
ELEV, AZIM = 30, 45

# ─── Palette ─────────────────────────────────────────────────
COLOR_MINUS = '#0a3d6b'   # bleu profond — m-
COLOR_GAS   = '#3d0a00'   # rouge très sombre — m+ gaz
COLOR_STAR  = 'white'     # proto-étoile établie
COLOR_NEW   = '#ffff44'   # flash nouvelle étoile
COLOR_HALO  = '#ff8800'   # halo orange du flash


def render_frame(data: dict, frame_idx: int, out_dir: Path):
    plus  = data['plus']
    minus = data['minus']
    z, t  = data['z'], data['t']

    fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_pane_color((0, 0, 0, 0))
    ax.view_init(elev=ELEV, azim=AZIM)

    # m- : bleu profond transparent — TOUTES les particules
    if len(minus) > 0:
        ax.scatter(minus['x'], minus['y'], minus['z'],
                   c=COLOR_MINUS, alpha=0.10, s=0.3,
                   linewidths=0, rasterized=True)

    # m+ gaz : rouge très sombre
    mask_gas = plus['is_star'] == 0
    if mask_gas.sum() > 0:
        ax.scatter(plus['x'][mask_gas],
                   plus['y'][mask_gas],
                   plus['z'][mask_gas],
                   c=COLOR_GAS, alpha=0.12, s=0.3,
                   linewidths=0, rasterized=True)

    # Proto-étoiles établies : blanc, taille ∝ overdensité
    mask_star = (plus['is_star'] == 1) & (plus['is_new'] == 0)
    if mask_star.sum() > 0:
        sizes = np.clip(plus['od'][mask_star] * 0.5, 2, 20)
        ax.scatter(plus['x'][mask_star],
                   plus['y'][mask_star],
                   plus['z'][mask_star],
                   c=COLOR_STAR, alpha=0.80, s=sizes,
                   linewidths=0)

    # NOUVELLES proto-étoiles : flash jaune + halo orange
    mask_new = plus['is_new'] == 1
    if mask_new.sum() > 0:
        # Halo d'abord (derrière)
        ax.scatter(plus['x'][mask_new],
                   plus['y'][mask_new],
                   plus['z'][mask_new],
                   c=COLOR_HALO, alpha=0.35, s=200,
                   linewidths=0)
        # Flash central
        ax.scatter(plus['x'][mask_new],
                   plus['y'][mask_new],
                   plus['z'][mask_new],
                   c=COLOR_NEW, alpha=1.0, s=60,
                   edgecolors='white', linewidths=0.5,
                   zorder=10)

    # Titre
    n_s      = data['n_proto']
    n_n      = data['n_new']
    m_stellar = n_s * 5.1e11
    ax.set_title(
        f"Janus Cluster  |  z = {z:.3f}  |  t = {t:.2f} Gyr\n"
        f"M* = {m_stellar:.2e} M☉  |  +{n_n:,} new stellar populations",
        color='white', fontsize=15, pad=12, fontfamily='monospace'
    )

    # Axes discrets
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('#444444')
        axis.set_tick_params(colors='#333333')
    ax.set_xlabel('X [Mpc]', color='#555555', fontsize=9)
    ax.set_ylabel('Y [Mpc]', color='#555555', fontsize=9)
    ax.set_zlabel('Z [Mpc]', color='#555555', fontsize=9)

    # Limites FIXES — même bbox tout le film
    ax.set_xlim(CENTER[0] - RADIUS, CENTER[0] + RADIUS)
    ax.set_ylim(CENTER[1] - RADIUS, CENTER[1] + RADIUS)
    ax.set_zlim(CENTER[2] - RADIUS, CENTER[2] + RADIUS)

    out_path = out_dir / f'frame_{frame_idx:05d}.png'
    plt.savefig(out_path, dpi=DPI, facecolor='black',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main(gbin_dir: str, out_dir: str):
    gbin_dir   = Path(gbin_dir)
    out_dir    = Path(out_dir)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(gbin_dir.glob('cluster_*.gbin'))
    print(f"{len(files)} GBIN files to render")

    for i, f in enumerate(files):
        data = read_gbin(f)
        render_frame(data, i, frames_dir)
        if i % 20 == 0:
            print(f"  [{i/len(files)*100:5.1f}%] "
                  f"z={data['z']:.3f} | N*={data['n_proto']:,}")

    # Assembler vidéo 4K H.264
    os.system(
        f'ffmpeg -y -framerate {FPS} '
        f'-pattern_type glob -i "{frames_dir}/frame_*.png" '
        f'-c:v libx264 -crf 15 -preset slow '
        f'-pix_fmt yuv420p -movflags +faststart '
        f'"{out_dir}/janus_cluster_2.5D_4K.mp4"'
    )
    print(f"\nVideo → {out_dir}/janus_cluster_2.5D_4K.mp4")
```

---

## ORDRE D'EXÉCUTION — PARALLÈLE AU RUN GPU

```bash
# CPU libre pendant que GPU simule → extraction en parallèle

# 1. Vérifier espace disque
df -h /mnt/T2
# < 50 GB libres → utiliser --radius 15.0

# 2. Compiler
cd /mnt/T2/janus-sim
cargo build --release --bin cluster_extractor

# 3. Extraction background (CPU, ~4-6h)
nohup ./target/release/cluster_extractor \
  --snap-dir    output/janus_baryonic_calibrated/snapshots/ \
  --cluster-csv output/janus_baryonic_calibrated/cluster_analysis/cluster_catalog.csv \
  --cluster-id  0 \
  --out-dir     output/cluster_lightup/gbin/ \
  --radius      25.0 \
  --sf-threshold 5.0 \
  --n-neighbors  32 \
  --box-size    500.0 \
  > output/cluster_lightup/extractor.log 2>&1 &

echo "Extractor PID: $!"
tail -f output/cluster_lightup/extractor.log

# 4. Rendu quand extraction terminée (~6-8h pour 800 frames 4K)
nohup python3 analysis/cluster_lightup.py \
  --gbin-dir output/cluster_lightup/gbin/ \
  --out-dir  output/cluster_lightup/ \
  > output/cluster_lightup/render.log 2>&1 &
```

---

## PARAMÈTRES VISUELS

| Paramètre | Valeur | Justification |
|---|---|---|
| Résolution | 3840×2160 | 4K publication |
| DPI | 200 | Qualité maximale |
| FPS | 24 | Standard cinéma |
| Élévation | 30° | Vue isométrique |
| Azimut | 45° | Vue isométrique |
| Rayon | 25 Mpc | Amas + vide m- |
| m− alpha | 0.10 | Transparent mais visible |
| m+ gaz alpha | 0.12 | Très transparent |
| proto-étoile | blanc s∝OD | Intensité variable |
| nouvelle étoile | jaune s=60 + halo orange | Flash spectaculaire |

---

## CE QU'ON VEUT VOIR

**Acte 1 (z=4→2.3)** — Homogénéité primordiale
Rouge et bleu mélangés uniformément. Légères ondulations primordiales.

**Acte 2 (z=2.3→1.55)** — Séparation
Le bleu se retire vers les bords. Le rouge se concentre au centre. Tension.

**Acte 3 (z=1.55→1.0)** — PREMIERS FLASHS
Flashs jaunes au centre. Vide bleu qui se creuse en temps réel.
Croissance exponentielle des populations stellaires.

**Acte 4 (z=1.0→0.44)** — Amas mature
Cœur blanc brillant dense. Coquille bleue diffuse autour.
Structure lacunaire de Petit. Courbes de vitesse plates.

---

## NOTES TECHNIQUES

- Toutes les particules dans le rayon — pas d'échantillonnage
- bbox FIXE sur tout le film — angle fixe 30°/45°
- is_new_star comparaison step N vs N-1 en post-processing
- Conditions aux limites périodiques dans filter_sphere()
- Extraction CPU en parallèle du run GPU
- m_particule = 5.1×10¹¹ M☉ → chaque flash = population stellaire entière

---

*Version 2.0 — Avril 2026*
*Cluster #1 : center=(-5.3, 11.2, -39.6) Mpc | M=2.24×10¹⁵ M☉ | R_half=3.63 Mpc*
