# Zoom Adaptatif Progressif — Plan et Algorithme
*Avril 2026 — Version 2.0 — Lu et relu 3×*

---

## Vision

Un seul run continu z=10→0, avec une résolution qui augmente
automatiquement là où la matière s'effondre.
Pas de zoom "à la main" — la simulation se raffine d'elle-même
en suivant les sur-densités m+ au fur et à mesure qu'elles se forment.

La vidéo résultante : on part d'une vue cosmique (500 Mpc) et
on zoome progressivement sur 2-4 halos qui s'allument, comme une caméra
qui suit l'effondrement en temps réel.

---

## Principe fondamental : le splitting automatique

Au lieu de définir manuellement 3 niveaux de zoom (L1, L2, L3),
la simulation surveille en continu la densité locale.
Quand une région dépasse un seuil δ_split, les particules y sont
automatiquement subdivisées (×8 en volume = ×2 linéaire).

```
δ_local > δ_split[niveau] → split automatique dans cette région
```

Chaque split double la résolution spatiale. 7 splits = ×128.
C'est l'équivalent de "100 niveaux progressifs" en pratique.

---

## Format Snapshot v3 — auto-descriptif

Chaque snapshot contient toute l'information nécessaire
pour reprendre la simulation ou l'analyser sans paramètres externes.

### Header fixe (~380 bytes)

```
Offset  Taille  Type    Champ              Description
──────  ──────  ──────  ─────────────────  ───────────────────────────────
0       8       u64     magic              = 0x4A414E555356330A ("JANUSV3\n")
8       4       u32     version            = 3
12      4       u32     header_size        = 384 (bytes, extensible)
16      8       u64     N_total            nombre total de particules
24      8       f64     a                  facteur d'échelle (= 1/(1+z))
32      8       f64     t_Gyr              temps cosmique en Gyr
40      8       f64     L_box              taille de la boîte en Mpc
48      8       f64     H0                 constante de Hubble (km/s/Mpc)
56      8       f64     mu                 μ = ratio masse m-/m+
64      8       f64     omega_b            Ω_b (baryons uniquement)
72      8       f64     m_part_plus_base   masse de base m+ avant splits (M☉)
80      8       f64     m_part_minus_base  masse de base m- avant splits (M☉)
88      8       f64     eps_plus_base      softening de base m+ (Mpc)
96      8       f64     eps_minus_base     softening de base m- (Mpc)
104     4       u32     n_split_max        niveau de split max utilisé
108     4       u32     seed_ic            graine des ICs
112     8       f64     z_init             z de départ des ICs
120     8       u64     N_stars            nombre d'étoiles formées
128     8       f64     z_start_run        z de départ du run adaptatif
136     8       f64     SFR                taux de formation stellaire (M☉/Gyr)
144     8       f64     rho_max            densité maximum actuelle (M☉/Mpc³)
152     256     [u8;256] run_label         nom du run (UTF-8, null-padded)
408     N/A     —       début des particules
```

Note : `header_size` permet d'étendre le header dans les versions futures
sans casser la compatibilité. Le lecteur saute toujours `header_size` bytes
avant de lire les particules.

### Format particule — 36 bytes par particule

```
Offset  Taille  Type    Champ        Description
──────  ──────  ──────  ──────────   ───────────────────────────────
0       4       f32     pos_x        position X (Mpc)
4       4       f32     pos_y        position Y (Mpc)
8       4       f32     pos_z        position Z (Mpc)
12      4       f32     vel_x        vitesse X (km/s)
16      4       f32     vel_y        vitesse Y (km/s)
20      4       f32     vel_z        vitesse Z (km/s)
24      4       f32     mass         masse (M☉) — varie selon split_level
28      4       f32     epsilon      softening individuel (Mpc)
32      1       u8      sign         +1 = m+, -1 = m- (stocké comme 1 ou 255)
33      1       u8      split_level  niveau de raffinement (0 = original)
34      1       u8      is_star      0 = gaz/DM, 1 = étoile formée
35      1       u8      flags        réservé (bit 0: is_HR, bit 1: is_active)
──────────────────────────────────────────────────────────────────
Total : 36 bytes/particule
```

### Relations entre champs

```
mass(particule)    = m_part_plus_base / 8^split_level   (pour m+)
                   = m_part_minus_base / 8^split_level  (pour m-)
epsilon(particule) = eps_plus_base / 2^split_level      (pour m+)
                   = eps_minus_base / 2^split_level     (pour m-)

Vérification : mass × 8^split_level = m_part_base (constante par espèce)
```

### Module src/snapshot_v3.rs

```rust
pub struct SnapshotHeaderV3 {
    pub magic:              u64,
    pub version:            u32,
    pub header_size:        u32,
    pub n_total:            u64,
    pub a:                  f64,
    pub t_gyr:              f64,
    pub l_box:              f64,
    pub h0:                 f64,
    pub mu:                 f64,
    pub omega_b:            f64,
    pub m_part_plus_base:   f64,
    pub m_part_minus_base:  f64,
    pub eps_plus_base:      f64,
    pub eps_minus_base:     f64,
    pub n_split_max:        u32,
    pub seed_ic:            u32,
    pub z_init:             f64,
    pub n_stars:            u64,
    pub z_start_run:        f64,
    pub sfr:                f64,
    pub rho_max:            f64,
    pub run_label:          [u8; 256],
}

pub struct ParticleV3 {
    pub pos:         [f32; 3],
    pub vel:         [f32; 3],
    pub mass:        f32,
    pub epsilon:     f32,
    pub sign:        u8,
    pub split_level: u8,
    pub is_star:     u8,
    pub flags:       u8,
}

// API publique
pub fn write_snapshot_v3(path: &Path, header: &SnapshotHeaderV3,
                         particles: &[ParticleV3]) -> Result<()>

pub fn read_snapshot_v3(path: &Path)
    -> Result<(SnapshotHeaderV3, Vec<ParticleV3>)>

pub fn read_header_only(path: &Path) -> Result<SnapshotHeaderV3>
// → rapide, ne charge pas les particules
// → utile pour identifier z, L_box, N sans tout lire

pub fn snapshot_info(path: &Path) -> Result<String>
// → affiche un résumé lisible du snapshot
// → utilisé pour le diagnostic
```

### Tests unitaires obligatoires

```rust
#[test]
fn test_roundtrip_header() {
    // Écrire puis relire 100 particules
    // Vérifier que TOUS les champs header sont bit-à-bit identiques
    // Vérifier magic, version, N_total, a, L_box, mu
}

#[test]
fn test_split_level_consistency() {
    // Vérifier mass × 8^split_level = m_part_base pour chaque particule
    // Vérifier epsilon × 2^split_level = eps_base
}

#[test]
fn test_header_size_field() {
    // Vérifier que header_size = 408 (fixe pour v3)
    // Vérifier que le lecteur peut sauter header_size bytes
    // sans connaître la structure
}

#[test]
fn test_read_header_only_fast() {
    // Créer un snapshot de 1M particules
    // Vérifier que read_header_only() lit en < 1ms
    // (ne doit pas lire les données particules)
}
```

---

## Architecture — deux composants

### Composant 1 : Run de référence (déjà fait)

```
Run principal : output/janus_baryonic_calibrated/
  L_box  = 500 Mpc
  N      = 10M particules
  z      : 5 → 0  (ou 10 → 0 si on relance)
  Format : JSNP v1 (ancien format, sans header auto-descriptif)
  Rôle   : fournit l'environnement cosmologique correct
           et identifie les halos à suivre

Note : les anciens snapshots JSNP v1 seront convertis en v3
via un script de migration (src/bin/migrate_snapshots.rs)
```

### Composant 2 : Run adaptatif (à implémenter)

```
Démarre depuis un snapshot v3 du run principal (z_start)
Extrait une boîte autour du halo cible (R_extract = 50-100 Mpc)
Évolue z_start → 0 avec splitting automatique
Écrit uniquement des snapshots v3 (auto-descriptifs)
```

---

## Algorithme détaillé

### Étape 0 — Identification des halos cibles

```python
# Script : identify_halos.py
# Sur le snapshot final du run principal (z=0) :
# 1. Calculer la densité locale par KNN (k=32 voisins)
# 2. Trier par densité décroissante
# 3. Identifier N_halos = 3-4 pics de densité distincts
#    (séparés d'au moins 10 Mpc)
# 4. Écrire halos.json avec position, rho_max, z_form estimé

halos = identify_top_halos(snap_final, N_halos=4, min_separation=10.0)

# halos.json :
# [
#   {"id":0, "pos":[-5.3, 11.2, -39.6], "rho_max":787, "z_form":0.46},
#   {"id":1, "pos":[42.1, -18.7, 103.2], "rho_max":234, "z_form":0.31},
#   ...
# ]
```

### Étape 1 — Extraction et conversion en v3

```rust
// Lire un snapshot v1 (run principal), convertir en v3
// Extraire la région d'intérêt autour du halo cible

fn extract_and_convert(
    snap_v1_path: &Path,
    center: [f64; 3],
    r_extract: f64,       // 80-100 Mpc
    run_params: &RunParams,
) -> Result<(SnapshotHeaderV3, Vec<ParticleV3>)> {

    // 1. Lire le snapshot v1 (ancien format)
    let (pos, vel, signs) = read_snapshot_v1(snap_v1_path)?;

    // 2. Filtrer les particules dans la région
    let in_region = filter_sphere(pos, center, r_extract);

    // 3. Construire les particules v3
    // Toutes les particules démarrent à split_level = 0
    // mass et epsilon = valeurs de base du run principal

    // 4. Construire le header v3 avec tous les paramètres
    let header = SnapshotHeaderV3 {
        l_box: r_extract * 2.0,
        m_part_plus_base: run_params.m_part_plus,
        eps_plus_base: run_params.eps_plus,
        n_split_max: 0,
        run_label: b"adaptive_zoom_extracted\0...",
        // ... autres champs
    };

    Ok((header, particles_v3))
}
```

### Étape 2 — Boucle principale avec splitting adaptatif

```
POUR chaque step de simulation :

1. Calculer les forces GPU (BH + SPH)

2. Intégrer les positions/vitesses (leapfrog)

3. Tous les N_check = 50 steps :
   Calculer les densités SPH pour toutes les m+ actives
   
   POUR chaque particule i :
     si sign[i] != +1 → skip (m- pas de split)
     ρ = densité_sph[i]
     L = split_level[i]
     
     si ρ > DELTA_SPLIT[L] ET L < N_SPLIT_MAX ET N_total < N_MAX :
       → créer 8 filles autour de i (Blue Noise, rayon = h_sph/3)
       → mass_fille = mass_i / 8
       → epsilon_fille = epsilon_i / 2
       → split_level_fille = L + 1
       → vel_fille = vel_i + gaussienne(σ = 1 km/s)
       → supprimer la particule mère i
       → N_total += 7  (8 filles - 1 mère)
       → mettre à jour l'arbre BH

4. Tous les 20 steps : écrire snapshot v3
   (header avec N_total, split_max courant, rho_max, N_stars)

5. Tous les 5 steps : écrire ligne CSV métriques
```

### Étape 3 — Critères de splitting

```rust
// Seuils de déclenchement (densité locale en M☉/Mpc³)
// Calibrés sur les résultats du zoom v1 :
// z=0.46 : rho_max = 787 → premier split à ~1000
const DELTA_SPLIT: [f64; 10] = [
    1.0e3,   // niveau 0→1 : filament naissant
    1.0e4,   // niveau 1→2 : proto-halo
    1.0e5,   // niveau 2→3 : halo en formation
    5.0e5,   // niveau 3→4 : cœur dense
    2.0e6,   // niveau 4→5 : pré-effondrement baryonique
    1.0e7,   // niveau 5→6 : effondrement baryonique
    5.0e7,   // niveau 6→7 : seuil SF approche
    2.0e8,   // niveau 7→8 : SF active
    1.0e9,   // niveau 8→9 : cœur stellaire dense
    1.0e10,  // niveau 9→10: réservé
];

// Résolution atteinte à chaque niveau (depuis ε_0 = 0.5 Mpc) :
// niveau 0 : ε = 500 kpc  (résolution run principal)
// niveau 1 : ε = 250 kpc
// niveau 2 : ε = 125 kpc
// niveau 3 : ε =  63 kpc
// niveau 4 : ε =  31 kpc
// niveau 5 : ε =  16 kpc
// niveau 6 : ε =   8 kpc
// niveau 7 : ε =   4 kpc  ← résolution sub-kpc ✓

// Contrainte VRAM RTX 3060 (12 GB) :
const N_MAX_TOTAL: usize = 8_000_000;
// Estimation : 4 halos × 10 000 m+ × 128 = 5.1M + 3M BG ✓
```

### Étape 4 — Zoom du renderer (continu)

```python
# Le renderer calcule r_zoom depuis le CSV à chaque frame

def compute_zoom_radius(step, csv_data, particles):
    # Trouver les particules à split_level > 0 (région active)
    active = particles[particles.split_level > 0]
    
    if len(active) == 0:
        return R_MAX  # pas encore de split → vue large
    
    # Rayon contenant 90% des particules actives
    center = active.pos.mean(axis=0)
    distances = np.linalg.norm(active.pos - center, axis=1)
    r_active = np.percentile(distances, 90)
    
    # Zoom = 1.5× le rayon actif (marge visuelle)
    r_zoom = max(r_active * 1.5, R_MIN)
    
    return r_zoom

# Lissage exponentiel pour éviter les sauts visuels
r_smooth = r_smooth_prev * 0.95 + r_zoom * 0.05
```

---

## Pipeline complet

```
[Run principal z=5→0, 500 Mpc, JSNP v1]  ← déjà fait
         │
         ▼
[identify_halos.py]
  → halos.json : 4 halos identifiés avec positions et z_form
         │
         ▼ pour chaque halo séquentiellement
         │
[extract_and_convert]
  → Lecture snapshot v1 au z_start
  → Conversion JSNP v1 → v3
  → Extraction région 80 Mpc centrée sur halo
  → snap_halo_N_start.bin (format v3)
         │
         ▼
[janus_adaptive_zoom]  z_start → z=0
  Boucle leapfrog + BH + SPH
  Vérification densités tous les 50 steps
  Splitting automatique quand δ > seuil
  Snapshots v3 toutes les 20 steps
  CSV métriques toutes les 5 steps
         │
         ▼
[adaptive_renderer.py]
  Lit snapshots v3 (header auto-descriptif → pas de paramètres)
  Zoom radius calculé depuis split_level des particules
  Génère frames + vidéo
```

---

## Ce qu'on verra dans la vidéo finale

```
z=5 → z=3   : Vue large ~80 Mpc, split_level=0 partout
               Filaments m+ se dessinent (toile cosmique)
               m- crée les vides (ségrégation visible)

z=3 → z=1   : Zoom progressif 80 → 20 Mpc
               Proto-halos visibles, split_level=1-2 dans les nœuds
               Résolution ×2-4 dans les structures

z=1 → z=0.3 : Zoom 20 → 5 Mpc
               Halo central en effondrement
               split_level=3-5 dans le cœur, ε = 16-63 kpc

z=0.3 → z=0 : Zoom 5 → 0.5 Mpc
               Étoiles qui s'allument (N★ croît visiblement)
               split_level=6-7, ε = 4-8 kpc
               Vide m- parfaitement établi dans le rayon HR
               SFR visible sur le panneau temps réel
```

---

## Étapes d'implémentation pour CLI

```
ÉTAPE 1 — src/snapshot_v3.rs  (fondation, à faire EN PREMIER)
  Struct SnapshotHeaderV3
  Struct ParticleV3
  write_snapshot_v3(), read_snapshot_v3(), read_header_only()
  snapshot_info() → affichage diagnostique
  4 tests unitaires (voir section Tests)
  cargo test snapshot_v3
  DURÉE : 2h

ÉTAPE 2 — src/bin/migrate_snapshots.rs
  Lire JSNP v1, écrire JSNP v3
  Paramètres du run injectés via CLI (m_part, eps, etc.)
  Tester sur un snapshot du run principal
  DURÉE : 1h

ÉTAPE 3 — src/bin/identify_halos.rs (ou Python)
  Lire snapshot v3 final
  KNN densité (k=32)
  Identifier 4 pics séparés de > 10 Mpc
  Écrire halos.json
  DURÉE : 1h

ÉTAPE 4 — src/adaptive_split.rs
  fn check_and_split(particles, delta_split, n_max) → Vec<ParticleV3>
  Réutiliser fibonacci_sphere() existant
  Réutiliser blue_noise_positions() existant
  Tests unitaires : conservation masse, split_level cohérent
  DURÉE : 2h

ÉTAPE 5 — src/bin/janus_adaptive_zoom.rs
  Réutiliser : GpuSphPressure, GpuBH, star_formation
  Lire snap v3, boucle principale, appel adaptive_split
  Écrire snap v3 + CSV
  DURÉE : 3h

ÉTAPE 6 — adaptive_renderer.py
  Lire snap v3 (header auto-descriptif → 0 paramètre externe)
  Zoom radius depuis split_level
  Lissage exponentiel du zoom
  DURÉE : 1h

TOTAL DÉVELOPPEMENT : ~10h
DURÉE RUN (4 halos × 20h) : ~80h séquentiels (~3-4 jours)
```

---

## Règles de codage pour CLI

```
1. Toujours écrire des snapshots v3 — jamais v1
2. Toujours lire le header avant les particules
   (read_header_only() pour le diagnostic rapide)
3. Ne jamais hardcoder L_box, m_part, eps — toujours depuis le header
4. Vérifier magic = 0x4A414E555356330A au début de chaque lecture
5. Vérifier version = 3 et header_size = 408
```

---

## Ce qu'on NE fait PAS

```
✗ Boîte indépendante 50 Mpc (trop petite, physique incorrecte)
✗ Zoom Lagrangien complet depuis z=10 (trop complexe)
✗ Relaxation artificielle (Zel'dovich → équilibre naturel)
✗ Damping de vitesses (brise la conservation)
✗ Niveaux de zoom hardcodés (c'est le concept abandonné)
✗ Snapshots sans header auto-descriptif (impossible à relire)
```

---

## Question ouverte — z_start

Avant de lancer ÉTAPE 2, vérifier le z du premier snapshot :

```bash
# Sur le serveur :
python3 -c "
import struct, os
snap_dir = 'output/janus_baryonic_calibrated/snapshots/'
first = sorted(os.listdir(snap_dir))[0]
with open(snap_dir + first, 'rb') as f:
    N = struct.unpack('Q', f.read(8))[0]
    a = struct.unpack('d', f.read(8))[0]
    print(f'{first}: N={N:,}, z={1/a-1:.2f}')
"
```

- Si z_start ≈ 5 → Option B : parfait, on démarre de là
- Si z_start < 3 → Option A : on part d'un état déjà effondré,
  moins intéressant mais fonctionnel
- Si z_start = 0 (un seul snapshot final) → relancer le run
  principal avec plus de snapshots (snap_interval plus serré)

---

*Version 2.0 — Avril 2026*
*Mise à jour : header snapshot v3 complet avec offsets et tailles*
*Donner à CLI — commencer par ÉTAPE 1 uniquement*
