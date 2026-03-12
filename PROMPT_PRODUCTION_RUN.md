# MISSION : Run production Janus — N maximum + snapshots vidéo
# Lis ce fichier en entier avant toute action.

---

## CONTEXTE INDISPENSABLE

Lis ces fichiers dans l'ordre avant de toucher au code :
1. FILAMENTS_ROADMAP.md (en entier, surtout la fin)
2. RUNS.md (les entrées récentes)

Les deux bugs critiques de cette session :
- FIX-015 : box doit être proportionnelle à N^(1/3) × 2.15 Mpc
- FIX-016 : dtau_per_dt = tau_range / (TOTAL_STEPS × DT) — PAS tau_range / TOTAL_STEPS

---

## ÉTAPE 1 — Mesurer le N_max réel avec les nouvelles ICs

Avant tout run long, lancer un test de 100 steps pour mesurer la VRAM :

```
N       = valeur à tester (commencer par 12_000_000)
Box     = n_side × 2.15 Mpc  avec  n_side = N^(1/3) arrondi
ICs     = new() positifs d'abord puis négatifs (ICs février)
Virialization = virialize() PE full
Steps   = 100
Snapshots = NON (pas de sauvegarde pour ce test)
```

Mesurer nvidia-smi pendant le run. Si VRAM < 10 GB → essayer N=14M.
N_max = plus grand N avec VRAM < 11 GB (garder 1 GB de marge).
Noter le N_max retenu et calculer la box correspondante.

---

## ÉTAPE 2 — Ajouter la sauvegarde de snapshots au code

Modifier le binaire de production pour sauvegarder des snapshots binaires
toutes les SNAPSHOT_INTERVAL steps.

### Format snapshot (compact — positions + signes uniquement) :

```rust
// Structure d'un snapshot : snap_XXXXXX.bin
// Header : 3 × u64 = [n_particles, step, reserved]
// Data   : n_particles × 4 × f32 = [x, y, z, sign_as_f32]
//          sign_as_f32 = +1.0 pour masse+, -1.0 pour masse−
// Taille : 24 + N × 16 bytes
// Exemple 12M particules : 24 + 12M × 16 = ~192 MB/snapshot
```

```rust
fn save_snapshot(positions: &[f64], signs: &[i8], step: usize, output_dir: &str) {
    let path = format!("{}/snap_{:06}.bin", output_dir, step);
    let n = signs.len();
    let mut buf = Vec::with_capacity(24 + n * 16);
    // Header
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&(step as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Data
    for i in 0..n {
        buf.extend_from_slice(&(positions[i*3+0] as f32).to_le_bytes());
        buf.extend_from_slice(&(positions[i*3+1] as f32).to_le_bytes());
        buf.extend_from_slice(&(positions[i*3+2] as f32).to_le_bytes());
        buf.extend_from_slice(&(signs[i] as f32).to_le_bytes());
    }
    std::fs::write(&path, &buf).expect("snapshot write failed");
}
```

Appeler `save_snapshot()` toutes les SNAPSHOT_INTERVAL steps dans la boucle principale.

### Paramètre à définir :
```rust
const SNAPSHOT_INTERVAL: usize = 20;  // 1 snapshot / 20 steps
```

---

## ÉTAPE 3 — Vérifier l'espace disque

Avant de lancer :

```bash
df -h /mnt/T2/
```

Calcul de l'espace nécessaire :
```
N_max particules × 16 bytes × (TOTAL_STEPS / SNAPSHOT_INTERVAL) snapshots
Exemple 12M × 16 × 500 = ~96 GB
```

Si espace insuffisant : augmenter SNAPSHOT_INTERVAL à 50 ou 100.
Documenter le choix dans RUNS.md.

---

## ÉTAPE 4 — Paramètres du run production

```
N               = [N_max mesuré à l'étape 1]
Box             = n_side × 2.15 Mpc
Softening       = 0.65 Mpc
θ               = 0.7
dt              = 0.01
z_init          = 5.0
Steps           = 10000
ICs             = new() positifs d'abord puis négatifs
Virialization   = virialize() PE full
dtau_per_dt     = (tau_end - tau_start).abs() / (10000.0 * DT)  ← FIX-016 IMPÉRATIF
SNAPSHOT_INTERVAL = 20 (ou ajuster selon espace disque)
Output          = /mnt/T2/janus-sim/output/production_NM_snapshots/
```

---

## ÉTAPE 5 — Checklist avant lancement

```
□ nvidia-smi → GPU propre (0 MB utilisé)
□ df -h /mnt/T2/ → espace suffisant calculé
□ Test 100 steps → N_max confirmé
□ dtau_per_dt vérifié dans le code (PAS dtau = tau_range/TOTAL_STEPS)
□ ICs = new() positifs d'abord (PAS new_with_state() signes mélangés)
□ Virialization = virialize() PE full (PAS virialize_sampled())
□ SNAPSHOT_INTERVAL défini et cohérent avec l'espace disque
□ KE/KE₀ au step 5 < 1.05
□ git push origin main
□ Container ID sauvegardé
```

---

## ÉTAPE 6 — Monitoring

Vérifier toutes les 30 minutes :
```bash
tail -5 /mnt/T2/janus-sim/output/production_NM_snapshots/time_series.csv
ls -lh /mnt/T2/janus-sim/output/production_NM_snapshots/snap_*.bin | tail -3
df -h /mnt/T2/
```

Milestones à noter dans RUNS.md :
- Step 100 : KE/KE₀ (doit être < 0.95 — friction active)
- Step ~1800 (z≈2.5) : onset ségrégation attendu
- Step ~2100 (z≈2.2) : pic ségrégation attendu (Seg > 0.2)
- Step 10000 : run complet

---

## ÉTAPE 7 — Documentation finale

Après le run, mettre à jour RUNS.md et FILAMENTS_ROADMAP.md avec :
- N_max retenu et VRAM mesurée
- Seg_max et z du pic
- Nombre de snapshots générés et espace disque utilisé
- git push final

---

## RÈGLES ABSOLUES (rappel)

```
JAMAIS  : docker stop $(docker ps -q)
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
TOUJOURS : vérifier KE au step 5
JAMAIS  : modifier dtau_per_dt sans vérifier FIX-016
```

---

## NOTE POUR LA GÉNÉRATION D'IMAGES

Les snapshots snap_XXXXXX.bin seront traités par un script Python externe
(non géré par CLI). Format documenté à l'étape 2. CLI n'a pas à générer
d'images — sauvegarder uniquement les .bin et le time_series.csv.
