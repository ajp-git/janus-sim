# MISSION : Générer toutes les frames 4K + vidéo YouTube
# Lis ce fichier en entier avant toute action.

---

## CONTEXTE

Run en cours : production_pktrunc_12m_v2
Snapshots : /mnt/T2/janus-sim/output/production_pktrunc_12m_v2/snap_XXXXXX.bin
Script rendu : /mnt/T2/janus-sim/render_frame.py  (copier depuis outputs)
Format snapshot : header 24 bytes + N×16 bytes (x,y,z,sign en f32)
Steps : 20000 total, interval=20 → 1000 snapshots max
Run pas encore terminé → générer les frames disponibles maintenant,
compléter après la fin du run.

---

## ÉTAPE 0 — Préparer l'environnement

```bash
# Copier le script de rendu
cp /mnt/user-data/outputs/render_frame.py /mnt/T2/janus-sim/render_frame.py

# Installer dépendances si nécessaire
pip install matplotlib scipy numpy --break-system-packages --quiet

# Créer dossier frames
mkdir -p /mnt/T2/janus-sim/output/frames_4k

# Lister les snapshots disponibles
ls /mnt/T2/janus-sim/output/production_pktrunc_12m_v2/snap_*.bin | wc -l
```

---

## ÉTAPE 1 — Extraire les métadonnées depuis time_series.csv

Le script a besoin de z, seg, ke pour chaque step.
Construire un dictionnaire step → (z, seg, ke) depuis le CSV.

```python
# Script Python à exécuter une fois
import csv, os

ts = {}
csv_path = "/mnt/T2/janus-sim/output/production_pktrunc_12m_v2/time_series.csv"
with open(csv_path) as f:
    for row in csv.DictReader(f):
        step = int(row['step'])
        ts[step] = (float(row['z']), float(row['seg']), float(row['ke_ratio']))

print(f"Métadonnées chargées : {len(ts)} steps")
print(f"Steps disponibles : {min(ts.keys())} → {max(ts.keys())}")
```

---

## ÉTAPE 2 — Générer toutes les frames en parallèle

```bash
# Script de génération parallèle
python3 << 'EOF'
import subprocess, csv, os, glob
from concurrent.futures import ProcessPoolExecutor, as_completed

SNAP_DIR   = "/mnt/T2/janus-sim/output/production_pktrunc_12m_v2"
FRAMES_DIR = "/mnt/T2/janus-sim/output/frames_4k"
SCRIPT     = "/mnt/T2/janus-sim/render_frame.py"
N_WORKERS  = 4  # parallélisme CPU (pas GPU)

# Charger métadonnées
ts = {}
with open(f"{SNAP_DIR}/time_series.csv") as f:
    for row in csv.DictReader(f):
        step = int(row['step'])
        ts[step] = (float(row['z']), float(row['seg']), float(row['ke_ratio']))

# Lister snapshots disponibles
snaps = sorted(glob.glob(f"{SNAP_DIR}/snap_*.bin"))
print(f"Snapshots trouvés : {len(snaps)}")

def render_one(snap_path):
    basename = os.path.basename(snap_path)
    step_str = basename.replace('snap_', '').replace('.bin', '')
    step = int(step_str)
    out  = f"{FRAMES_DIR}/frame_{step_str}.png"

    # Skip si déjà généré
    if os.path.exists(out):
        return f"SKIP {step_str}"

    # Métadonnées
    if step in ts:
        z, seg, ke = ts[step]
    else:
        z, seg, ke = None, None, None

    cmd = [
        "python3", SCRIPT, snap_path, out
    ]
    if z   is not None: cmd.extend([str(z)])
    if seg is not None: cmd.extend([str(seg)])
    if ke  is not None: cmd.extend([str(ke)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return f"ERROR {step_str}: {result.stderr[-200:]}"
    return f"OK {step_str}"

os.makedirs(FRAMES_DIR, exist_ok=True)

with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(render_one, s): s for s in snaps}
    done = 0
    for fut in as_completed(futures):
        result = fut.result()
        done += 1
        if done % 50 == 0 or 'ERROR' in result:
            print(f"[{done}/{len(snaps)}] {result}")

print(f"\nTerminé : {done} frames générées dans {FRAMES_DIR}")
EOF
```

---

## ÉTAPE 3 — Assembler la vidéo 4K avec ffmpeg

```bash
# Vérifier que toutes les frames sont là
ls /mnt/T2/janus-sim/output/frames_4k/frame_*.png | wc -l

# Vidéo principale 4K (YouTube)
ffmpeg -y \
  -framerate 30 \
  -pattern_type glob \
  -i '/mnt/T2/janus-sim/output/frames_4k/frame_*.png' \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -pix_fmt yuv420p \
  -vf "scale=3840:2160:flags=lanczos" \
  -movflags +faststart \
  /mnt/T2/janus-sim/output/janus_segregation_4K.mp4

echo "Vidéo générée : $(du -h /mnt/T2/janus-sim/output/janus_segregation_4K.mp4)"
```

**Paramètres vidéo :**
- 30 fps × 1000 frames = ~33 secondes
- CRF 18 = haute qualité (YouTube recommande ≤ 18)
- yuv420p = compatibilité maximale
- faststart = streaming progressif

---

## ÉTAPE 4 — Vérification qualité

```bash
# Vérifier la vidéo
ffprobe -v quiet -print_format json -show_streams \
  /mnt/T2/janus-sim/output/janus_segregation_4K.mp4 | \
  python3 -c "import json,sys; s=json.load(sys.stdin)['streams'][0]; \
  print(f\"Résolution: {s['width']}×{s['height']}\"); \
  print(f\"Durée: {s['duration']}s\"); \
  print(f\"FPS: {s['r_frame_rate']}\")"

# Extraire frame de vérification (milieu de la vidéo)
ffmpeg -y -ss 16 -i /mnt/T2/janus-sim/output/janus_segregation_4K.mp4 \
  -vframes 1 /mnt/T2/janus-sim/output/check_frame_mid.png
```

Uploader check_frame_mid.png pour validation visuelle.

---

## NOTES IMPORTANTES

**Si le run n'est pas encore terminé :**
Générer les frames disponibles maintenant avec N_WORKERS=4.
Quand le run se termine, relancer le script — il skipera automatiquement
les frames déjà générées (check `os.path.exists(out)`).
Puis relancer ffmpeg pour la vidéo finale.

**Temps estimé rendu :**
- ~15-20s par frame sur CPU
- 1000 frames × 15s / 4 workers = ~60 min

**Espace disque frames :**
- ~2-3 MB par PNG 4K
- 1000 frames ≈ 2-3 GB

**Si ffmpeg n'est pas installé :**
```bash
apt-get install -y ffmpeg
```

---

## RÈGLES ABSOLUES

```
JAMAIS  : docker stop $(docker ps -q) sans filtre
TOUJOURS : vérifier espace disque avant génération
TOUJOURS : uploader check_frame_mid.png après la vidéo
```
