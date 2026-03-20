
# Janus Simulation Visualization Guide
## Analysis Images + Publication-Quality ("Wow") Rendering

This document defines **two complementary visualization pipelines** for Janus simulations.

1. **Analysis images**
   - diagnostic
   - used during parameter exploration
   - highlights physical issues

2. **Wow rendering**
   - Illustris/Millennium-style
   - visually reveals cosmic web structure
   - used for interpretation and presentation

Both pipelines should be generated for the **final snapshot** of each run.

```
snapshot_step = 10000
runs = 50
```

---

# PART 1 — Analysis Images (Diagnostic)

Purpose:

- detect dipoles
- detect numerical artifacts
- measure polarization structure
- verify filament formation

Each run generates:

```
run_XX_web.png
run_XX_polarization.png
run_XX_slice.png
```

Total images:

```
50 × 3 = 150
```

---

## 1 Cosmic Web Projection

Projection of total density.

```
ρ = ρ+ + ρ-
```

Projection:

```
XY
```

Processing:

```
log10 density
Gaussian smoothing σ ≈ 1–2 pixels
```

Resolution:

```
2048 × 2048
```

Color map:

```
magma
```

Purpose:

visualize

- voids
- sheets
- filaments
- nodes

---

## 2 Polarization Map

Definition:

```
P = (ρ+ − ρ−) / (ρ+ + ρ−)
```

Projection:

```
XY
```

Color map:

```
blue  = negative mass
white = neutral
red   = positive mass
```

Purpose:

detect

- Janus interfaces
- species segregation
- filament polarity

---

## 3 Diagnostic Slice

Slice plane:

```
XZ
```

Thickness:

```
~5 Mpc
```

Purpose:

detect

- dipole
- slab instabilities
- anisotropies

---

# Diagnostic Rendering Script

```python
import numpy as np
import struct
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BOX = 492
RES = 1024
HALF = BOX/2

def load_snapshot(path):

    with open(path,"rb") as f:
        n,step,_ = struct.unpack("<QQQ",f.read(24))
        data = np.frombuffer(f.read(n*16),dtype=np.float32).reshape(n,4)

    pos = data[:,:3]
    mass = data[:,3]

    return pos,mass

def grid(pos):

    xi=((pos[:,0]+HALF)/BOX*RES).astype(int)%RES
    yi=((pos[:,1]+HALF)/BOX*RES).astype(int)%RES
    zi=((pos[:,2]+HALF)/BOX*RES).astype(int)%RES

    grid=np.zeros((RES,RES,RES))

    np.add.at(grid,(xi,yi,zi),1)

    return gaussian_filter(grid,1.5)
```

---

# PART 2 — Wow Rendering (Illustris Style)

Purpose:

- reveal cosmic web visually
- highlight filaments and nodes
- generate presentation-quality images

This rendering uses:

```
thick projection
log compression
adaptive smoothing
tone mapping
```

---

## Rendering Concept

The density field is projected through the box:

```
Σ(x,y) = ∫ ρ(x,y,z) dz
```

Then processed with:

```
log scaling
contrast stretch
gamma correction
```

---

## Rendering Script

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def render_cosmic_web(grid):

    proj = grid.sum(axis=2)

    proj = gaussian_filter(proj,1.0)

    proj = np.log10(proj+1)

    proj -= proj.min()

    proj /= proj.max()

    proj = proj**0.6

    plt.figure(figsize=(10,10))

    plt.imshow(proj,cmap="inferno")

    plt.axis("off")
```

---

# Optional Node Highlight

Nodes can be emphasized by enhancing high-density peaks.

Example:

```
nodes = proj > percentile(99.5)
``

Then apply brighter color mapping.

---

# Mosaic Generation

Create mosaics to compare runs.

Example layout:

```
5 columns
10 rows
```

Script idea:

```python
import matplotlib.pyplot as plt
from PIL import Image
import glob

images = sorted(glob.glob("images/web/*.png"))

cols = 5
rows = 10
```

---

# Visual Indicators of a Good Run

Look for:

### Filament network

- connected nodes
- long filaments
- void regions

### Balanced polarization

- red/blue interfaces
- no large-scale dipole

### Filament thickness

Target:

```
1–3 Mpc
```

---

# Expected Output

```
150 diagnostic images
50 wow renderings
5 mosaics
```

---

End of document
