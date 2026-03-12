# MISSION : Deux runs validation 500K — comparaison ε=0.15 vs ε=0.25

## Paramètres communs
```
N          = 500_000
k_min      = 3
H          = 0.01
η          = 1.045
signes     = ALÉATOIRES
k_min      = 3 × 2π/L  (suppression modes k=1,k=2)
steps      = 5000
dt         = 0.01
θ          = 0.7
SNAPSHOT_INTERVAL = 50  → 100 snapshots
z_init     = 5.0
```

## Run A
```
nom        = val_500k_eps015
ε          = 0.15 Mpc
output     = /mnt/T2/janus-sim/output/val_500k_eps015/
```

## Run B
```
nom        = val_500k_eps025
ε          = 0.25 Mpc
output     = /mnt/T2/janus-sim/output/val_500k_eps025/
```

---

## Procédure

1. Compiler deux binaires (ou un seul avec ε en paramètre CLI)
2. Lancer Run A, attendre fin (~1h)
3. Lancer Run B, attendre fin (~1h)
4. Comparer les métriques

---

## Métriques de comparaison (à la fin de chaque run)

```bash
python3 << 'EOF'
import csv

for run in ['val_500k_eps015', 'val_500k_eps025']:
    path = f"/mnt/T2/janus-sim/output/{run}/time_series.csv"
    rows = list(csv.DictReader(open(path)))
    last = rows[-1]
    
    # Seg max
    seg_max = max(float(r['seg']) for r in rows)
    seg_final = float(last['seg'])
    ke_final = float(last['ke_ratio'])
    ke_max = max(float(r['ke_ratio']) for r in rows)
    
    print(f"\n=== {run} ===")
    print(f"  Steps      : {last['step']}")
    print(f"  z final    : {float(last['z']):.3f}")
    print(f"  Seg final  : {seg_final:.4f}")
    print(f"  Seg max    : {seg_max:.4f}")
    print(f"  KE/KE0 max : {ke_max:.3f}")
    print(f"  KE/KE0 fin : {ke_final:.3f}")
EOF
```

---

## Test Hessian sur le dernier snapshot de chaque run

```bash
python3 << 'EOF'
import struct, numpy as np
from scipy.ndimage import gaussian_filter

BOX = 492; RES = 128  # 128 suffit pour 500K

for run in ['val_500k_eps015', 'val_500k_eps025']:
    import glob, os
    snaps = sorted(glob.glob(f"/mnt/T2/janus-sim/output/{run}/snap_*.bin"))
    path = snaps[-1]
    
    with open(path, 'rb') as f:
        import struct
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        data = np.frombuffer(f.read(n*16), dtype=np.float32).reshape(n,4)
    
    pos = data[:, :3]
    half = BOX/2
    xi = ((pos[:,0]+half)/BOX*RES).astype(int)%RES
    yi = ((pos[:,1]+half)/BOX*RES).astype(int)%RES
    zi = ((pos[:,2]+half)/BOX*RES).astype(int)%RES
    grid = np.zeros((RES,RES,RES))
    np.add.at(grid,(xi,yi,zi),1)
    grid = gaussian_filter(grid, 1.2)
    
    # Variance
    var = np.var(grid / grid.mean())
    
    # Hessian
    gx,gy,gz = np.gradient(grid)
    gxx,gxy,gxz = np.gradient(gx)
    _,  gyy,gyz = np.gradient(gy)
    _,  _,  gzz = np.gradient(gz)
    
    f_count = n_count = 0
    for ii in range(0,RES,4):
        for jj in range(0,RES,4):
            for kk in range(0,RES,4):
                H = np.array([
                    [gxx[ii,jj,kk],gxy[ii,jj,kk],gxz[ii,jj,kk]],
                    [gxy[ii,jj,kk],gyy[ii,jj,kk],gyz[ii,jj,kk]],
                    [gxz[ii,jj,kk],gyz[ii,jj,kk],gzz[ii,jj,kk]]
                ])
                eig = np.linalg.eigvalsh(H)
                n_neg = np.sum(eig < 0)
                if n_neg == 2: f_count += 1
                if n_neg == 3: n_count += 1
    
    print(f"\n=== {run} (step {step}) ===")
    print(f"  σ²         : {var:.5f}  ({'✓' if var > 0.01 else '✗'})")
    print(f"  filaments  : {f_count}  ({'✓' if f_count > 200 else '✗'})")
    print(f"  nodes      : {n_count}  ({'✓' if n_count > 10  else '✗'})")
EOF
```

---

## Critères de décision

| Critère          | ε=0.15 gagne si...         | ε=0.25 gagne si...         |
|------------------|---------------------------|---------------------------|
| Filaments        | f_count nettement > ε=0.25 | comparable ou meilleur     |
| Stabilité KE     | KE/KE0 < 3.0               | KE/KE0 < 3.0               |
| Seg finale       | < 0.3 (pas de dipôle)      | < 0.3 (pas de dipôle)      |
| Variance σ²      | plus élevée                | comparable                 |

Si ε=0.15 est instable (KE > 3.0 ou crash) → choisir ε=0.25 pour production.
Si comparable → choisir ε=0.25 (plus conservateur).
Si ε=0.15 clairement meilleur en filaments ET stable → reconsidérer.

---

## RÈGLE ABSOLUE
Ne pas lancer le run 12M avant d'avoir les résultats des deux runs 500K.
