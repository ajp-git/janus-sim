# Relance run µ=19 propre — Procédure complète autonome

## Contexte

Le run prod µ=19 actuel (PID janus_jpp_production) est contaminé par un bug d'IC : l'IFFT du déplacement Zel'dovich dans `src/bin/janus_jpp_production.rs` lignes 235-246 ne fait que la passe x, pas les passes y et z. Conséquence : champ ψ(x,y,z) hybride k-space/real-space qui crée des structures alignées sur les axes coordonnées, visible dans les snapshots à z=3.5 (bandes verticales dans scatter brut, pic à k=L/8 dans P(k)).

L'utilisateur (AJP) part 2 heures et te demande d'agir de manière complètement autonome jusqu'à la relance du run propre. **Tu as l'autorisation de stopper le run actuel, modifier le code, recompiler, valider, et relancer**, à condition de respecter strictement la procédure ci-dessous.

## Règles d'or

1. **Si une étape échoue ou produit un résultat hors critère, NE PAS continuer**. Documente le problème dans `RELANCE_LOG.md`, laisse le système dans un état propre, et attends le retour d'AJP.
2. **Tous les fichiers modifiés doivent être commités séparément** avec message clair. Pas de commit géant.
3. **Préserve TOUJOURS les données existantes**. Tag les anciens runs avant tout, ne supprime aucun snapshot.
4. **Documente chaque étape** dans `RELANCE_LOG.md` au fur et à mesure : timestamp, action, résultat, verdict.

---

## ÉTAPE 0 — Préparation (5 min)

```bash
cd /chemin/du/projet  # adapter au repo
git status  # doit être propre, sinon stash ou commit avant
git checkout main  # branche principale post-merge
git pull  # si remote configuré
```

Crée le fichier de log :
```bash
echo "# Relance run µ=19 propre — Log autonome
Démarré : $(date -Iseconds)
Branche : $(git branch --show-current)
HEAD : $(git rev-parse --short HEAD)
" > RELANCE_LOG.md
```

---

## ÉTAPE 1 — Arrêt propre du run buggy (5 min)

### 1.1 Identifier les processus

```bash
ps -ef | grep -E "(janus_jpp_production|render_daemon|postprocess)" | grep -v grep
```

Note les PIDs dans RELANCE_LOG.md.

### 1.2 Arrêt SIGTERM ordonné

Arrête les post-processeurs d'abord (ils lisent les snapshots, pas critique de les flusher) :
```bash
kill -TERM <PID_postprocess_sigma8>
kill -TERM <PID_postprocess_lambda_debye>
kill -TERM <PID_render_daemon>
sleep 5
```

Puis le run principal :
```bash
kill -TERM <PID_janus_jpp_production>
```

Attends jusqu'à 60 secondes que le process exit propre :
```bash
for i in $(seq 1 60); do
  if ! ps -p <PID_janus_jpp_production> > /dev/null; then
    echo "Run stopped cleanly after ${i}s"
    break
  fi
  sleep 1
done
```

Si le process est encore vivant après 60s, **NE PAS faire SIGKILL**. Documente dans RELANCE_LOG.md et attends AJP. SIGKILL pourrait corrompre les HDF5 en cours d'écriture.

### 1.3 Vérification HDF5 intact

```bash
# Vérifie que le dernier snapshot écrit est lisible
ls -lh /mnt/T2/janus-sim/snapshots/janus_mu19/ | tail -5
python3 -c "import h5py; f=h5py.File('<dernier_snapshot.h5>','r'); print(list(f.keys())); f.close()"
```

Si le dernier snapshot est corrompu (lecture échoue), le supprimer (les précédents sont sains).

### 1.4 Tag git et archive

```bash
git tag run-mu19-IFFT-bug-20260428 $(git rev-parse HEAD)
mv /mnt/T2/janus-sim/snapshots/janus_mu19 /mnt/T2/janus-sim/snapshots/janus_mu19_IFFT_buggy
mv /mnt/T2/janus-sim/output /mnt/T2/janus-sim/output_IFFT_buggy
mkdir /mnt/T2/janus-sim/snapshots/janus_mu19
mkdir /mnt/T2/janus-sim/output
```

Documente dans RELANCE_LOG.md : nb snapshots conservés, dernier z atteint, taille totale archive.

---

## ÉTAPE 2 — Fix IFFT 3D (30 min)

### 2.1 Localiser le code buggué

```bash
grep -n "ifft.process" src/bin/janus_jpp_production.rs
```

Tu devrais voir les lignes 235-246 environ avec une boucle 1D-only.

### 2.2 Vérifier la convention de stockage

**CRITIQUE** : avant de coder le fix, vérifie comment psi_x, psi_y, psi_z sont stockés en mémoire.

```bash
grep -B 5 -A 30 "let mut psi_x" src/bin/janus_jpp_production.rs | head -60
```

Détermine :
- Layout : row-major (C-order) ou column-major (Fortran-order) ?
- Indexation : `psi[iz * n² + iy * n + ix]` ou `psi[ix * n² + iy * n + iz]` ?
- Type : `Vec<Complex<f64>>` ou `Vec<f64>` ?

Note la convention exacte dans RELANCE_LOG.md.

### 2.3 Implémentation du fix (3 passes IFFT)

**Approche choisie : copies temporaires** (clarté > performance, l'IC se fait une fois).

Pour layout row-major standard `psi[iz * n² + iy * n + ix]` :

```rust
// IFFT 3D complète : 3 passes successives sur axes x, y, z
let mut planner = FftPlanner::new();
let ifft = planner.plan_fft_inverse(n_fft);

// Helper pour appliquer IFFT sur les 3 champs en strided
fn ifft_axis(field: &mut Vec<Complex<f64>>, n: usize, axis: usize, ifft: &Arc<dyn Fft<f64>>) {
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    match axis {
        0 => {
            // Axis x : déjà contigu en row-major, pas de copie nécessaire
            for iz in 0..n {
                for iy in 0..n {
                    let start = iz * n * n + iy * n;
                    ifft.process(&mut field[start..start+n]);
                }
            }
        },
        1 => {
            // Axis y : stride = n
            for iz in 0..n {
                for ix in 0..n {
                    // Copie dans buffer
                    for iy in 0..n {
                        buffer[iy] = field[iz * n * n + iy * n + ix];
                    }
                    ifft.process(&mut buffer);
                    // Recopie
                    for iy in 0..n {
                        field[iz * n * n + iy * n + ix] = buffer[iy];
                    }
                }
            }
        },
        2 => {
            // Axis z : stride = n²
            for iy in 0..n {
                for ix in 0..n {
                    for iz in 0..n {
                        buffer[iz] = field[iz * n * n + iy * n + ix];
                    }
                    ifft.process(&mut buffer);
                    for iz in 0..n {
                        field[iz * n * n + iy * n + ix] = buffer[iz];
                    }
                }
            }
        },
        _ => panic!("axis must be 0, 1, or 2"),
    }
}

// Application aux 3 champs
for axis in 0..3 {
    ifft_axis(&mut psi_x, n_fft, axis, &ifft);
    ifft_axis(&mut psi_y, n_fft, axis, &ifft);
    ifft_axis(&mut psi_z, n_fft, axis, &ifft);
}
```

**ATTENTION** : 
- `Complex<f64>` doit être déjà dans le type de psi_x/y/z. Vérifie en lisant le code original. Si c'est `Vec<f64>` (real seulement), il faut une RFFT, pas FFT classique. Adapter en conséquence.
- Le facteur 1/N de l'IFFT : rustfft ne normalise pas par défaut. Si l'IFFT 1D originale ne normalisait pas, garde la même convention. Si elle divisait par N, divise par N³ après les 3 passes (ou par N à chaque passe).
- Le `Arc<dyn Fft<f64>>` peut nécessiter ajustement selon la version de rustfft. Adapter.

**Si la signature exacte du type psi_x ne permet pas une adaptation simple en 30 min**, alternative robuste : utiliser le crate `ndrustfft` qui fait directement FFT 3D :

```toml
# Cargo.toml
ndrustfft = "0.4"
ndarray = "0.15"
```

```rust
use ndrustfft::{ndifft, Complex, FftHandler};
use ndarray::{Array3, Axis};

let mut handler_x = FftHandler::new(n_fft);
let mut handler_y = FftHandler::new(n_fft);
let mut handler_z = FftHandler::new(n_fft);

let mut psi_x_3d: Array3<Complex<f64>> = Array3::from_shape_vec((n_fft, n_fft, n_fft), psi_x.clone()).unwrap();
let mut tmp = Array3::zeros((n_fft, n_fft, n_fft));

ndifft(&psi_x_3d, &mut tmp, &mut handler_x, 0);
ndifft(&tmp, &mut psi_x_3d, &mut handler_y, 1);
ndifft(&psi_x_3d, &mut tmp, &mut handler_z, 2);
psi_x = tmp.into_raw_vec();
// Idem pour psi_y, psi_z
```

Choisis la méthode selon ce qui est compatible avec le code existant. Documente le choix dans RELANCE_LOG.md.

### 2.4 Compilation

```bash
cargo build --release --bin janus_jpp_production 2>&1 | tee compile.log
```

Si erreurs de compilation, debug et reprend. Si succès :

```bash
git add src/bin/janus_jpp_production.rs Cargo.toml Cargo.lock
git commit -m "fix(IC): complete 3D IFFT in Zeldovich displacement generation

Previous code applied IFFT only along x-axis, leaving y and z in k-space.
This produced anisotropic displacement field with axis-aligned structures
visible in z<5 snapshots and a spurious peak at k=L/8 in power spectrum.

Fix: apply IFFT successively on all three axes (x, y, z) using 
[chosen method: copies temporaires / ndrustfft].

Identified via directional power spectrum P(k_x), P(k_y), P(k_z) showing
isolated peak at k=0.10 1/Mpc = 2pi/(L/8) in run-mu19-IFFT-bug-20260428.

Refs: RELANCE_LOG.md"
```

---

## ÉTAPE 3 — Audits critiques avant relance (1h)

### 3.1 Audit conservation E_VSL (CRITIQUE pour préprint)

L'objectif : déterminer si E_VSL_drift = 0.000% est une vraie mesure ou une identité numérique tautologique.

```bash
grep -n "E_VSL\|S_VSL\|E_naive" src/bin/janus_jpp_production.rs src/nbody_gpu.rs
```

Pour chaque occurrence, lis le code et détermine :
- E_naive_drift : calculé comment ? (Δ entre E courant et E initial)
- S_VSL : calculé comment ? (intégration au cours du run, ou recalcul à chaque step)
- E_VSL : calculé comment ? (E_naive + S_VSL ? E_naive - S_VSL ?)

**Test critique** : si E_VSL = E_naive ± S_VSL et S_VSL est défini comme `S_VSL = E_initial - E_naive_courant`, alors **E_VSL = E_initial par construction tautologique**. Le drift à 0.000% est mathématiquement forcé, pas une mesure.

Documente le résultat dans RELANCE_LOG.md sous une section "Audit E_VSL" avec :
- Définition exacte de E_naive (formule)
- Définition exacte de S_VSL (formule)  
- Définition exacte de E_VSL (formule)
- Verdict : conservation vraie / identité tautologique / partiellement vraie

**Si tautologique** : ne pas modifier le code maintenant, mais documenter clairement dans le préprint que la "conservation" est une identité de définition. Ne pas cacher.

**Si vraie mesure** : tout va bien, c'est un résultat publiable.

### 3.2 Vérifier seed RNG Morton offset

```bash
grep -B 3 -A 10 "Phase 13" src/nbody_gpu.rs
grep -B 3 -A 10 "thread_rng" src/nbody_gpu.rs
```

Vérifier que `rand::thread_rng()` est appelé **à l'intérieur** de la fonction d'initialisation de l'octree, pas une seule fois au démarrage du programme. Si appelé une seule fois au démarrage avec seed mémorisé → fixer.

Documente dans RELANCE_LOG.md.

### 3.3 Audit cohérence test EdS / prod

Compare les deux générateurs d'IC :

```bash
diff <(grep -A 50 "fft\|ifft\|psi\|zeldovich" src/bin/test_eds_growing_mode.rs) \
     <(grep -A 50 "fft\|ifft\|psi\|zeldovich" src/bin/janus_jpp_production.rs)
```

Si les deux ont des fonctions différentes pour la même chose (génération IC Zel'dovich), c'est un problème méthodologique.

**Action** : créer un module commun `src/ic_generator.rs` avec la fonction `generate_zeldovich_ic_3d(n_grid, l_box, sigma_8_target, h, omega_m, z_init, seed) -> (positions, velocities)`.

Importer ce module dans les deux binaires. **Ne pas dupliquer le code de génération IC.**

Si le refactoring prend > 45 minutes, l'abandonner pour cette relance et créer une issue dans RELANCE_LOG.md pour AJP.

### 3.4 Vérifier critère Courant (rapide, 10 min)

Avec dt = 0.001 Gyr et v_pec_max attendu ~7000 km/s à z<3 :
- v_max = 7000 km/s = 7.16 Mpc/Gyr
- Déplacement comoving en 1 step à z=3.5 : v·dt/a = 7.16 × 0.001 / 0.22 = 0.033 Mpc
- Ratio déplacement / softening m- (0.10 Mpc) = 0.33

**0.33 est dans la marge acceptable (< 0.5).** Pas de modification de dt nécessaire pour le run principal.

Documente dans RELANCE_LOG.md la vérification.

### 3.5 Header HDF5 enrichi (15 min)

Dans `src/bin/janus_jpp_production.rs`, à la création de chaque snapshot HDF5, ajouter ces métadonnées au header :

```rust
let git_hash = std::process::Command::new("git")
    .args(&["rev-parse", "--short", "HEAD"])
    .output()
    .ok()
    .and_then(|o| String::from_utf8(o.stdout).ok())
    .unwrap_or_else(|| "unknown".to_string());

f.attr("git_commit").write_scalar(&git_hash)?;
f.attr("timestamp").write_scalar(&chrono::Utc::now().to_rfc3339())?;
f.attr("ic_seed").write_scalar(&ic_seed)?;
f.attr("mu").write_scalar(&MU)?;
f.attr("eta").write_scalar(&ETA)?;
f.attr("box_mpc").write_scalar(&L_BOX)?;
f.attr("n_plus").write_scalar(&n_plus)?;
f.attr("n_minus").write_scalar(&n_minus)?;
f.attr("dt_gyr").write_scalar(&DT)?;
f.attr("z_init").write_scalar(&Z_INIT)?;
f.attr("ifft_3d_fix").write_scalar(&"applied")?;
```

(Adapter aux noms de variables réels du code et à l'API HDF5 utilisée)

```bash
git add src/bin/janus_jpp_production.rs
git commit -m "feat(prod): enrich HDF5 header with reproducibility metadata"
```

---

## ÉTAPE 4 — Validation IC fixée (30 min)

**Avant de lancer la prod 64h, valider que l'IC corrigée est saine.**

### 4.1 Compilation

```bash
cargo build --release --bin janus_jpp_production 2>&1 | tee compile_v2.log
```

### 4.2 Run d'IC seulement (1 step)

Configure une variable d'environnement ou un flag pour ne faire qu'1 step :

```bash
JANUS_MAX_STEPS=1 ./target/release/janus_jpp_production 2>&1 | tee ic_validation.log
```

Si le binaire ne supporte pas MAX_STEPS, fais un mini-run de 5 steps (plus prudent que 1 pour avoir un snapshot écrit).

### 4.3 Tests de validation

Crée `validate_ic.py` :

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

snap = "/mnt/T2/janus-sim/snapshots/janus_mu19/snapshot_0000.h5"
f = h5py.File(snap, 'r')

# Charger positions
pos_plus = f['positions_plus'][:]   # adapter au nom réel
pos_minus = f['positions_minus'][:]
f.close()

print(f"N+ = {len(pos_plus)}, N- = {len(pos_minus)}")

# Test 1 : scatter brut, projections xy, xz, yz
n_sub = 50000
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for col, (axis_pair, names) in enumerate([((0,1),'xy'), ((0,2),'xz'), ((1,2),'yz')]):
    for row, (pos, color, label) in enumerate([(pos_plus, 'blue', 'm+'), (pos_minus, 'red', 'm-')]):
        idx = np.random.choice(len(pos), min(n_sub, len(pos)), replace=False)
        ax1, ax2 = axis_pair
        axes[row, col].scatter(pos[idx, ax1], pos[idx, ax2], s=0.5, alpha=0.3, c=color)
        axes[row, col].set_title(f"{label} {names} z=10")
        axes[row, col].set_aspect('equal')

plt.savefig('/mnt/T2/janus-sim/output/ic_validation_scatter.png', dpi=100)
print("Scatter saved")

# Test 2 : Spectre directionnel P(k_x), P(k_y), P(k_z)
def compute_directional_pk(pos, l_box, n_grid=128):
    # CIC
    delta = np.zeros((n_grid, n_grid, n_grid))
    cell = l_box / n_grid
    idx = (pos / cell).astype(int) % n_grid
    np.add.at(delta, (idx[:,0], idx[:,1], idx[:,2]), 1)
    delta = delta / delta.mean() - 1
    
    # FFT 3D
    delta_k = np.fft.fftn(delta)
    pk_3d = np.abs(delta_k)**2 / n_grid**3
    
    # Moyennes par axe
    k_axis = np.fft.fftfreq(n_grid, d=cell) * 2 * np.pi
    pk_x = pk_3d.mean(axis=(1,2))
    pk_y = pk_3d.mean(axis=(0,2))
    pk_z = pk_3d.mean(axis=(0,1))
    
    # Premier moitié seulement (positive k)
    half = n_grid // 2
    return k_axis[:half], pk_x[:half], pk_y[:half], pk_z[:half]

l_box = 500.0
k, pkx_p, pky_p, pkz_p = compute_directional_pk(pos_plus + l_box/2, l_box)
k, pkx_m, pky_m, pkz_m = compute_directional_pk(pos_minus + l_box/2, l_box)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].loglog(k[1:], pkx_p[1:], 'r-', label='P(k_x)')
axes[0].loglog(k[1:], pky_p[1:], 'g-', label='P(k_y)')
axes[0].loglog(k[1:], pkz_p[1:], 'b-', label='P(k_z)')
axes[0].set_title('m+ directional')
axes[0].legend()
axes[0].set_xlabel('k (1/Mpc)')

axes[1].loglog(k[1:], pkx_m[1:], 'r-', label='P(k_x)')
axes[1].loglog(k[1:], pky_m[1:], 'g-', label='P(k_y)')
axes[1].loglog(k[1:], pkz_m[1:], 'b-', label='P(k_z)')
axes[1].set_title('m- directional')
axes[1].legend()
axes[1].set_xlabel('k (1/Mpc)')

plt.savefig('/mnt/T2/janus-sim/output/ic_validation_pk_directional.png', dpi=100)

# Calcul métriques d'isotropie
spread_plus = np.std([pkx_p, pky_p, pkz_p], axis=0) / np.mean([pkx_p, pky_p, pkz_p], axis=0)
spread_minus = np.std([pkx_m, pky_m, pkz_m], axis=0) / np.mean([pkx_m, pky_m, pkz_m], axis=0)

print(f"\n=== Isotropy spread ===")
print(f"m+ : mean spread = {spread_plus[1:].mean()*100:.2f}%, max = {spread_plus[1:].max()*100:.2f}%")
print(f"m- : mean spread = {spread_minus[1:].mean()*100:.2f}%, max = {spread_minus[1:].max()*100:.2f}%")

# Vérifier absence de pic à k = L/8 = 0.10 1/Mpc
k_target = 2 * np.pi / (l_box / 8)  # 0.10 1/Mpc
idx_target = np.argmin(np.abs(k - k_target))
print(f"\n=== k=L/8 peak check ===")
print(f"k target = {k_target:.4f}, idx = {idx_target}, k[idx] = {k[idx_target]:.4f}")
for label, pk_axes in [('m+', [pkx_p, pky_p, pkz_p]), ('m-', [pkx_m, pky_m, pkz_m])]:
    for axis, pk in zip(['x','y','z'], pk_axes):
        ratio = pk[idx_target] / np.mean([pk[max(0,idx_target-2):idx_target], pk[idx_target+1:idx_target+3]])
        print(f"  {label} P(k_{axis})[L/8] / neighbors ratio = {ratio:.2f}")

# Sigma_8 IC
def sigma_R(pos, l_box, R, n_grid=128):
    delta = np.zeros((n_grid, n_grid, n_grid))
    cell = l_box / n_grid
    idx = (pos / cell).astype(int) % n_grid
    np.add.at(delta, (idx[:,0], idx[:,1], idx[:,2]), 1)
    delta = delta / delta.mean() - 1
    
    delta_k = np.fft.fftn(delta)
    kx = np.fft.fftfreq(n_grid, d=cell) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Top-hat
    KR = K * R
    W = np.where(KR > 1e-10, 3 * (np.sin(KR) - KR * np.cos(KR)) / KR**3, 1.0)
    
    sigma2 = (np.abs(delta_k)**2 * W**2).sum() / n_grid**6
    return np.sqrt(sigma2)

R8 = 8.0 / 0.699  # 8 Mpc/h en Mpc
sigma8_plus = sigma_R(pos_plus + l_box/2, l_box, R8)
sigma8_minus = sigma_R(pos_minus + l_box/2, l_box, R8)
print(f"\n=== sigma_8 IC ===")
print(f"sigma_8(m+) = {sigma8_plus:.4f}")
print(f"sigma_8(m-) = {sigma8_minus:.4f}")
print(f"Target LCDM-like : ~0.07-0.10 à z=10")
```

Lance la validation :
```bash
python3 validate_ic.py 2>&1 | tee ic_validation_results.log
```

### 4.4 Critères de validation (TOUS DOIVENT PASSER)

Lis `ic_validation_results.log` et applique ces critères :

| Critère | Seuil pass | Action si fail |
|---------|-----------|----------------|
| spread P(k) m+ max | < 15% | STOP, anisotropie résiduelle |
| spread P(k) m- max | < 15% | STOP, anisotropie résiduelle |
| Ratio P[L/8]/neighbors m+ tous axes | < 1.5 | STOP, pic résiduel |
| Ratio P[L/8]/neighbors m- tous axes | < 1.5 | STOP, pic résiduel |
| sigma_8(m+) à z=10 | dans [0.05, 0.15] | STOP, IC mal normalisée |
| sigma_8(m-) à z=10 | dans [0.05, 0.20] | STOP, IC mal normalisée |
| Visual scatter (ic_validation_scatter.png) | pas de bandes alignées | STOP, anisotropie persistante |

**Si TOUS les critères passent** → IC validée, on peut continuer vers ÉTAPE 5.

**Si UN critère échoue** :
1. NE PAS lancer la prod
2. Documenter en détail dans RELANCE_LOG.md
3. Ne PAS tenter de "deviner" un autre fix
4. Laisser le système dans un état propre, attendre AJP

Le scatter PNG nécessite inspection visuelle. Si tu n'es pas sûr, copie le path dans RELANCE_LOG.md et marque "À vérifier visuellement par AJP avant prod".

### 4.5 Si validation OK

```bash
git add validate_ic.py
git commit -m "test(IC): add post-fix validation script with isotropy criteria"
git tag run-mu19-IFFT-fixed-validated-20260428
```

---

## ÉTAPE 5 — Mini-run 200 steps (15-30 min)

Avant le full 64h, mini-run pour vérifier dynamique saine :

```bash
JANUS_MAX_STEPS=200 ./target/release/janus_jpp_production 2>&1 | tee mini_run_200.log
```

(Adapter selon syntaxe MAX_STEPS du binaire)

### 5.1 Critères mini-run

À l'issue des 200 steps, lire le CSV :
```bash
tail -10 /mnt/T2/janus-sim/output/evolution_phase2.csv
```

Critères :
| Métrique | Attendu | Action si fail |
|----------|---------|----------------|
| v_rms+ à step 200 | < 500 km/s | STOP, instabilité |
| v_rms- à step 200 | < 1000 km/s | STOP, instabilité |
| corr_delta à step 200 | < 0 (anti-correlation) | STOP, segregation absente |
| ρ_max+ à step 200 | < 0.10 | STOP, clustering anormal |
| ρ_max- à step 200 | < 0.50 | STOP, clustering anormal |
| E_naive_drift step 200 | < 0.5% | STOP, énergie dérive |
| Pas de NaN | aucun | STOP, calcul instable |
| Présence de tous les snapshots attendus | 20 snaps | STOP, écriture cassée |

Génère un quick-look snapshot z=9 (step 100) :
```bash
python3 validate_ic.py --snapshot snapshot_0100.h5 --output mini_run_step100_check.png
```

Vérifie : pas de bandes alignées, P(k) directional spread < 15%, pic L/8 < 1.5×.

### 5.2 Si mini-run OK

Documente dans RELANCE_LOG.md avec les métriques. Procède à l'ÉTAPE 6.

### 5.3 Si mini-run KO

STOP. Documente. Attends AJP. Ne tente PAS de re-fixer en aveugle.

---

## ÉTAPE 6 — Lancement full prod (autonome)

### 6.1 Vérifier ressources

```bash
df -h /mnt/T2  # doit avoir > 500 GB libres pour les snapshots full run
nvidia-smi  # GPU disponible, pas de processus zombie
free -h  # RAM OK
```

Si espace disque insuffisant ou GPU occupé → STOP, documenter, attendre AJP.

### 6.2 Lancer en tmux

```bash
tmux new-session -d -s janus_prod './target/release/janus_jpp_production 2>&1 | tee /mnt/T2/janus-sim/output/prod_full.log'
```

### 6.3 Lancer post-processeurs en parallèle

```bash
# σ_8 multi-scale + cross-power (script existant déjà testé)
nohup python3 postprocess_sigma8.py > /mnt/T2/janus-sim/output/sigma8_postproc.log 2>&1 &
echo "sigma8 postproc PID: $!"

# λ_Debye (script existant déjà testé)  
nohup python3 postprocess_lambda_debye.py > /mnt/T2/janus-sim/output/lambda_debye_postproc.log 2>&1 &
echo "lambda_debye postproc PID: $!"

# Render daemon
nohup python3 render_daemon_adaptive_v2.py > /mnt/T2/janus-sim/output/render.log 2>&1 &
echo "render daemon PID: $!"
```

Documente tous les PIDs dans RELANCE_LOG.md.

### 6.4 Vérification après 5 min

```bash
sleep 300
ps -ef | grep -E "(janus_jpp_production|postprocess|render)" | grep -v grep
tail -20 /mnt/T2/janus-sim/output/prod_full.log
ls /mnt/T2/janus-sim/snapshots/janus_mu19/ | wc -l
```

Documente dans RELANCE_LOG.md :
- Tous les PIDs vivants ?
- Combien de steps fait après 5 min (estimation rate s/step) ?
- Combien de snapshots écrits ?

### 6.5 Vérification après 30 min

Idem 6.4 mais après 30 min. Vérifie aussi :
```bash
tail -5 /mnt/T2/janus-sim/output/evolution_phase2.csv
```

Critères santé à 30 min (~150-200 steps si 10s/step) :
- v_rms borné, croissance régulière
- corr_delta négatif et croissant en magnitude (bonne segregation)
- E_naive_drift < 0.1%
- Pas de NaN

Si tout OK : laisse tourner et documente "Prod healthy at T+30min, ETA ~64h".

Si problème : NE PAS arrêter, documente précisément le problème dans RELANCE_LOG.md, marque "Needs AJP review on return".

---

## ÉTAPE 7 — Rapport final

Avant ton dernier message à AJP, mets à jour `RELANCE_LOG.md` avec une section finale :

```markdown
## Synthèse pour retour AJP

### Statut final
- [✓/✗] Run buggy arrêté proprement
- [✓/✗] Bug IFFT 3D corrigé et commité (commit hash: ...)
- [✓/✗] Audits réalisés (E_VSL, Morton, IC cohérence)
- [✓/✗] IC validée (tous critères passés)
- [✓/✗] Mini-run 200 steps sain
- [✓/✗] Full prod relancée et active

### Métriques run propre à T+30min
- Step actuel : ...
- z : ...
- v_rms+/v_rms- : ...
- corr_delta : ...
- E_naive_drift : ...
- Rate : ...s/step
- ETA fin de run : ...

### Points qui nécessitent ton attention
- [Liste tout ce qui a été flagué pendant la procédure]
- [Audit E_VSL : conclusion à valider]
- [Etc.]

### Tags git créés
- run-mu19-IFFT-bug-20260428 (état avant fix)
- run-mu19-IFFT-fixed-validated-20260428 (après validation IC)
- (si tout OK) run-mu19-IFFT-fixed-running-20260428 (full prod active)

### Fichiers à examiner au retour
- RELANCE_LOG.md (ce fichier, en entier)
- /mnt/T2/janus-sim/output/ic_validation_scatter.png
- /mnt/T2/janus-sim/output/ic_validation_pk_directional.png
- /mnt/T2/janus-sim/output/mini_run_step100_check.png
- /mnt/T2/janus-sim/output/prod_full.log (si prod active)
```

---

## RÈGLES DE NON-AUTONOMIE

Tu dois STOP et attendre AJP si :

1. Le process janus_jpp_production ne s'arrête pas après 60s SIGTERM (ne pas SIGKILL)
2. La compilation après fix IFFT échoue de manière non triviale (pas juste typo)
3. Le refactoring IC commun prendrait > 45 min → l'abandonner et logguer
4. **N'IMPORTE QUEL critère de validation IC échoue (ÉTAPE 4.4)**
5. **N'IMPORTE QUEL critère du mini-run échoue (ÉTAPE 5.1)**
6. Espace disque < 500 GB ou GPU occupé par autre processus
7. Audit E_VSL révèle une identité tautologique : continue le run mais marque comme **point critique pour le préprint**
8. Tu hésites entre deux approches techniques sans avoir d'évidence claire
9. Tu dois "deviner" ou "estimer" plus de 2 fois dans une même décision

Dans tous ces cas : RELANCE_LOG.md détaillé, état système propre, attente.

## Ce que tu peux faire de manière autonome

- Stopper proprement les processus avec SIGTERM
- Modifier le code Rust (fix IFFT, header HDF5, refactoring si simple)
- Compiler, debug erreurs de syntaxe simples
- Lancer scripts de validation Python
- Lire et interpréter logs
- Tagger git, commit avec messages clairs
- Lancer la prod si tous les critères de validation sont passés
- Démarrer post-processeurs en parallèle
- Documenter dans RELANCE_LOG.md

## Ce que tu NE PEUX PAS faire

- SIGKILL sur janus_jpp_production
- Supprimer des snapshots (même de l'ancien run)
- Modifier les paramètres physiques (µ, η, box, dt, N) sans validation AJP
- Pousser vers remote git (push) sans validation AJP
- Lancer la prod si UN critère de validation échoue, même de peu
- Décider de "rerun avec autres paramètres" si quelque chose cloche
- Ignorer une discordance entre mesures
- Modifier les kernels CUDA (drift, kick, BVH) qui sont validés

## Estimations de temps

- Étape 0 : 5 min
- Étape 1 : 5-10 min  
- Étape 2 : 30-45 min (selon complexité de l'IFFT)
- Étape 3 : 1h (1.1 audit E_VSL est le plus long, 30 min)
- Étape 4 : 30 min
- Étape 5 : 30 min mini-run + 5 min vérification
- Étape 6 : 10 min lancement + 30 min vérification

**Total avant relance : ~3h**
**+ 30 min de vérification post-lancement = 3h30**

Tu as 2h. Si tu vois que ça déborde :
- Priorité absolue : Étapes 0, 1, 2, 4, 5 (fix + validation + mini-run)
- Étape 3 (audits) peut être dégradée : faire 3.1 (E_VSL critique) et 3.4 (Courant), reporter 3.2 et 3.3 à AJP
- Étape 6 (lancement full prod) : seulement si tout le reste a passé sans souci

Si tu n'as pas le temps d'arriver jusqu'au lancement full prod, c'est OK. Stoppe à mini-run validé et écris dans RELANCE_LOG.md "Ready for full prod launch, AJP to confirm".

## Une dernière chose

L'utilisateur a dit "Vérifie bien ce que tu écris". Cela s'applique à toi aussi. Avant chaque action irréversible (commit, lancement prod, kill process) :
1. Relis l'étape de cette procédure
2. Vérifie que tu as bien fait toutes les étapes amont
3. Si tu as un doute, relis RELANCE_LOG.md depuis le début
4. Si doute persiste, STOP et attends AJP

Bonne procédure.
