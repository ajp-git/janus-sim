# Tâche : Analyse spatiale du méga-halo− (168, 127, 73) Mpc

## Objectif
Lancer `analyse_halo_spatial.py` sur les snapshots Janus disponibles,
produire les figures d'analyse, et synthétiser les résultats.

## Snapshots cibles (par ordre de priorité)
1. step 3500 (z=0.38) — état final, merger accompli
2. step 1500 (z=1.63) — ségrégation en cours
3. step 500  (z=3.39) — état initial mélangé

## Étapes

1. Localiser les snapshots dans le répertoire de sortie de la simulation :
   - Chercher dans : `output/`, `snapshots/`, `/mnt/T2/janus-sim/output/`
   - Formats possibles : `.hdf5`, `.h5`, `.npy`, binaire Rust (`.bin`)
   - Lister les fichiers disponibles et identifier le format exact

2. Adapter `analyse_halo_spatial.py` si nécessaire :
   - Si HDF5 : vérifier les clés avec `h5py` (`f.visit(print)`)
   - Si binaire Rust : vérifier le stride (7 floats par particule ? 8 ?)
   - Si NPY : vérifier la shape

3. Lancer pour chaque snapshot disponible :
   ```bash
   python analyse_halo_spatial.py --snap <chemin> --step <N> --z <valeur>
   ```

4. Vérifier les figures produites (`halo_spatial_stepXXXX.png`) :
   - Profil de densité m− doit être piqué au centre
   - Pureté P(r) doit être ≈ 1.0 pour r < 60 Mpc au step 3500
   - Fraction de fuite m+ doit être > 0.5 pour r < 60 Mpc

5. Produire un rapport `halo_spatial_rapport.md` avec :
   - Tableau comparatif des 3 steps (densité centrale, pureté, fuite)
   - Interprétation physique de chaque panneau
   - Les valeurs numériques clés pour JPP

## Notes
- HALO_POS = [168, 127, 73] Mpc, BOX_SIZE = 256 Mpc
- Rayon étendu = 120 Mpc (vs 60 Mpc dans le suivi CSV)
- Ne pas modifier les paramètres physiques du halo
- Si un snapshot manque, utiliser les disponibles et le noter dans le rapport
