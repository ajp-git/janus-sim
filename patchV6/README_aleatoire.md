# Validation ICs v6 avant production — 5 min

## But
Vérifier statistiquement et visuellement que les ICs (random + CIC Zel'dovich)
ne produisent PAS d'artefact de grille, AVANT d'engager 30h de simulation.

## Étape 1 : run minimaliste (5 min)

Petite boîte, peu de particules, **un seul snapshot à z=10** (IC pure) :
```
./target/release/janus_adaptive_zoom \
  --n-grid 80 --l-box 100 --z-init 10.0 --z-final 9.999 \
  --snap-interval 1 --steps-check 999 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.0001 --theta 0.7 \
  --out-dir /app/output/janus_v6_ic_check \
  --run-label v6_ic_check
```

(z_final=9.999 + dt=0.0001 → ~5 steps, le snap_00000.bin est l'IC pure)

## Étape 2 : analyse

```
pip install matplotlib numpy  # si pas déjà installé
python3 validate_ics.py --snap /app/output/janus_v6_ic_check/snapshots/snap_00000.bin
```

## Critères GO / NO-GO

**GO** : max axe FFT < 1.15 (pas d'anisotropie axiale)
**AMBIGU** : max axe FFT ∈ [1.15, 1.20]
**NO-GO** : max axe FFT > 1.20 (artefact de grille présent)

## Ce que doit montrer validate_ics.png

**Panneaux (1) et (2)** : scatter de m+/m- doit être **homogène et diffus**.
Pas de motif rectiligne, pas de croix, pas de grille visible à l'œil.

**Panneau (3) - Spectre angulaire** : courbe oscillant autour de 1.0.
Les lignes orange verticales à 0°, 90°, 180°, 270°, 360° doivent NE PAS
coïncider avec des pics. Si des pics y apparaissent → artefact de grille.

**Panneau (4) - P(k)** : courbe suivant approximativement la pente k^(n_s-3) ≈ k^(-2.035).

## Si NO-GO
Envoyer l'image validate_ics.png pour diagnostic. Options :
1. Amplitude Zeldovich encore trop grande → baisser target_disp à 0.2 × spacing
2. n_grid de la FFT trop faible pour supprimer les modes axes → doubler n_grid
3. Bug dans le code CIC → vérifier ligne par ligne
