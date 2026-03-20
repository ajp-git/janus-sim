# JANUS ANALYSIS V6 --- CUMULATIVE PIPELINE

Version: cumulative (V1+V2+V3+V4+V5+V6) Designed for: Claude CLI
autonomous execution (\~3 hours available runtime)

------------------------------------------------------------------------

# OBJECTIVE

Perform a **complete scientific analysis** of all Janus simulation runs
and snapshots.

This pipeline integrates:

-   V1: core statistics
-   V2: density / polarization metrics
-   V3: time evolution
-   V4: cosmological statistics
-   V5: topology analysis
-   V6: percolation and large‑scale structure analysis

Goal:

Build a **full cosmological dataset** from all simulation outputs.

------------------------------------------------------------------------

# INPUT DATA

Root directory:

/mnt/T2/janus-sim/output/lhs_exploration/

Runs structure:

lhs_run_01/ lhs_run_02/ ... lhs_run_50/

Snapshots location:

snapshots/snap\_\*.bin

Snapshot format:

struct { float x float y float z float sign }

sign \> 0 → positive mass sign \< 0 → negative mass

------------------------------------------------------------------------

# PERFORMANCE CONSTRAINTS

The full analysis must complete within \~3 hours.

Rules:

-   Use **grid-based analysis**
-   Avoid particle pair loops
-   Avoid halo finder
-   Avoid O(N²) algorithms

Allowed complexity:

O(N) FFT grid operations

------------------------------------------------------------------------

# GLOBAL PARAMETERS

BOX_SIZE = 492 Mpc GRID_RESOLUTION = 256³ SMOOTHING_SIGMA = 3 voxels

Voxel size:

492 / 256 ≈ 1.92 Mpc

Effective smoothing scale:

\~5--6 Mpc

------------------------------------------------------------------------

# ANALYSIS PIPELINE

The pipeline runs in the following order.

------------------------------------------------------------------------

# STEP 1 --- LOAD SNAPSHOT

For each run:

load final snapshot:

snap_010000.bin

Separate particles:

positive particles negative particles

------------------------------------------------------------------------

# STEP 2 --- BUILD DENSITY GRIDS

Create grids:

ρ+ ρ−

Then:

ρ = ρ+ + ρ−

Apply Gaussian smoothing:

σ = 3 voxels

------------------------------------------------------------------------

# STEP 3 --- POLARIZATION FIELD

Compute:

P = (ρ+ − ρ−) / (ρ+ + ρ−)

Compute statistics:

σ_P = std(P)

mean(\|P\|)

Also store:

histogram of P

------------------------------------------------------------------------

# STEP 4 --- DENSITY CONTRAST

Compute:

δ = (ρ − mean(ρ)) / mean(ρ)

Compute:

σ_ρ = std(δ)

Then:

R = σ_ρ / σ_P

------------------------------------------------------------------------

# STEP 5 --- POWER SPECTRUM

Compute FFT of δ.

Compute radial average:

P(k)

Extract:

k_peak λ_domain = BOX_SIZE / k_peak

Store full P(k).

------------------------------------------------------------------------

# STEP 6 --- COSMIC WEB CLASSIFICATION

Compute Hessian of δ.

Compute eigenvalues.

Classify voxels:

void sheet filament node

Compute fractions:

void_fraction sheet_fraction filament_fraction node_fraction

------------------------------------------------------------------------

# STEP 7 --- JANUS CROSS CORRELATION

Compute cross field:

C = ρ+ × ρ−

Normalize by mean density.

Compute:

mean(C) std(C)

Also compute correlation coefficient:

corr(ρ+,ρ−)

------------------------------------------------------------------------

# STEP 8 --- TEMPORAL EVOLUTION

Repeat core statistics on selected snapshots:

snap_001000 snap_005000 snap_010000

Compute evolution:

σP(z) R(z) filament_fraction(z) λ_domain(z)

------------------------------------------------------------------------

# STEP 9 --- TOPOLOGY ANALYSIS (V5)

Compute topology of density field.

Metrics:

Betti_0 → connected components Betti_1 → loops Betti_2 → cavities

Also compute:

Euler characteristic

------------------------------------------------------------------------

# STEP 10 --- PERCOLATION ANALYSIS (V6)

Threshold density field.

Identify connected clusters.

Compute:

largest_cluster_fraction percolation_threshold cluster_size_distribution

------------------------------------------------------------------------

# DATASET OUTPUT

For each run record:

run_id k_min epsilon eta H alpha

sigma_P sigma_rho R

lambda_domain

filament_fraction sheet_fraction void_fraction node_fraction

mean_abs_P cross_corr_mean

betti_0 betti_1 betti_2

largest_cluster_fraction

------------------------------------------------------------------------

# OUTPUT FILES

Create directory:

analysis_v6/

Files:

analysis_v6/dataset_runs.csv analysis_v6/dataset_snapshots.csv

analysis_v6/betti_numbers.csv analysis_v6/percolation_stats.csv

------------------------------------------------------------------------

# GENERATED FIGURES

Create plots:

R_distribution.png

sigmaP_distribution.png

lambda_domain_distribution.png

filament_fraction_distribution.png

Pk_mean.png

sigmaP_vs_z.png

percolation_curve.png

------------------------------------------------------------------------

# SCIENTIFIC CHECKS

Check:

σP stability across runs

Check:

λ_domain ≈ L/k_min

Check:

filament_fraction \~ ΛCDM (\~10--15%)

Check:

percolation behaviour of cosmic web

------------------------------------------------------------------------

# FINAL SUMMARY

Produce report:

analysis_v6/report.txt

Contents:

-   statistics summary
-   mean values
-   standard deviations
-   key cosmological metrics
