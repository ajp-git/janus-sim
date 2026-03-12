#!/usr/bin/env python3
"""
V10 Fast Extended Analysis - Optimized for speed
"""

import numpy as np
import json
import os
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters
INPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500_extended"
L_BOX = 200.0
N_GRID = 256
CELL_SIZE = L_BOX / N_GRID

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading fields...")
P = np.load(f"{INPUT_DIR}/polarization_field.npy").astype(np.float64)
rho = np.load(f"{INPUT_DIR}/density_field.npy").astype(np.float64)

# ============================================================
# PART 1: GRADIENT (already done, just read)
# ============================================================
print("\n=== PART 1: GRADIENT ===")
gx, gy, gz = np.gradient(P, CELL_SIZE)
grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
mean_grad = np.mean(grad_mag)
median_grad = np.median(grad_mag)
print(f"mean(|∇P|) = {mean_grad:.4f} Mpc⁻¹")
print(f"median(|∇P|) = {median_grad:.4f} Mpc⁻¹")

# ============================================================
# PART 2: JANUS SCALE (already computed)
# ============================================================
print("\n=== PART 2: JANUS SCALE ===")
mean_P2 = np.mean(P**2)
mean_grad2 = np.mean(grad_mag**2)
L_J = np.sqrt(mean_P2 / mean_grad2)
print(f"L_J = {L_J:.4f} Mpc")

# ============================================================
# PART 3: FAST DOMAIN DETECTION
# ============================================================
print("\n=== PART 3: DOMAIN DETECTION ===")

# Use a faster approach: sample largest domains only
pos_mask = P > 0.8
neg_mask = P < -0.8

# Quick count using binary erosion to estimate domain count
n_pos_voxels = np.sum(pos_mask)
n_neg_voxels = np.sum(neg_mask)

print(f"Voxels with P > 0.8: {n_pos_voxels:,} ({100*n_pos_voxels/P.size:.1f}%)")
print(f"Voxels with P < -0.8: {n_neg_voxels:,} ({100*n_neg_voxels/P.size:.1f}%)")

# Use morphological opening to separate domains (faster than label)
from scipy.ndimage import binary_opening, generate_binary_structure

struct = generate_binary_structure(3, 1)  # 6-connectivity

# Fast domain analysis: subsample the grid
print("Analyzing domains on subsampled grid...")
subsample = 2
P_sub = P[::subsample, ::subsample, ::subsample]
cell_sub = CELL_SIZE * subsample

pos_sub = P_sub > 0.8
neg_sub = P_sub < -0.8

labeled_pos, n_pos = ndimage.label(pos_sub)
labeled_neg, n_neg = ndimage.label(neg_sub)

print(f"Positive domains (subsampled): {n_pos}")
print(f"Negative domains (subsampled): {n_neg}")

# Compute sizes of largest domains
def get_domain_stats(labeled, n_domains, cell_size):
    if n_domains == 0:
        return {"n": 0, "D_max": 0, "D_median": 0}

    sizes = ndimage.sum(np.ones_like(labeled), labeled, range(1, n_domains + 1))
    sizes = np.array(sizes)

    # Convert to diameter
    vol_mpc3 = sizes * (cell_size**3)
    diameters = 2 * (3 * vol_mpc3 / (4 * np.pi))**(1/3)

    valid = diameters[sizes > 1]

    if len(valid) == 0:
        return {"n": n_domains, "n_valid": 0, "D_max": 0, "D_median": 0}

    return {
        "n": int(n_domains),
        "n_valid": int(len(valid)),
        "D_min": float(valid.min()),
        "D_median": float(np.median(valid)),
        "D_mean": float(np.mean(valid)),
        "D_max": float(valid.max()),
        "D_p75": float(np.percentile(valid, 75)),
        "D_p90": float(np.percentile(valid, 90))
    }

stats_pos = get_domain_stats(labeled_pos, n_pos, cell_sub)
stats_neg = get_domain_stats(labeled_neg, n_neg, cell_sub)

print(f"\nPositive domains: n={stats_pos['n']}, median D={stats_pos.get('D_median', 0):.2f} Mpc")
print(f"Negative domains: n={stats_neg['n']}, median D={stats_neg.get('D_median', 0):.2f} Mpc")

# Save stats
domain_stats = {
    "threshold": 0.8,
    "subsampled": True,
    "subsample_factor": subsample,
    "effective_cell_size_mpc": cell_sub,
    "positive": stats_pos,
    "negative": stats_neg,
    "L_J_mpc": float(L_J)
}

with open(f"{OUTPUT_DIR}/domain_stats.json", 'w') as f:
    json.dump(domain_stats, f, indent=2)

# ============================================================
# PART 4: POLARIZATION-DENSITY
# ============================================================
print("\n=== PART 4: POLARIZATION-DENSITY ===")
rho_mean = np.mean(rho)
delta = (rho - rho_mean) / rho_mean

P_flat = P.flatten()
delta_flat = delta.flatten()

corr_P_delta = np.corrcoef(P_flat, delta_flat)[0, 1]
corr_absP_delta = np.corrcoef(np.abs(P_flat), delta_flat)[0, 1]

print(f"corr(P, δ) = {corr_P_delta:.4f}")
print(f"corr(|P|, δ) = {corr_absP_delta:.4f}")

# Plot
percentiles = np.arange(5, 100, 5)
delta_thresh = np.percentile(delta_flat, percentiles)
mean_absP = []
for i, p in enumerate(percentiles):
    if i == 0:
        mask = delta_flat <= delta_thresh[i]
    else:
        mask = (delta_flat > delta_thresh[i-1]) & (delta_flat <= delta_thresh[i])
    mean_absP.append(np.mean(np.abs(P_flat[mask])) if np.sum(mask) > 0 else np.nan)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(percentiles, mean_absP, 'bo-', linewidth=2, markersize=6)
ax.set_xlabel('Density Percentile')
ax.set_ylabel('Mean |P|')
ax.set_title(f'Polarization vs Density (corr(|P|, δ) = {corr_absP_delta:.3f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_density_relation.png", dpi=150)
plt.close()
print("Saved: polarization_density_relation.png")

# ============================================================
# PART 5: DOMAIN SIZE HISTOGRAM
# ============================================================
print("\n=== PART 5: HISTOGRAMS ===")

# Get all domain diameters from subsampled analysis
all_diameters = []
for labeled, n_dom in [(labeled_pos, n_pos), (labeled_neg, n_neg)]:
    if n_dom > 0:
        sizes = ndimage.sum(np.ones_like(labeled), labeled, range(1, n_dom + 1))
        sizes = np.array(sizes)
        vol = sizes * (cell_sub**3)
        diam = 2 * (3 * vol / (4 * np.pi))**(1/3)
        all_diameters.extend(diam[sizes > 1])

all_diameters = np.array(all_diameters)

if len(all_diameters) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_diameters, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(L_J, color='r', linestyle='--', linewidth=2, label=f'L_J = {L_J:.2f} Mpc')
    ax.axvline(cell_sub, color='g', linestyle=':', linewidth=2, label=f'Cell = {cell_sub:.2f} Mpc')
    ax.axvline(np.median(all_diameters), color='b', linestyle='--', linewidth=2,
               label=f'Median = {np.median(all_diameters):.2f} Mpc')
    ax.set_xlabel('Domain Diameter (Mpc)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Domain Size Distribution (|P| > 0.8)')
    ax.legend()
    ax.set_xlim(0, min(30, np.percentile(all_diameters, 99)))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/domain_size_hist.png", dpi=150)
    plt.close()
    print("Saved: domain_size_hist.png")

# ============================================================
# PART 6: 3D VISUALIZATION (simplified - just static image)
# ============================================================
print("\n=== PART 6: 3D VISUALIZATION ===")

try:
    import pyvista as pv
    pv.start_xvfb()

    # Subsample for faster rendering
    P_viz = P[::2, ::2, ::2]
    from scipy.ndimage import gaussian_filter
    P_smooth = gaussian_filter(P_viz, sigma=0.5)

    ng_viz = P_viz.shape[0]
    cell_viz = L_BOX / ng_viz

    grid = pv.ImageData(dimensions=(ng_viz+1, ng_viz+1, ng_viz+1),
                        spacing=(cell_viz, cell_viz, cell_viz),
                        origin=(0, 0, 0))

    P_padded = np.pad(P_smooth, ((0,1), (0,1), (0,1)), mode='edge')
    grid.point_data["P"] = P_padded.flatten(order='F')

    iso_pos = grid.contour([0.8], scalars="P")
    iso_neg = grid.contour([-0.8], scalars="P")

    print(f"Positive surface: {iso_pos.n_cells} cells")
    print(f"Negative surface: {iso_neg.n_cells} cells")

    # Create single high-quality image
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    if iso_pos.n_cells > 0:
        plotter.add_mesh(iso_pos, color='red', opacity=0.6)
    if iso_neg.n_cells > 0:
        plotter.add_mesh(iso_neg, color='blue', opacity=0.6)

    plotter.add_mesh(pv.Box(bounds=[0, L_BOX, 0, L_BOX, 0, L_BOX]),
                     style='wireframe', color='white', line_width=1)

    center = [L_BOX/2, L_BOX/2, L_BOX/2]
    radius = L_BOX * 1.8
    plotter.camera.position = [center[0] + radius*0.7, center[1] + radius*0.7, center[2] + radius*0.5]
    plotter.camera.focal_point = center
    plotter.camera.up = [0, 0, 1]

    plotter.screenshot(f"{OUTPUT_DIR}/janus_3d_view.png")
    plotter.close()
    print("Saved: janus_3d_view.png")

    # Create rotation animation
    print("Creating rotation animation...")
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    if iso_pos.n_cells > 0:
        plotter.add_mesh(iso_pos, color='red', opacity=0.6)
    if iso_neg.n_cells > 0:
        plotter.add_mesh(iso_neg, color='blue', opacity=0.6)

    plotter.add_mesh(pv.Box(bounds=[0, L_BOX, 0, L_BOX, 0, L_BOX]),
                     style='wireframe', color='white', line_width=1)

    plotter.camera.focal_point = center
    plotter.camera.up = [0, 0, 1]

    n_frames = 120
    os.makedirs(f"{OUTPUT_DIR}/frames", exist_ok=True)

    for i in range(n_frames):
        angle = i * 360 / n_frames
        cam_x = center[0] + radius * np.cos(np.radians(angle))
        cam_y = center[1] + radius * np.sin(np.radians(angle))
        cam_z = center[2] + radius * 0.4
        plotter.camera.position = [cam_x, cam_y, cam_z]
        plotter.screenshot(f"{OUTPUT_DIR}/frames/frame_{i:04d}.png")
        if (i + 1) % 30 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    plotter.close()

    # Compile video
    import subprocess
    video_path = f"{OUTPUT_DIR}/janus_polarization_rotation.mp4"
    cmd = ['ffmpeg', '-y', '-framerate', '30',
           '-i', f'{OUTPUT_DIR}/frames/frame_%04d.png',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
           video_path]
    subprocess.run(cmd, capture_output=True)

    if os.path.exists(video_path):
        print(f"Saved: janus_polarization_rotation.mp4")
        import shutil
        shutil.rmtree(f"{OUTPUT_DIR}/frames")

except Exception as e:
    print(f"3D visualization error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  GRADIENT:
    mean(|∇P|) = {mean_grad:.4f} Mpc⁻¹

  CHARACTERISTIC SCALE:
    L_J = {L_J:.4f} Mpc
    L_J / cell = {L_J / CELL_SIZE:.2f}

  DOMAINS (|P| > 0.8):
    Positive: n={stats_pos['n']}, D_median={stats_pos.get('D_median', 0):.2f} Mpc
    Negative: n={stats_neg['n']}, D_median={stats_neg.get('D_median', 0):.2f} Mpc

  DENSITY COUPLING:
    corr(|P|, δ) = {corr_absP_delta:.4f}
""")
print("=" * 60)
