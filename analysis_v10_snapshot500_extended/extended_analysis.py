#!/usr/bin/env python3
"""
V10 Extended Analysis - Gradient, Domain Walls, Janus Scale, 3D Visualization
"""

import numpy as np
import json
import os
from scipy import ndimage
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
INPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500_extended"
L_BOX = 200.0  # Mpc
N_GRID = 256
CELL_SIZE = L_BOX / N_GRID  # 0.78125 Mpc

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD FIELDS
# ============================================================
print("=" * 60)
print("LOADING FIELDS")
print("=" * 60)

P = np.load(f"{INPUT_DIR}/polarization_field.npy").astype(np.float64)
rho = np.load(f"{INPUT_DIR}/density_field.npy").astype(np.float64)

print(f"  P shape: {P.shape}")
print(f"  P range: [{P.min():.4f}, {P.max():.4f}]")
print(f"  rho shape: {rho.shape}")
print(f"  Cell size: {CELL_SIZE:.4f} Mpc")

# ============================================================
# PART 1: GRADIENT / DOMAIN WALL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PART 1: GRADIENT / DOMAIN WALL ANALYSIS")
print("=" * 60)

# Compute gradient (with proper spacing in Mpc)
gx, gy, gz = np.gradient(P, CELL_SIZE)
grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

mean_grad = np.mean(grad_mag)
median_grad = np.median(grad_mag)
max_grad = np.max(grad_mag)

print(f"  mean(|∇P|) = {mean_grad:.4f} Mpc⁻¹")
print(f"  median(|∇P|) = {median_grad:.4f} Mpc⁻¹")
print(f"  max(|∇P|) = {max_grad:.4f} Mpc⁻¹")

# Gradient histogram
fig, ax = plt.subplots(figsize=(10, 6))
grad_flat = grad_mag.flatten()
ax.hist(grad_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
ax.axvline(mean_grad, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_grad:.3f}')
ax.axvline(median_grad, color='g', linestyle='--', linewidth=2, label=f'Median = {median_grad:.3f}')
ax.set_xlabel('|∇P| (Mpc⁻¹)')
ax.set_ylabel('Probability Density')
ax.set_title('Polarization Gradient Magnitude (Domain Wall Indicator)')
ax.legend()
ax.set_xlim(0, np.percentile(grad_flat, 99))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gradient_hist.png", dpi=150)
plt.close()
print(f"  Saved: gradient_hist.png")

# Gradient slice visualization
mid = N_GRID // 2
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Polarization slice
ax = axes[0]
im = ax.imshow(P[:, :, mid].T, origin='lower', cmap='RdBu_r',
               vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title('Polarization P')
plt.colorbar(im, ax=ax, label='P')

# Gradient slice
ax = axes[1]
im = ax.imshow(grad_mag[:, :, mid].T, origin='lower', cmap='hot',
               extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title('|∇P| (Domain Walls)')
plt.colorbar(im, ax=ax, label='|∇P| (Mpc⁻¹)')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gradient_slice.png", dpi=150)
plt.close()
print(f"  Saved: gradient_slice.png")

# ============================================================
# PART 2: CHARACTERISTIC JANUS SCALE
# ============================================================
print("\n" + "=" * 60)
print("PART 2: CHARACTERISTIC JANUS SCALE")
print("=" * 60)

mean_P2 = np.mean(P**2)
mean_grad2 = np.mean(grad_mag**2)

# L_J = sqrt(<P²> / <|∇P|²>)
L_J = np.sqrt(mean_P2 / mean_grad2)

print(f"  <P²> = {mean_P2:.6f}")
print(f"  <|∇P|²> = {mean_grad2:.6f} Mpc⁻²")
print(f"  L_J = sqrt(<P²>/<|∇P|²>) = {L_J:.4f} Mpc")

# Alternative: correlation length from P(k)
# L_corr ≈ 2π / k_peak

with open(f"{OUTPUT_DIR}/janus_scale.txt", 'w') as f:
    f.write(f"Characteristic Janus Scale Analysis\n")
    f.write(f"====================================\n\n")
    f.write(f"<P²> = {mean_P2:.6f}\n")
    f.write(f"<|∇P|²> = {mean_grad2:.6f} Mpc⁻²\n")
    f.write(f"\n")
    f.write(f"L_J = sqrt(<P²> / <|∇P|²>) = {L_J:.4f} Mpc\n")
    f.write(f"\n")
    f.write(f"Interpretation:\n")
    f.write(f"  This is the characteristic scale over which\n")
    f.write(f"  polarization varies significantly.\n")
    f.write(f"\n")
    f.write(f"  L_J / cell_size = {L_J / CELL_SIZE:.2f}\n")
print(f"  Saved: janus_scale.txt")

# ============================================================
# PART 3: IMPROVED DOMAIN DETECTION
# ============================================================
print("\n" + "=" * 60)
print("PART 3: IMPROVED DOMAIN DETECTION (|P| > 0.8)")
print("=" * 60)

# Detect positive domains (P > 0.8)
pos_mask = P > 0.8
labeled_pos, n_pos_domains = ndimage.label(pos_mask)

# Detect negative domains (P < -0.8)
neg_mask = P < -0.8
labeled_neg, n_neg_domains = ndimage.label(neg_mask)

print(f"  Positive domains (P > 0.8): {n_pos_domains}")
print(f"  Negative domains (P < -0.8): {n_neg_domains}")

def analyze_domains(labeled, n_domains, label_type):
    """Analyze domain statistics"""
    volumes = []
    for i in range(1, n_domains + 1):
        vol = np.sum(labeled == i)
        volumes.append(vol)
    volumes = np.array(volumes)

    # Convert to physical units
    vol_mpc3 = volumes * (CELL_SIZE**3)
    diameters = 2 * (3 * vol_mpc3 / (4 * np.pi))**(1/3)

    # Filter single-cell domains
    valid = diameters[volumes > 1]

    if len(valid) > 0:
        stats_dict = {
            "type": label_type,
            "n_total": int(n_domains),
            "n_valid": int(len(valid)),
            "D_min": float(valid.min()),
            "D_p25": float(np.percentile(valid, 25)),
            "D_median": float(np.median(valid)),
            "D_p75": float(np.percentile(valid, 75)),
            "D_p90": float(np.percentile(valid, 90)),
            "D_max": float(valid.max()),
            "D_mean": float(np.mean(valid)),
            "D_std": float(np.std(valid))
        }
    else:
        stats_dict = {
            "type": label_type,
            "n_total": int(n_domains),
            "n_valid": 0
        }

    return diameters, stats_dict

diameters_pos, stats_pos = analyze_domains(labeled_pos, n_pos_domains, "positive (P>0.8)")
diameters_neg, stats_neg = analyze_domains(labeled_neg, n_neg_domains, "negative (P<-0.8)")

print(f"\n  Positive domains:")
print(f"    Valid (>1 cell): {stats_pos.get('n_valid', 0)}")
if stats_pos.get('n_valid', 0) > 0:
    print(f"    Median D: {stats_pos['D_median']:.2f} Mpc")
    print(f"    Max D: {stats_pos['D_max']:.2f} Mpc")

print(f"\n  Negative domains:")
print(f"    Valid (>1 cell): {stats_neg.get('n_valid', 0)}")
if stats_neg.get('n_valid', 0) > 0:
    print(f"    Median D: {stats_neg['D_median']:.2f} Mpc")
    print(f"    Max D: {stats_neg['D_max']:.2f} Mpc")

# Combined statistics
all_diameters = np.concatenate([diameters_pos[diameters_pos > CELL_SIZE*1.2],
                                 diameters_neg[diameters_neg > CELL_SIZE*1.2]])

domain_stats = {
    "threshold": 0.8,
    "positive": stats_pos,
    "negative": stats_neg,
    "combined": {
        "n_valid": len(all_diameters),
        "D_median": float(np.median(all_diameters)) if len(all_diameters) > 0 else 0,
        "D_mean": float(np.mean(all_diameters)) if len(all_diameters) > 0 else 0,
        "D_max": float(np.max(all_diameters)) if len(all_diameters) > 0 else 0
    },
    "L_J_mpc": float(L_J),
    "cell_size_mpc": float(CELL_SIZE)
}

with open(f"{OUTPUT_DIR}/domain_stats.json", 'w') as f:
    json.dump(domain_stats, f, indent=2)
print(f"\n  Saved: domain_stats.json")

# Domain size histogram
fig, ax = plt.subplots(figsize=(10, 6))
if len(all_diameters) > 0:
    ax.hist(all_diameters, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(L_J, color='r', linestyle='--', linewidth=2, label=f'L_J = {L_J:.2f} Mpc')
    ax.axvline(CELL_SIZE, color='g', linestyle=':', linewidth=2, label=f'Cell = {CELL_SIZE:.2f} Mpc')
    if len(all_diameters) > 0:
        ax.axvline(np.median(all_diameters), color='b', linestyle='--',
                   linewidth=2, label=f'Median = {np.median(all_diameters):.2f} Mpc')
ax.set_xlabel('Domain Diameter (Mpc)')
ax.set_ylabel('Probability Density')
ax.set_title(f'Domain Size Distribution (|P| > 0.8, N = {len(all_diameters)})')
ax.legend()
ax.set_xlim(0, min(50, np.percentile(all_diameters, 99) if len(all_diameters) > 0 else 50))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/domain_size_hist.png", dpi=150)
plt.close()
print(f"  Saved: domain_size_hist.png")

# ============================================================
# PART 4: POLARIZATION VS DENSITY RELATION
# ============================================================
print("\n" + "=" * 60)
print("PART 4: POLARIZATION VS DENSITY RELATION")
print("=" * 60)

# Density contrast
rho_mean = np.mean(rho)
delta = (rho - rho_mean) / rho_mean

P_flat = P.flatten()
delta_flat = delta.flatten()
absP_flat = np.abs(P_flat)

corr_P_delta = np.corrcoef(P_flat, delta_flat)[0, 1]
corr_absP_delta = np.corrcoef(absP_flat, delta_flat)[0, 1]

print(f"  corr(P, δ) = {corr_P_delta:.4f}")
print(f"  corr(|P|, δ) = {corr_absP_delta:.4f}")

# Mean |P| per density percentile
percentiles = np.arange(5, 100, 5)
delta_thresholds = np.percentile(delta_flat, percentiles)
mean_absP = []
mean_P = []

for i, pctl in enumerate(percentiles):
    if i == 0:
        mask = delta_flat <= delta_thresholds[i]
    else:
        mask = (delta_flat > delta_thresholds[i-1]) & (delta_flat <= delta_thresholds[i])

    if np.sum(mask) > 0:
        mean_absP.append(np.mean(absP_flat[mask]))
        mean_P.append(np.mean(P_flat[mask]))
    else:
        mean_absP.append(np.nan)
        mean_P.append(np.nan)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(percentiles, mean_absP, 'bo-', linewidth=2, markersize=6)
ax.set_xlabel('Density Percentile')
ax.set_ylabel('Mean |P|')
ax.set_title(f'Polarization Magnitude vs Density (corr = {corr_absP_delta:.3f})')
ax.grid(True, alpha=0.3)
ax.axhline(np.mean(absP_flat), color='r', linestyle='--', alpha=0.5, label='Overall mean |P|')
ax.legend()

ax = axes[1]
ax.plot(percentiles, mean_P, 'ro-', linewidth=2, markersize=6)
ax.set_xlabel('Density Percentile')
ax.set_ylabel('Mean P')
ax.set_title(f'Polarization Sign vs Density (corr = {corr_P_delta:.3f})')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linestyle='-', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_density_relation.png", dpi=150)
plt.close()
print(f"  Saved: polarization_density_relation.png")

# ============================================================
# PART 5: 3D VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("PART 5: 3D VISUALIZATION")
print("=" * 60)

try:
    import pyvista as pv
    pv.start_xvfb()  # For headless rendering

    # Smooth the polarization field
    from scipy.ndimage import gaussian_filter
    P_smooth = gaussian_filter(P, sigma=1.0)

    print("  Creating 3D mesh...")

    # Create a uniform grid
    grid = pv.ImageData(dimensions=(N_GRID+1, N_GRID+1, N_GRID+1),
                        spacing=(CELL_SIZE, CELL_SIZE, CELL_SIZE),
                        origin=(0, 0, 0))

    # Add polarization data (cell-centered to point-centered)
    # Pad the array to match point dimensions
    P_padded = np.pad(P_smooth, ((0,1), (0,1), (0,1)), mode='edge')
    grid.point_data["P"] = P_padded.flatten(order='F')

    print("  Extracting isosurfaces...")

    # Extract isosurfaces
    iso_pos = grid.contour([0.8], scalars="P")
    iso_neg = grid.contour([-0.8], scalars="P")

    print(f"  Positive surface: {iso_pos.n_cells} cells")
    print(f"  Negative surface: {iso_neg.n_cells} cells")

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    # Add surfaces with colors
    if iso_pos.n_cells > 0:
        plotter.add_mesh(iso_pos, color='red', opacity=0.6, label='P > 0.8 (+)')
    if iso_neg.n_cells > 0:
        plotter.add_mesh(iso_neg, color='blue', opacity=0.6, label='P < -0.8 (-)')

    # Add bounding box
    plotter.add_mesh(pv.Box(bounds=[0, L_BOX, 0, L_BOX, 0, L_BOX]),
                     style='wireframe', color='white', line_width=1)

    # Camera setup
    center = [L_BOX/2, L_BOX/2, L_BOX/2]
    plotter.camera.focal_point = center

    print("  Rendering rotation animation (120 frames)...")

    # Create rotation animation
    n_frames = 120
    frames_dir = f"{OUTPUT_DIR}/frames"
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(n_frames):
        angle = i * 360 / n_frames
        # Position camera on a circle around the center
        radius = L_BOX * 1.8
        cam_x = center[0] + radius * np.cos(np.radians(angle))
        cam_y = center[1] + radius * np.sin(np.radians(angle))
        cam_z = center[2] + radius * 0.5

        plotter.camera.position = [cam_x, cam_y, cam_z]
        plotter.camera.up = [0, 0, 1]

        frame_path = f"{frames_dir}/frame_{i:04d}.png"
        plotter.screenshot(frame_path)

        if (i + 1) % 20 == 0:
            print(f"    Frame {i+1}/{n_frames}")

    plotter.close()

    # Compile video with ffmpeg
    print("  Compiling video...")
    import subprocess
    video_path = f"{OUTPUT_DIR}/janus_polarization_rotation.mp4"
    cmd = [
        'ffmpeg', '-y', '-framerate', '30',
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(video_path):
        print(f"  Saved: janus_polarization_rotation.mp4")
        # Clean up frames
        import shutil
        shutil.rmtree(frames_dir)
    else:
        print(f"  Video creation failed: {result.stderr}")

except ImportError as e:
    print(f"  PyVista not available: {e}")
    print("  Skipping 3D visualization")
except Exception as e:
    print(f"  3D visualization error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
  GRADIENT ANALYSIS:
    mean(|∇P|) = {mean_grad:.4f} Mpc⁻¹
    median(|∇P|) = {median_grad:.4f} Mpc⁻¹

  CHARACTERISTIC SCALE:
    L_J = {L_J:.4f} Mpc
    L_J / cell = {L_J / CELL_SIZE:.2f}

  DOMAIN DETECTION (|P| > 0.8):
    N_domains = {len(all_diameters)}
    D_median = {np.median(all_diameters):.2f} Mpc (if available)

  DENSITY COUPLING:
    corr(P, δ) = {corr_P_delta:.4f}
    corr(|P|, δ) = {corr_absP_delta:.4f}
""")

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
