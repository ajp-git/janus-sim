#!/usr/bin/env python3
"""
Analysis pipeline for Nuit 3 runs.
Includes improved filament detection and velocity field analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import struct
import json
import sys

# Add optim to path for filament_metrics_v2
sys.path.insert(0, '/mnt/T2/janus-sim/optim')
from filament_metrics_v2 import (
    load_snapshot, fof_halos, detect_interhalos_filaments,
    compute_pi_prime, analyze_velocity_field_interhalo
)

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from collections import defaultdict


def step_to_z(step, n_steps, z_start=5.0, z_end=0.0):
    """Convert step number to redshift."""
    a_start = 1.0 / (1.0 + z_start)
    a_end = 1.0 / (1.0 + z_end)
    a = a_start + (a_end - a_start) * step / n_steps
    return 1.0 / a - 1.0


def compute_dcom(pos, signs, box_size):
    """Compute ΔCOM between m+ and m-."""
    mask_plus = signs > 0
    mask_minus = signs < 0
    if mask_plus.sum() == 0 or mask_minus.sum() == 0:
        return 0.0, np.zeros(3)
    com_plus = pos[mask_plus].mean(axis=0)
    com_minus = pos[mask_minus].mean(axis=0)
    dcom_vec = com_plus - com_minus
    dcom_vec = dcom_vec - box_size * np.round(dcom_vec / box_size)
    return np.linalg.norm(dcom_vec), dcom_vec


def compute_pair_correlation(pos1, pos2, r_bins, box_size):
    """Compute pair correlation function g(r) - FAST version."""
    n1, n2 = len(pos1), len(pos2)

    if n1 == 0 or n2 == 0:
        return np.zeros(len(r_bins) - 1)

    max_particles = 10000
    if n1 > max_particles:
        idx1 = np.random.choice(n1, max_particles, replace=False)
        pos1 = pos1[idx1]
        n1 = max_particles
    if n2 > max_particles:
        idx2 = np.random.choice(n2, max_particles, replace=False)
        pos2 = pos2[idx2]
        n2 = max_particles

    pos1_shifted = pos1 + box_size / 2
    pos2_shifted = pos2 + box_size / 2

    tree1 = cKDTree(pos1_shifted, boxsize=box_size)
    tree2 = cKDTree(pos2_shifted, boxsize=box_size)

    r_max = r_bins[-1]
    pairs = tree1.sparse_distance_matrix(tree2, r_max, output_type='ndarray')

    if len(pairs) > 0:
        distances = pairs['v']
        distances = distances[distances > 0]
        counts, _ = np.histogram(distances, bins=r_bins)
    else:
        counts = np.zeros(len(r_bins) - 1)

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    dr = np.diff(r_bins)
    shell_volumes = 4 * np.pi * r_centers**2 * dr

    number_density = n2 / (box_size**3)
    expected = n1 * number_density * shell_volumes

    g_r = np.where(expected > 0, counts / expected, 0)
    return g_r


def nfw_profile(r, rho_s, r_s):
    """NFW density profile."""
    x = r / r_s
    return rho_s / (x * (1 + x)**2)


def analyze_run(run_dir, box_size, run_name, out_dir):
    """Complete analysis for one run."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {run_name}")
    print(f"{'='*70}")

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find snapshots
    snaps = sorted(run_dir.glob("snapshots/snap_*.bin"))
    if not snaps:
        print("  No snapshots found!")
        return None

    # Load z=0 snapshot
    snap_z0 = snaps[-1]
    print(f"  Loading {snap_z0.name}...")
    pos, signs, vel = load_snapshot(snap_z0)
    n = len(pos)

    print(f"  N = {n} particles")
    print(f"  Velocities: {'yes' if vel is not None else 'no'}")

    metrics = {
        'run_name': run_name,
        'n_particles': n,
        'box_size_mpc': box_size,
        'has_velocities': vel is not None
    }

    # =========================================================================
    # 2a. Density map z=0
    # =========================================================================
    print("\n  2a. Density map...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slice_mask = np.abs(pos[:, 2]) < 10
    pos_slice = pos[slice_mask]
    signs_slice = signs[slice_mask]

    mask_plus = signs_slice > 0
    mask_minus = signs_slice < 0

    # Scatter
    ax = axes[0]
    ax.scatter(pos_slice[mask_minus, 0], pos_slice[mask_minus, 1],
               s=0.1, c='blue', alpha=0.3, label='m-')
    ax.scatter(pos_slice[mask_plus, 0], pos_slice[mask_plus, 1],
               s=0.1, c='red', alpha=0.3, label='m+')
    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('|z| < 10 Mpc slice')
    ax.legend(markerscale=10)

    # Density m+
    ax = axes[1]
    extent = box_size / 2
    h_plus, xedges, yedges = np.histogram2d(
        pos_slice[mask_plus, 0], pos_slice[mask_plus, 1],
        bins=128, range=[[-extent, extent], [-extent, extent]]
    )
    ax.imshow(h_plus.T, origin='lower', extent=[-extent, extent, -extent, extent],
              cmap='Reds', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m+ density')
    ax.set_aspect('equal')

    # Density m-
    ax = axes[2]
    h_minus, _, _ = np.histogram2d(
        pos_slice[mask_minus, 0], pos_slice[mask_minus, 1],
        bins=128, range=[[-extent, extent], [-extent, extent]]
    )
    ax.imshow(h_minus.T, origin='lower', extent=[-extent, extent, -extent, extent],
              cmap='Blues', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m- density')
    ax.set_aspect('equal')

    # Compute metrics
    halos = fof_halos(pos, signs, box_size, b=0.2, min_particles=100)
    n_halos_plus = sum(1 for h in halos if h['sign'] == 1)
    n_halos_minus = sum(1 for h in halos if h['sign'] == -1)
    dcom_mag, _ = compute_dcom(pos, signs, box_size)

    # Segregation from time_series
    ts_file = run_dir / "time_series.csv"
    if ts_file.exists():
        with open(ts_file) as f:
            lines = f.readlines()
            last_line = lines[-1].strip().split(',')
            seg_final = float(last_line[5])
    else:
        seg_final = 0.0

    metrics['n_halos_plus'] = n_halos_plus
    metrics['n_halos_minus'] = n_halos_minus
    metrics['dcom_mpc'] = dcom_mag
    metrics['segregation'] = seg_final

    plt.suptitle(f'{run_name} z=0\nH+={n_halos_plus} H-={n_halos_minus} '
                 f'ΔCOM={dcom_mag:.1f} Mpc S={seg_final:.3f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "density_z0.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: density_z0.png")

    # =========================================================================
    # 2b. Temporal evolution
    # =========================================================================
    print("\n  2b. Temporal evolution...")
    n_snaps = min(8, len(snaps))
    snap_indices = np.linspace(0, len(snaps)-1, n_snaps, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    evolution_data = []

    for idx, snap_idx in enumerate(snap_indices[:8]):
        snap_path = snaps[snap_idx]
        step = int(snap_path.stem.split('_')[1])

        p, s, _ = load_snapshot(snap_path)
        z = step_to_z(step, 2000)

        h = fof_halos(p, s, box_size, b=0.2, min_particles=50)
        nh_plus = sum(1 for hh in h if hh['sign'] == 1)
        nh_minus = sum(1 for hh in h if hh['sign'] == -1)
        dc, _ = compute_dcom(p, s, box_size)

        evolution_data.append({
            'step': step, 'z': z,
            'n_halos_plus': nh_plus, 'n_halos_minus': nh_minus,
            'dcom': dc
        })

        ax = axes[idx]
        slice_m = np.abs(p[:, 2]) < 10
        ps = p[slice_m]
        ss = s[slice_m]

        ax.scatter(ps[ss < 0, 0], ps[ss < 0, 1], s=0.1, c='blue', alpha=0.3)
        ax.scatter(ps[ss > 0, 0], ps[ss > 0, 1], s=0.1, c='red', alpha=0.3)
        ax.set_xlim(-box_size/2, box_size/2)
        ax.set_ylim(-box_size/2, box_size/2)
        ax.set_aspect('equal')
        ax.set_title(f'z={z:.2f}\nH+={nh_plus} ΔCOM={dc:.1f}', fontsize=9)

        # Mark z=2 activation
        if 1.8 < z < 2.2:
            ax.set_facecolor('#ffffcc')

    metrics['evolution'] = evolution_data

    plt.suptitle(f'{run_name}: Temporal Evolution (yellow = z~2 activation)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "evolution_temporelle.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: evolution_temporelle.png")

    # =========================================================================
    # 2c. Inter-halo filaments
    # =========================================================================
    print("\n  2c. Inter-halo filament detection...")
    filament_result = detect_interhalos_filaments(pos, signs, box_size)

    metrics['filaments'] = {
        'n_filaments_real': filament_result['n_filaments_real'],
        'length_mean_real': filament_result['length_mean_real'],
        'length_max_real': filament_result['length_max_real']
    }

    print(f"    n_filaments_real: {filament_result['n_filaments_real']}")
    print(f"    length_max_real: {filament_result['length_max_real']:.1f} Mpc")

    # Visualize skeleton
    fig, ax = plt.subplots(figsize=(10, 10))

    if filament_result['filament_cells'] is not None:
        cells = filament_result['filament_cells']
        cell_size = box_size / 64
        cell_centers = cells * cell_size - box_size / 2 + cell_size / 2

        ax.scatter(cell_centers[:, 0], cell_centers[:, 1],
                   c=cell_centers[:, 2], cmap='viridis', s=20, alpha=0.7)

    # Mark halos
    for hpos in filament_result.get('halo_positions', []):
        ax.scatter([hpos[0]], [hpos[1]], marker='*', s=300,
                   c='red', edgecolor='black', zorder=10)

    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title(f'{run_name}: Inter-halo skeleton\n'
                 f'n_filaments={filament_result["n_filaments_real"]}, '
                 f'max_length={filament_result["length_max_real"]:.1f} Mpc')

    plt.savefig(out_dir / "skeleton_interhalos.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: skeleton_interhalos.png")

    # =========================================================================
    # 2d. Pair correlations g(r)
    # =========================================================================
    print("\n  2d. Pair correlations...")
    r_max = box_size / 2
    r_bins = np.arange(1, r_max + 2, 2)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    mask_plus = signs > 0
    mask_minus = signs < 0
    pos_plus = pos[mask_plus]
    pos_minus = pos[mask_minus]

    g_pp = compute_pair_correlation(pos_plus, pos_plus, r_bins, box_size)
    g_mm = compute_pair_correlation(pos_minus, pos_minus, r_bins, box_size)
    g_pm = compute_pair_correlation(pos_plus, pos_minus, r_bins, box_size)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_centers, g_pp, 'r-', linewidth=2, label='g++(r)')
    ax.plot(r_centers, g_mm, 'b-', linewidth=2, label='g--(r)')
    ax.plot(r_centers, g_pm, 'purple', linewidth=2, label='g+-(r)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    # Find secondary peak in g++
    if len(g_pp) > 10:
        # Look for peaks after r=10 Mpc
        r_mask = r_centers > 10
        if r_mask.sum() > 0:
            g_pp_far = g_pp[r_mask]
            r_far = r_centers[r_mask]
            peak_idx = np.argmax(g_pp_far)
            if g_pp_far[peak_idx] > 1.5:
                ax.axvline(r_far[peak_idx], color='red', linestyle=':',
                          label=f'g++ peak @ {r_far[peak_idx]:.0f} Mpc')
                metrics['g_pp_secondary_peak'] = {
                    'r': float(r_far[peak_idx]),
                    'value': float(g_pp_far[peak_idx])
                }

    # g+- minimum
    min_idx = np.argmin(g_pm)
    metrics['g_pm_min'] = {'r': float(r_centers[min_idx]), 'value': float(g_pm[min_idx])}

    ax.set_xlabel('r (Mpc)')
    ax.set_ylabel('g(r)')
    ax.set_title(f'{run_name}: Pair correlations\ng+-(min)={g_pm[min_idx]:.3f} @ r={r_centers[min_idx]:.0f} Mpc')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    plt.savefig(out_dir / "correlation_gr.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: correlation_gr.png")

    # =========================================================================
    # 2e. Velocity field (if available)
    # =========================================================================
    print("\n  2e. Velocity field...")
    if vel is not None:
        vel_result = analyze_velocity_field_interhalo(pos, vel, signs, box_size, halos)
        metrics['velocity_field'] = {
            'status': vel_result['status'],
            'coherent_flow_fraction': vel_result.get('coherent_flow_fraction', None),
            'coherent_toward_halos': vel_result.get('coherent_toward_halos', None)
        }

        if vel_result['status'] == 'completed':
            print(f"    Coherent flow fraction: {vel_result['coherent_flow_fraction']:.3f}")

            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Velocity magnitude histogram
            ax = axes[0]
            v_mag = np.linalg.norm(vel, axis=1)
            v_plus = v_mag[mask_plus]
            v_minus = v_mag[mask_minus]

            bins = np.linspace(0, np.percentile(v_mag, 99), 50)
            ax.hist(v_plus, bins=bins, alpha=0.6, color='red', label='m+', density=True)
            ax.hist(v_minus, bins=bins, alpha=0.6, color='blue', label='m-', density=True)
            ax.set_xlabel('|v| (Mpc/Gyr)')
            ax.set_ylabel('PDF')
            ax.set_title('Velocity distribution')
            ax.legend()

            # Velocity field map
            ax = axes[1]
            vx = vel_result.get('vx_grid')
            vy = vel_result.get('vy_grid')
            count = vel_result.get('count_grid')

            if vx is not None:
                # Project XY plane (sum over z)
                vx_2d = np.nanmean(vx, axis=2)
                vy_2d = np.nanmean(vy, axis=2)
                count_2d = count.sum(axis=2)

                n_cells = vx_2d.shape[0]
                cell_size = box_size / n_cells
                x = np.arange(n_cells) * cell_size - box_size / 2 + cell_size / 2
                X, Y = np.meshgrid(x, x)

                mask = count_2d > 0
                vx_2d[~mask] = 0
                vy_2d[~mask] = 0

                v_mag_2d = np.sqrt(vx_2d**2 + vy_2d**2)
                ax.quiver(X, Y, vx_2d.T, vy_2d.T, v_mag_2d.T, cmap='viridis')

                # Mark halos
                halos_plus = [h for h in halos if h['sign'] == 1]
                for h in halos_plus:
                    ax.scatter([h['com'][0]], [h['com'][1]], marker='*',
                              s=200, c='red', edgecolor='black', zorder=10)

            ax.set_xlim(-box_size/2, box_size/2)
            ax.set_ylim(-box_size/2, box_size/2)
            ax.set_aspect('equal')
            ax.set_xlabel('x (Mpc)')
            ax.set_ylabel('y (Mpc)')
            ax.set_title(f'Inter-halo velocity field\nCoherence: {vel_result["coherent_flow_fraction"]:.2f}')

            plt.suptitle(f'{run_name}: Velocity Field Analysis')
            plt.tight_layout()
            plt.savefig(out_dir / "velocity_field.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: velocity_field.png")
        else:
            print(f"    Status: {vel_result['status']}")
    else:
        metrics['velocity_field'] = {'status': 'no_velocities'}
        print("    No velocities available")

    # =========================================================================
    # 2f. NFW profiles
    # =========================================================================
    print("\n  2f. NFW profiles...")
    halos_plus = [h for h in halos if h['sign'] == 1 and h['n_particles'] > 50000]

    if len(halos_plus) > 0:
        fig, axes = plt.subplots(1, min(4, len(halos_plus)),
                                  figsize=(5*min(4, len(halos_plus)), 5))
        if len(halos_plus) == 1:
            axes = [axes]

        nfw_results = []

        for idx, halo in enumerate(halos_plus[:4]):
            ax = axes[idx]
            com = halo['com']
            halo_pos = halo['positions']

            dr = halo_pos - com
            r = np.linalg.norm(dr, axis=1)

            r_bins = np.logspace(np.log10(0.5), np.log10(30), 20)
            r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])

            counts, _ = np.histogram(r, bins=r_bins)
            shell_volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
            rho = counts / shell_volumes

            valid = rho > 0
            if valid.sum() >= 3:
                r_valid = r_centers[valid]
                rho_valid = rho[valid]

                try:
                    popt, _ = curve_fit(nfw_profile, r_valid, rho_valid,
                                        p0=[rho_valid.max(), 2.0],
                                        bounds=([0, 0.1], [1e10, 50]),
                                        maxfev=5000)
                    rho_s, r_s = popt

                    r_fit = np.logspace(np.log10(0.5), np.log10(25), 100)
                    rho_fit = nfw_profile(r_fit, rho_s, r_s)

                    ax.loglog(r_valid, rho_valid, 'ro', markersize=8, label='Data')
                    ax.loglog(r_fit, rho_fit, 'b-', linewidth=2,
                             label=f'NFW r_s={r_s:.1f}')

                    nfw_results.append({
                        'n_particles': halo['n_particles'],
                        'r_s': float(r_s),
                        'rho_s': float(rho_s)
                    })

                except Exception:
                    ax.loglog(r_valid, rho_valid, 'ro', markersize=8)
                    nfw_results.append({'n_particles': halo['n_particles'], 'fit': 'failed'})

            ax.set_xlabel('r (Mpc)')
            ax.set_ylabel('ρ')
            ax.set_title(f'Halo {idx}: N={halo["n_particles"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        metrics['nfw'] = nfw_results

        plt.suptitle(f'{run_name}: NFW profiles')
        plt.tight_layout()
        plt.savefig(out_dir / "nfw_profiles.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: nfw_profiles.png")
    else:
        metrics['nfw'] = []
        print("    No large halos for NFW fit")

    # =========================================================================
    # Compute Pi'
    # =========================================================================
    if vel is not None:
        sigma_v = np.std(np.linalg.norm(vel[mask_plus], axis=1))
        rho_mean = n / (box_size**3) * 3e17 / n  # Rough M_sun/Mpc^3
        # Get eta and lambda from config (hardcoded for now)
        # TODO: read from config
        eta = 0.50 if 'eta05' in run_name else 0.88
        lambda_base = 8.0

        pi_prime = compute_pi_prime(eta, lambda_base, sigma_v, rho_mean)
        metrics['pi_prime'] = pi_prime
        print(f"\n  Π' = {pi_prime:.4f}")

    # Save metrics
    with open(out_dir / "metrics_summary.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved: metrics_summary.json")

    return metrics


def main():
    runs = [
        ("/mnt/T2/janus-sim/output/nuit3/P3_eta05_lambda8_Z1", 150.0, "P3_eta05_lambda8_Z1"),
        ("/mnt/T2/janus-sim/output/nuit3/P2_eta088_lambda8_Z1", 150.0, "P2_eta088_lambda8_Z1"),
        ("/mnt/T2/janus-sim/output/nuit3/P1_eta088_lambda8_300mpc", 300.0, "P1_eta088_lambda8_300mpc"),
    ]

    all_metrics = {}

    for run_dir, box_size, run_name in runs:
        out_dir = f"/mnt/T2/janus-sim/output/nuit3/{run_name}/figures"
        if Path(run_dir).exists() and list(Path(run_dir).glob("snapshots/*.bin")):
            metrics = analyze_run(run_dir, box_size, run_name, out_dir)
            if metrics:
                all_metrics[run_name] = metrics
        else:
            print(f"\n{run_name}: Not ready yet")

    # Decision
    print("\n" + "="*70)
    print("DECISION FINALE")
    print("="*70)

    filaments_found = False
    best_run = None
    best_length = 0

    for name, m in all_metrics.items():
        fil = m.get('filaments', {})
        n_fil = fil.get('n_filaments_real', 0)
        max_len = fil.get('length_max_real', 0)

        if n_fil > 0 and max_len > 10:
            filaments_found = True
            if max_len > best_length:
                best_length = max_len
                best_run = name

        print(f"\n{name}:")
        print(f"  n_filaments_real = {n_fil}")
        print(f"  length_max_real = {max_len:.1f} Mpc")
        print(f"  g+-(min) = {m.get('g_pm_min', {}).get('value', 'N/A')}")

    # Write decision
    decision_file = Path("/mnt/T2/janus-sim/output/nuit3/decision_finale.txt")
    with open(decision_file, 'w') as f:
        if filaments_found:
            f.write(f"FILAMENTS CONFIRMÉS : OUI\n")
            f.write(f"RUN GAGNANT : {best_run}\n")
            f.write(f"LONGUEUR MAX : {best_length:.1f} Mpc\n")
            f.write(f"PROCHAINE ÉTAPE : trichotomie autour de {best_run}\n")
        else:
            f.write("FILAMENTS CONFIRMÉS : NON\n")
            f.write("RUN GAGNANT : aucun\n")
            f.write("PROCHAINE ÉTAPE : préparer preprint\n")
            f.write("\nConclusion : Le modèle Janus avec répulsion Yukawa isotrope\n")
            f.write("ne produit pas de filaments cosmiques dans ce régime.\n")
            f.write("Résultat publiable : ségrégation dipolaire sans web cosmique.\n")

    print(f"\n\nDecision written to: {decision_file}")

    if filaments_found:
        print(f"\n*** FILAMENTS CONFIRMED in {best_run} ***")
        print(f"*** Max length: {best_length:.1f} Mpc ***")
    else:
        print("\n*** NO FILAMENTS DETECTED ***")
        print("*** Prepare preprint with dipole segregation results ***")


if __name__ == '__main__':
    main()
