#!/usr/bin/env python3
"""
Render summary images for exploration grid cases A-F.
Generates density maps at step 0 and step 2000 for each case.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct

def read_render_data(filepath):
    """Read binary render data file."""
    with open(filepath, 'rb') as f:
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        redshift = struct.unpack('<d', f.read(8))[0]
        n_particles = struct.unpack('<I', f.read(4))[0]

        # Read positions (f32)
        pos_data = np.frombuffer(f.read(n_particles * 3 * 4), dtype=np.float32)
        positions = pos_data.reshape(n_particles, 3)

        # Read signs (i8)
        signs = np.frombuffer(f.read(n_particles), dtype=np.int8)

    return {
        'step': step,
        'box_size': box_size,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'redshift': redshift,
        'positions': positions,
        'signs': signs
    }

def compute_density_grid(positions, signs, box_size, grid_size=64):
    """Compute 2D projected density for + and - particles."""
    half_box = box_size / 2

    # Separate populations
    pos_mask = signs > 0
    neg_mask = signs < 0

    pos_plus = positions[pos_mask]
    pos_minus = positions[neg_mask]

    # Project to XY plane
    bins = np.linspace(-half_box, half_box, grid_size + 1)

    rho_plus, _, _ = np.histogram2d(pos_plus[:, 0], pos_plus[:, 1], bins=[bins, bins])
    rho_minus, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 1], bins=[bins, bins])

    return rho_plus, rho_minus

def render_case_comparison(case_letter, output_dir, grid_output_dir):
    """Render step 0 vs step 2000 comparison for a case."""
    case_dir = grid_output_dir / f"grid_{case_letter}_100k" / "render_data"

    if not case_dir.exists():
        print(f"  Case {case_letter}: directory not found")
        return None

    step0_file = case_dir / "step_000000.bin"
    step2000_file = case_dir / "step_002000.bin"

    if not step0_file.exists() or not step2000_file.exists():
        print(f"  Case {case_letter}: missing render files")
        return None

    data0 = read_render_data(step0_file)
    data2000 = read_render_data(step2000_file)

    # Compute density grids
    rho_plus_0, rho_minus_0 = compute_density_grid(data0['positions'], data0['signs'], data0['box_size'])
    rho_plus_f, rho_minus_f = compute_density_grid(data2000['positions'], data2000['signs'], data2000['box_size'])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Step 0
    vmax = max(rho_plus_0.max(), rho_minus_0.max())
    axes[0, 0].imshow(np.log1p(rho_plus_0.T), origin='lower', cmap='Blues', vmin=0, vmax=np.log1p(vmax))
    axes[0, 0].set_title(f'Step 0 | z={data0["redshift"]:.1f} | + particles')
    axes[0, 1].imshow(np.log1p(rho_minus_0.T), origin='lower', cmap='Reds', vmin=0, vmax=np.log1p(vmax))
    axes[0, 1].set_title(f'Step 0 | Seg={data0["seg"]:.4f} | - particles')

    # Step 2000
    vmax = max(rho_plus_f.max(), rho_minus_f.max())
    axes[1, 0].imshow(np.log1p(rho_plus_f.T), origin='lower', cmap='Blues', vmin=0, vmax=np.log1p(vmax))
    axes[1, 0].set_title(f'Step 2000 | z={data2000["redshift"]:.2f} | + particles')
    axes[1, 1].imshow(np.log1p(rho_minus_f.T), origin='lower', cmap='Reds', vmin=0, vmax=np.log1p(vmax))
    axes[1, 1].set_title(f'Step 2000 | Seg={data2000["seg"]:.4f} | - particles')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Case {case_letter} | KE/KE0={data2000["ke_ratio"]:.2f}', fontsize=14)
    plt.tight_layout()

    outfile = output_dir / f"case_{case_letter}_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Case {case_letter}: saved {outfile.name}")
    return True

def render_mosaic(output_dir, grid_output_dir):
    """Render 6x2 mosaic of all cases."""
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))

    cases = ['A', 'B', 'C', 'D', 'E', 'F']
    ic_types = ['uniform', 'density 0.3x', 'density 1.0x', 'density 2.0x', '±psi 0.3x', '±psi 1.0x']

    for row, (case, ic_type) in enumerate(zip(cases, ic_types)):
        case_dir = grid_output_dir / f"grid_{case}_100k" / "render_data"

        if not case_dir.exists():
            continue

        step0_file = case_dir / "step_000000.bin"
        step2000_file = case_dir / "step_002000.bin"

        if not step0_file.exists() or not step2000_file.exists():
            continue

        data0 = read_render_data(step0_file)
        data2000 = read_render_data(step2000_file)

        rho_plus_0, rho_minus_0 = compute_density_grid(data0['positions'], data0['signs'], data0['box_size'])
        rho_plus_f, rho_minus_f = compute_density_grid(data2000['positions'], data2000['signs'], data2000['box_size'])

        # Row: [+step0, -step0, +step2000, -step2000]
        vmax0 = max(rho_plus_0.max(), rho_minus_0.max())
        vmaxf = max(rho_plus_f.max(), rho_minus_f.max())

        axes[row, 0].imshow(np.log1p(rho_plus_0.T), origin='lower', cmap='Blues', vmin=0, vmax=np.log1p(vmax0))
        axes[row, 0].set_ylabel(f'{case}: {ic_type}', fontsize=10)

        axes[row, 1].imshow(np.log1p(rho_minus_0.T), origin='lower', cmap='Reds', vmin=0, vmax=np.log1p(vmax0))

        axes[row, 2].imshow(np.log1p(rho_plus_f.T), origin='lower', cmap='Blues', vmin=0, vmax=np.log1p(vmaxf))

        axes[row, 3].imshow(np.log1p(rho_minus_f.T), origin='lower', cmap='Reds', vmin=0, vmax=np.log1p(vmaxf))
        axes[row, 3].text(1.05, 0.5, f'Seg={data2000["seg"]:.3f}\nKE={data2000["ke_ratio"]:.2f}',
                          transform=axes[row, 3].transAxes, fontsize=9, va='center')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title('+ (z=5)', fontsize=11)
    axes[0, 1].set_title('- (z=5)', fontsize=11)
    axes[0, 2].set_title('+ (z=0)', fontsize=11)
    axes[0, 3].set_title('- (z=0)', fontsize=11)

    fig.suptitle('Exploration Grid: 100K particles | z=5 to z=0 | 2000 steps', fontsize=14)
    plt.tight_layout()

    outfile = output_dir / "grid_mosaic.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Mosaic saved: {outfile}")

if __name__ == "__main__":
    output_dir = Path("/mnt/T2/janus-sim/output/grid_summary")
    grid_output_dir = Path("/mnt/T2/janus-sim/output")

    output_dir.mkdir(exist_ok=True)

    print("Rendering grid exploration summary...")

    # Individual comparisons
    for case in ['A', 'B', 'C', 'D', 'E', 'F']:
        render_case_comparison(case, output_dir, grid_output_dir)

    # Mosaic
    render_mosaic(output_dir, grid_output_dir)

    print("\nDone!")
