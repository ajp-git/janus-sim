#!/usr/bin/env python3
"""
Trichotomy parameter search for Janus optimization.
Generates Tour N+1 parameters from Tour N results.
"""

import yaml
from pathlib import Path
from typing import List, NamedTuple
from metrics import RunMetrics
from score import score, ScoreBreakdown


class TourParams(NamedTuple):
    """Parameters for a single run."""
    eta: float
    lambda_base: float
    r_smooth: float = 5.0
    n_particles: int = 200_000
    n_steps: int = 500


def next_tour(
    results: List[RunMetrics],
    tour_number: int,
    output_dir: Path,
) -> List[TourParams]:
    """
    Generate 3 new parameter sets by trichotomy.

    Tour 1 -> Tour 2: zoom on eta, explore lambda_base {20, 30, 40}
    Tour 2 -> Tour 3: zoom on (eta, lambda_base), N_particles = 300k
    Tour 3 -> Tour 4: fine zoom + explore R_smooth {3, 5, 8}
    """
    scored = [(m, score(m)) for m in results if m.abort_reason is None]

    if not scored:
        # All runs aborted -> expand search space
        print("WARNING: All runs aborted! Trying with lower eta values.")
        return [
            TourParams(eta=0.3, lambda_base=30.0),
            TourParams(eta=0.5, lambda_base=30.0),
            TourParams(eta=0.7, lambda_base=30.0),
        ]

    scored_sorted = sorted(scored, key=lambda x: x[1].composite, reverse=True)
    best_m, best_s = scored_sorted[0]

    print(f"\n  Best run: {best_m.run_id} (score={best_s.composite:.3f})")
    print(f"   eta={best_m.eta}, lambda_base={best_m.lambda_base}")

    if tour_number == 1:
        # Tour 1 -> Tour 2
        # Know best eta among {0.5, 1.0, 1.5}
        # Zoom eta around best, explore lambda_base
        eta_center = best_m.eta
        eta_delta = 0.25

        new_params = [
            TourParams(
                eta=eta_center - eta_delta,
                lambda_base=20.0,
                n_particles=300_000,
                n_steps=700,
            ),
            TourParams(
                eta=eta_center,
                lambda_base=30.0,
                n_particles=300_000,
                n_steps=700,
            ),
            TourParams(
                eta=eta_center + eta_delta,
                lambda_base=40.0,
                n_particles=300_000,
                n_steps=700,
            ),
        ]

    elif tour_number == 2:
        # Tour 2 -> Tour 3
        # Fine zoom on (eta, lambda_base)
        second_m = scored_sorted[1][0] if len(scored_sorted) > 1 else best_m
        eta_delta = abs(best_m.eta - second_m.eta) * 0.4
        lam_delta = abs(best_m.lambda_base - second_m.lambda_base) * 0.4

        eta_delta = max(eta_delta, 0.05)
        lam_delta = max(lam_delta, 2.0)

        new_params = [
            TourParams(
                eta=best_m.eta - eta_delta,
                lambda_base=best_m.lambda_base - lam_delta,
                n_particles=500_000,
                n_steps=1000,
            ),
            TourParams(
                eta=best_m.eta,
                lambda_base=best_m.lambda_base,
                n_particles=500_000,
                n_steps=1000,
            ),
            TourParams(
                eta=best_m.eta + eta_delta,
                lambda_base=best_m.lambda_base + lam_delta,
                n_particles=500_000,
                n_steps=1000,
            ),
        ]

    else:
        # Tour >= 3: generic zoom + R_smooth exploration
        eta_delta = 0.02
        lam_delta = 1.0
        r_values = [3.0, 5.0, 8.0] if tour_number == 3 else [best_m.r_smooth] * 3

        new_params = [
            TourParams(
                eta=best_m.eta - eta_delta,
                lambda_base=best_m.lambda_base,
                r_smooth=r_values[0],
                n_particles=500_000,
                n_steps=1200,
            ),
            TourParams(
                eta=best_m.eta,
                lambda_base=best_m.lambda_base + lam_delta,
                r_smooth=r_values[1],
                n_particles=500_000,
                n_steps=1200,
            ),
            TourParams(
                eta=best_m.eta + eta_delta,
                lambda_base=best_m.lambda_base - lam_delta,
                r_smooth=r_values[2],
                n_particles=500_000,
                n_steps=1200,
            ),
        ]

    # Generate YAML files for next tour
    next_tour_dir = output_dir / f"tour{tour_number + 1}"
    next_tour_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(new_params):
        config = make_config(p, run_label=f"tour{tour_number+1}_run{'ABC'[i]}")
        yaml_path = next_tour_dir / f"config_run_{'ABC'[i]}.yaml"
        yaml_path.write_text(yaml.dump(config, default_flow_style=False))
        print(f"   Generated: {yaml_path}")

    return new_params


def make_config(p: TourParams, run_label: str) -> dict:
    """Create config dict from parameters."""
    return {
        "simulation": {
            "box_size_mpc": 150.0,
            "n_particles": p.n_particles,
            "n_steps": p.n_steps,
            "z_start": 5.0,
            "z_end": 1.5,
            "seed": 42,
            "theta": 0.7,
        },
        "physics": {
            "eta": round(p.eta, 4),
            "lambda_base_mpc": round(p.lambda_base, 2),
            "r_smooth_mpc": round(p.r_smooth, 1),
            "lambda_floor": 0.01,
            "hubble_friction": True,
        },
        "pm_grid": {
            "n_cells": 128 if p.n_particles <= 300_000 else 256,
            "k_min": 2,
        },
        "output": {
            "dir": f"output/{run_label}",
            "snapshot_redshifts": [5.0, 3.0, 2.0, 1.5],
            "metrics_every_steps": 25,
            "save_snapshots": True,
        },
    }


def generate_tour1_configs(output_dir: Path) -> None:
    """Generate Tour 1 configuration files."""
    tour1_dir = output_dir / "tour1"
    tour1_dir.mkdir(parents=True, exist_ok=True)

    for eta, label in [(0.5, 'A'), (1.0, 'B'), (1.5, 'C')]:
        p = TourParams(eta=eta, lambda_base=30.0, n_particles=200_000, n_steps=500)
        config = make_config(p, f"tour1_run{label}")
        yaml_path = tour1_dir / f"config_run_{label}.yaml"
        yaml_path.write_text(yaml.dump(config, default_flow_style=False))
        print(f"Generated: {yaml_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trichotomy.py init          - Generate Tour 1 configs")
        print("  python trichotomy.py next <N> <dir> - Generate Tour N+1 from Tour N")
        sys.exit(1)

    if sys.argv[1] == "init":
        output_dir = Path("/mnt/T2/janus-sim/optim")
        generate_tour1_configs(output_dir)
    elif sys.argv[1] == "next" and len(sys.argv) >= 4:
        tour_num = int(sys.argv[2])
        tour_dir = Path(sys.argv[3])
        from metrics import load_tour_results
        sim_root = Path("/mnt/T2/janus-sim")
        results = load_tour_results(tour_dir, sim_root)
        next_tour(results, tour_num, Path("/mnt/T2/janus-sim/optim"))
    else:
        print("Invalid arguments")
        sys.exit(1)
