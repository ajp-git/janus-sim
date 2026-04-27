#!/usr/bin/env python3
"""
Tour orchestration for Janus optimization.
Launches sequential runs, analyzes results, decides next tour.
"""

import subprocess
import sys
import yaml
import time
from pathlib import Path
from metrics import RunMetrics, load_tour_results
from score import score, print_scoreboard
from trichotomy import next_tour

# Binary path (adjust if needed)
BINARY = Path("/mnt/T2/janus-sim/target/release/janus_optim")
SIM_ROOT = Path("/mnt/T2/janus-sim")


def run_simulation(config_path: Path) -> bool:
    """Launch a single run. Returns True if successful."""
    config = yaml.safe_load(config_path.read_text())
    output_dir = SIM_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"

    print(f"\n  Launching: {config_path.name}")
    print(f"   eta={config['physics']['eta']}, lambda_base={config['physics']['lambda_base_mpc']}")
    print(f"   N={config['simulation']['n_particles']:,}, steps={config['simulation']['n_steps']}")
    print(f"   -> {output_dir}")

    t0 = time.time()

    with open(log_path, "w") as log:
        # Run with Docker compose
        cmd = [
            "docker", "compose", "run", "--rm", "dev",
            "cargo", "run", "--release", "--features", "cuda,cufft",
            "--bin", "janus_optim", "--",
            "--config", str(config_path.relative_to(SIM_ROOT))
        ]

        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(SIM_ROOT),
        )

    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"   {status} in {elapsed/60:.1f} min (code={result.returncode})")

    return result.returncode == 0


def analyze_tour(tour_dir: Path, tour_number: int) -> list:
    """Load and analyze all runs from a tour."""
    return load_tour_results(tour_dir, SIM_ROOT)


def main(tour_number: int, config_dir: Path):
    optim_dir = SIM_ROOT / "optim"

    # 1. Launch the 3 runs
    configs = sorted(config_dir.glob("config_run*.yaml"))
    if not configs:
        print(f"ERROR: No configs found in {config_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  TOUR {tour_number} - {len(configs)} runs")
    print(f"{'='*60}")

    for cfg in configs:
        run_simulation(cfg)

    # 2. Analyze
    print(f"\n  Analyzing Tour {tour_number}...")
    results = analyze_tour(config_dir, tour_number)

    if not results:
        print("ERROR: No exploitable results")
        sys.exit(1)

    scored = [(m, score(m)) for m in results]
    print_scoreboard(scored)

    # 3. Global convergence criterion
    best_score = max(s.composite for _, s in scored)

    if best_score > 0.80:
        print(f"\n  CONVERGENCE REACHED - score={best_score:.3f} > 0.80")
        print("-> Proceed to 1M particle validation run")
        sys.exit(0)

    if tour_number >= 6:
        print(f"\n  Tour {tour_number}: 6-tour limit reached, stopping")
        sys.exit(0)

    # 4. Generate Tour N+1
    print(f"\n  Generating Tour {tour_number + 1}...")
    next_tour(results, tour_number, optim_dir)

    print(f"\n  To launch Tour {tour_number + 1}:")
    print(f"   python run_tour.py {tour_number + 1} {optim_dir}/tour{tour_number+1}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_tour.py <tour_number> <config_dir>")
        print("\nExample:")
        print("  python run_tour.py 1 tour1/")
        sys.exit(1)

    main(int(sys.argv[1]), Path(sys.argv[2]))
