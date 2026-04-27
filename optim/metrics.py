#!/usr/bin/env python3
"""
Metrics loading and analysis for Janus optimization.
FROZEN - do not modify between trichotomy tours.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import yaml


@dataclass
class RunMetrics:
    """Metrics from a single simulation run."""
    run_id: str
    eta: float
    lambda_base: float
    r_smooth: float

    # Final metrics (last snapshot)
    s_segregation: float
    filament_mean_mpc: float
    filament_max_mpc: float
    void_fraction: float
    void_mode_mpc: float
    pk_slope: float
    pk_excess_lcdm: float
    fil_matter_fraction: float

    # Trajectory metrics
    s_at_z3: Optional[float]
    s_at_z2: Optional[float]
    steps_completed: int
    abort_reason: Optional[str]

    @classmethod
    def from_jsonl(cls, run_dir: Path, config: dict) -> "RunMetrics":
        """Load metrics from JSONL file."""
        metrics_file = run_dir / "metrics.jsonl"
        if not metrics_file.exists():
            raise FileNotFoundError(f"No metrics in {run_dir}")

        lines = [json.loads(l) for l in metrics_file.read_text().splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"Empty metrics file: {metrics_file}")

        # Last step
        last = lines[-1]

        # Find intermediate metrics
        def find_at_z(target_z: float, tol: float = 0.3) -> Optional[dict]:
            return next(
                (m for m in lines if abs(m.get("redshift", 99) - target_z) < tol),
                None
            )

        at_z3 = find_at_z(3.0)
        at_z2 = find_at_z(2.0)

        # Read abort reason from log
        log_file = run_dir / "run.log"
        abort_reason = None
        if log_file.exists():
            log = log_file.read_text()
            for line in log.splitlines():
                if "ABORT" in line or "Abort" in line:
                    abort_reason = line.split(":", 1)[-1].strip()
                    break

        return cls(
            run_id=run_dir.name,
            eta=config["physics"]["eta"],
            lambda_base=config["physics"].get("lambda_base_mpc", 0.0),
            r_smooth=config["physics"].get("r_smooth_mpc", 5.0),
            s_segregation=last.get("s_segregation", 0.0),
            filament_mean_mpc=last.get("filament_mean_mpc", 0.0),
            filament_max_mpc=last.get("filament_max_mpc", 0.0),
            void_fraction=last.get("void_fraction", 1.0),
            void_mode_mpc=last.get("void_mode_mpc", 0.0),
            pk_slope=last.get("pk_slope", 0.0),
            pk_excess_lcdm=last.get("pk_excess_lcdm", 0.0),
            fil_matter_fraction=last.get("fil_matter_fraction", 0.0),
            s_at_z3=at_z3["s_segregation"] if at_z3 else None,
            s_at_z2=at_z2["s_segregation"] if at_z2 else None,
            steps_completed=last.get("step", 0),
            abort_reason=abort_reason,
        )

    @classmethod
    def from_time_series(cls, run_dir: Path, config: dict) -> "RunMetrics":
        """Load basic metrics from time_series.csv (fallback)."""
        ts_file = run_dir / "time_series.csv"
        if not ts_file.exists():
            raise FileNotFoundError(f"No time_series.csv in {run_dir}")

        import csv
        with open(ts_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty time series: {ts_file}")

        last = rows[-1]

        return cls(
            run_id=run_dir.name,
            eta=config["physics"]["eta"],
            lambda_base=config["physics"].get("lambda_base_mpc", 0.0),
            r_smooth=config["physics"].get("r_smooth_mpc", 5.0),
            s_segregation=float(last.get("segregation", 0.0)),
            filament_mean_mpc=0.0,
            filament_max_mpc=0.0,
            void_fraction=0.0,
            void_mode_mpc=0.0,
            pk_slope=0.0,
            pk_excess_lcdm=0.0,
            fil_matter_fraction=0.0,
            s_at_z3=None,
            s_at_z2=None,
            steps_completed=int(last.get("step", 0)),
            abort_reason=None,
        )


def load_tour_results(tour_dir: Path, sim_root: Path) -> List[RunMetrics]:
    """Load all results from a tour directory."""
    results = []

    for config_path in sorted(tour_dir.glob("config_run*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        run_output = sim_root / config["output"]["dir"]

        try:
            # Try JSONL first
            m = RunMetrics.from_jsonl(run_output, config)
            results.append(m)
        except FileNotFoundError:
            try:
                # Fallback to time_series.csv
                m = RunMetrics.from_time_series(run_output, config)
                results.append(m)
            except Exception as e:
                print(f"Warning: Could not load {run_output}: {e}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metrics.py <tour_dir>")
        sys.exit(1)

    tour_dir = Path(sys.argv[1])
    sim_root = Path("/mnt/T2/janus-sim")

    results = load_tour_results(tour_dir, sim_root)
    for m in results:
        print(f"{m.run_id}: eta={m.eta}, S={m.s_segregation:.3f}, steps={m.steps_completed}")
