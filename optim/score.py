#!/usr/bin/env python3
"""
Composite score calculation for Janus optimization.
ATTENTION: Only modify weights AFTER seeing Tour 1 results.
"""

from dataclasses import dataclass
from typing import List, Tuple
from metrics import RunMetrics


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown."""
    s1_segregation: float   # [0,1]
    s2_filaments: float     # [0,1]
    s3_fil_matter: float    # [0,1]
    s4_voids: float         # [0,1]
    composite: float        # weighted sum


# Weight configuration - FROZEN after Tour 1 calibration
WEIGHTS = {
    "segregation": 0.35,
    "filaments":   0.30,
    "fil_matter":  0.20,
    "voids":       0.15,
}


def score(m: RunMetrics) -> ScoreBreakdown:
    """
    Compute composite score in [0, 1]. 1 = perfect cosmic structure.

    Formula (identical to metrics.rs - DO NOT desynchronize):
      score = 0.35 * min(S(z=0) / 0.5, 1)
            + 0.30 * min(filament_mean_mpc / 10, 1)
            + 0.20 * min(fil_matter_fraction / 0.15, 1)
            + 0.15 * (1 if void_fraction < 0.70, decreasing otherwise)
    """

    # S1: Segregation - target S > 0.5
    s1 = min(m.s_segregation / 0.5, 1.0)

    # S2: Filaments - target mean length > 10 Mpc
    s2 = min(m.filament_mean_mpc / 10.0, 1.0) if m.filament_mean_mpc > 0 else 0.0

    # S3: Matter in filaments - target > 15% (DESI/Euclid ~ 18-25%)
    s3 = min(m.fil_matter_fraction / 0.15, 1.0) if m.fil_matter_fraction > 0 else 0.0

    # S4: Voids - target void_fraction < 0.70 (not a ghost universe)
    if m.void_fraction < 0.70:
        s4 = 1.0
    elif m.void_fraction < 0.95:
        s4 = max(0.0, 1.0 - (m.void_fraction - 0.70) / 0.25)
    else:
        s4 = 0.0

    composite = (
        WEIGHTS["segregation"] * s1
        + WEIGHTS["filaments"] * s2
        + WEIGHTS["fil_matter"] * s3
        + WEIGHTS["voids"] * s4
    )

    return ScoreBreakdown(s1, s2, s3, s4, composite)


def score_simple(s_segregation: float) -> float:
    """
    Simple score based on segregation only.
    Used when full metrics are not available.
    """
    return min(s_segregation / 0.5, 1.0) * WEIGHTS["segregation"]


def print_scoreboard(results: List[Tuple[RunMetrics, ScoreBreakdown]]) -> None:
    """Print score table for a tour."""
    print(f"\n{'Run':<20} {'eta':>6} {'lambda':>8} {'S_seg':>6} {'Fil':>6} "
          f"{'FilMat':>6} {'Void':>6} {'SCORE':>7}")
    print("-" * 80)

    for m, s in sorted(results, key=lambda x: x[1].composite, reverse=True):
        if s.composite > 0.5:
            status = "WINNER"
        elif s.composite > 0.2:
            status = "~"
        else:
            status = "X"

        abort = f" [{m.abort_reason[:25]}]" if m.abort_reason else ""

        print(f"{m.run_id:<20} {m.eta:>6.2f} {m.lambda_base:>8.1f} "
              f"{s.s1_segregation:>6.2f} {s.s2_filaments:>6.2f} "
              f"{s.s3_fil_matter:>6.2f} {s.s4_voids:>6.2f} "
              f"{s.composite:>7.3f} {status}{abort}")


if __name__ == "__main__":
    # Test with dummy data
    m = RunMetrics(
        run_id="test",
        eta=1.0,
        lambda_base=30.0,
        r_smooth=5.0,
        s_segregation=0.5,
        filament_mean_mpc=10.0,
        filament_max_mpc=20.0,
        void_fraction=0.5,
        void_mode_mpc=25.0,
        pk_slope=-2.8,
        pk_excess_lcdm=1.0,
        fil_matter_fraction=0.15,
        s_at_z3=0.3,
        s_at_z2=0.4,
        steps_completed=500,
        abort_reason=None,
    )

    s = score(m)
    print(f"Test score: {s}")
    print(f"Composite: {s.composite:.3f} (should be 1.0)")
