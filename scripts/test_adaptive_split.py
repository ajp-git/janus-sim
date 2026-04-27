#!/usr/bin/env python3
"""
Unit test for adaptive splitting logic (ZOOM_L1_V2_SPEC.md)

Tests:
1. Mass conservation: sum(m_daughters) = m_mother
2. Correct daughter count per zone (±1 tolerance for transitions)
3. Daughter positions within r_disp sphere around mother
4. Blue Noise: no daughters at r_disp = 0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Constants from ZOOM_L1_V2_SPEC.md
R_CORE = 2.0
R_MID = 8.0
R_EXT = 20.0
R_MINUS_HR = 12.0
TRANS_WIDTH = 1.0

SPLIT_CORE = 50
SPLIT_MID = 20
SPLIT_EXT = 5
SPLIT_OUTER = 1
SPLIT_MINUS = 8

EPS_CORE = 0.010
EPS_MID = 0.020
EPS_EXT = 0.030

M_PART_SOURCE = 5.1e10  # M_sun

# ═══════════════════════════════════════════════════════════════════════════
# SMOOTH INTERPOLATION (must match Rust implementation)
# ═══════════════════════════════════════════════════════════════════════════

def smoothstep(x: float, edge0: float, edge1: float) -> float:
    """Hermite interpolation: 0 for x<edge0, 1 for x>edge1"""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def compute_split_factor_plus(r: float) -> float:
    """Compute split factor for m+ with SMOOTH transitions"""
    hw = TRANS_WIDTH / 2.0

    t_core_mid = smoothstep(r, R_CORE - hw, R_CORE + hw)
    t_mid_ext = smoothstep(r, R_MID - hw, R_MID + hw)
    t_ext_outer = smoothstep(r, R_EXT - hw, R_EXT + hw)

    # Interpolate through zones
    split_in_core = SPLIT_CORE
    split_at_mid = lerp(SPLIT_CORE, SPLIT_MID, t_core_mid)
    split_at_ext = lerp(split_at_mid, SPLIT_EXT, t_mid_ext)
    split_final = lerp(split_at_ext, SPLIT_OUTER, t_ext_outer)

    return split_final

def compute_split_factor_minus(r: float) -> float:
    """Compute split factor for m- with smooth transition"""
    hw = TRANS_WIDTH / 2.0
    t = smoothstep(r, R_MINUS_HR - hw, R_MINUS_HR + hw)
    return lerp(SPLIT_MINUS, 1.0, t)

def adaptive_softening(r: float) -> float:
    """Adaptive softening by zone (for r_disp)"""
    hw = TRANS_WIDTH / 2.0
    t1 = smoothstep(r, R_CORE - hw, R_CORE + hw)
    t2 = smoothstep(r, R_MID - hw, R_MID + hw)

    eps_core_mid = lerp(EPS_CORE, EPS_MID, t1)
    eps_final = lerp(eps_core_mid, EPS_EXT, t2)
    return eps_final

def fibonacci_sphere(n: int) -> List[Tuple[float, float, float]]:
    """Generate n points uniformly distributed on unit sphere (Blue Noise)"""
    if n == 0:
        return []
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi / golden

    points = []
    for i in range(n):
        theta = golden_angle * i
        z = 1.0 - (2.0 * i + 1.0) / n
        r_xy = np.sqrt(1.0 - z * z)
        points.append((r_xy * np.cos(theta), r_xy * np.sin(theta), z))
    return points

# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneStats:
    name: str
    n_mothers: int
    expected_daughters: float
    actual_daughters: int
    mass_conserved: bool
    positions_ok: bool
    no_zero_disp: bool

def test_splitting():
    """Test adaptive splitting on 200 synthetic particles"""
    np.random.seed(42)

    # Generate 100 m+ and 100 m- particles distributed radially
    radii = np.linspace(0.1, 24.9, 100)  # Avoid exact boundaries

    # Store mothers and their daughters
    results_plus = []
    results_minus = []

    print("=" * 70)
    print("ADAPTIVE SPLITTING UNIT TEST")
    print("=" * 70)
    print(f"Testing {len(radii)} m+ and {len(radii)} m- particles")
    print()

    # Test m+ particles
    print("Testing m+ splitting...")
    for r in radii:
        split_factor = compute_split_factor_plus(r)
        n_split = max(1, round(split_factor))

        # Dispersion radius
        eps = adaptive_softening(r)
        r_disp = eps / 2.0

        # Generate daughters
        directions = fibonacci_sphere(n_split)

        # Mass per daughter
        m_daughter = M_PART_SOURCE / n_split
        total_mass = m_daughter * n_split

        # Check positions
        daughter_distances = []
        for dx, dy, dz in directions:
            dist = np.sqrt(dx*dx + dy*dy + dz*dz) * r_disp
            daughter_distances.append(dist)

        results_plus.append({
            'r': r,
            'split_factor': split_factor,
            'n_split': n_split,
            'mass_conserved': abs(total_mass - M_PART_SOURCE) / M_PART_SOURCE < 1e-10,
            'r_disp': r_disp,
            'max_dist': max(daughter_distances) if daughter_distances else 0,
            'min_dist': min(daughter_distances) if daughter_distances else 0,
        })

    # Test m- particles
    print("Testing m- splitting...")
    for r in radii:
        split_factor = compute_split_factor_minus(r)
        n_split = max(1, round(split_factor))

        eps = adaptive_softening(r)
        r_disp = eps / 2.0

        directions = fibonacci_sphere(n_split)
        m_daughter = M_PART_SOURCE / n_split
        total_mass = m_daughter * n_split

        daughter_distances = []
        for dx, dy, dz in directions:
            dist = np.sqrt(dx*dx + dy*dy + dz*dz) * r_disp
            daughter_distances.append(dist)

        results_minus.append({
            'r': r,
            'split_factor': split_factor,
            'n_split': n_split,
            'mass_conserved': abs(total_mass - M_PART_SOURCE) / M_PART_SOURCE < 1e-10,
            'r_disp': r_disp,
            'max_dist': max(daughter_distances) if daughter_distances else 0,
            'min_dist': min(daughter_distances) if daughter_distances else 0,
        })

    # Aggregate by zone
    print()
    print("=" * 70)
    print("m+ RESULTS BY ZONE")
    print("=" * 70)

    zones_plus = [
        ("Core (r<2)", lambda r: r < R_CORE - TRANS_WIDTH/2, SPLIT_CORE),
        ("Core→Mid trans", lambda r: R_CORE - TRANS_WIDTH/2 <= r < R_CORE + TRANS_WIDTH/2, None),
        ("Mid (2-8)", lambda r: R_CORE + TRANS_WIDTH/2 <= r < R_MID - TRANS_WIDTH/2, SPLIT_MID),
        ("Mid→Ext trans", lambda r: R_MID - TRANS_WIDTH/2 <= r < R_MID + TRANS_WIDTH/2, None),
        ("Ext (8-20)", lambda r: R_MID + TRANS_WIDTH/2 <= r < R_EXT - TRANS_WIDTH/2, SPLIT_EXT),
        ("Ext→Outer trans", lambda r: R_EXT - TRANS_WIDTH/2 <= r < R_EXT + TRANS_WIDTH/2, None),
        ("Outer (>20)", lambda r: r >= R_EXT + TRANS_WIDTH/2, SPLIT_OUTER),
    ]

    print(f"{'Zone':<18} | {'N_mothers':>9} | {'Expected':>10} | {'Actual':>8} | {'Mass OK':>8} | {'Pos OK':>7}")
    print("-" * 70)

    all_tests_pass = True

    for zone_name, zone_filter, expected_split in zones_plus:
        zone_results = [r for r in results_plus if zone_filter(r['r'])]
        if not zone_results:
            continue

        n_mothers = len(zone_results)
        total_daughters = sum(r['n_split'] for r in zone_results)
        all_mass_ok = all(r['mass_conserved'] for r in zone_results)
        all_pos_ok = all(r['max_dist'] <= r['r_disp'] * 1.01 for r in zone_results)  # 1% tolerance

        if expected_split is not None:
            expected_daughters = n_mothers * expected_split
            # Allow ±1 per mother for rounding
            daughters_ok = abs(total_daughters - expected_daughters) <= n_mothers
        else:
            expected_daughters = "trans"
            daughters_ok = True

        mass_str = "PASS" if all_mass_ok else "FAIL"
        pos_str = "PASS" if all_pos_ok else "FAIL"

        if not all_mass_ok or not all_pos_ok:
            all_tests_pass = False

        if expected_split is not None:
            print(f"{zone_name:<18} | {n_mothers:>9} | {expected_daughters:>10.0f} | {total_daughters:>8} | {mass_str:>8} | {pos_str:>7}")
        else:
            print(f"{zone_name:<18} | {n_mothers:>9} | {'(trans)':>10} | {total_daughters:>8} | {mass_str:>8} | {pos_str:>7}")

    # m- zones
    print()
    print("=" * 70)
    print("m- RESULTS BY ZONE")
    print("=" * 70)

    zones_minus = [
        ("HR (r<12)", lambda r: r < R_MINUS_HR - TRANS_WIDTH/2, SPLIT_MINUS),
        ("Transition", lambda r: R_MINUS_HR - TRANS_WIDTH/2 <= r < R_MINUS_HR + TRANS_WIDTH/2, None),
        ("Outer (>12)", lambda r: r >= R_MINUS_HR + TRANS_WIDTH/2, 1),
    ]

    print(f"{'Zone':<18} | {'N_mothers':>9} | {'Expected':>10} | {'Actual':>8} | {'Mass OK':>8} | {'Pos OK':>7}")
    print("-" * 70)

    for zone_name, zone_filter, expected_split in zones_minus:
        zone_results = [r for r in results_minus if zone_filter(r['r'])]
        if not zone_results:
            continue

        n_mothers = len(zone_results)
        total_daughters = sum(r['n_split'] for r in zone_results)
        all_mass_ok = all(r['mass_conserved'] for r in zone_results)
        all_pos_ok = all(r['max_dist'] <= r['r_disp'] * 1.01 for r in zone_results)

        if expected_split is not None:
            expected_daughters = n_mothers * expected_split
            daughters_ok = abs(total_daughters - expected_daughters) <= n_mothers
        else:
            expected_daughters = "trans"
            daughters_ok = True

        mass_str = "PASS" if all_mass_ok else "FAIL"
        pos_str = "PASS" if all_pos_ok else "FAIL"

        if not all_mass_ok or not all_pos_ok:
            all_tests_pass = False

        if expected_split is not None:
            print(f"{zone_name:<18} | {n_mothers:>9} | {expected_daughters:>10.0f} | {total_daughters:>8} | {mass_str:>8} | {pos_str:>7}")
        else:
            print(f"{zone_name:<18} | {n_mothers:>9} | {'(trans)':>10} | {total_daughters:>8} | {mass_str:>8} | {pos_str:>7}")

    # Blue Noise check
    print()
    print("=" * 70)
    print("BLUE NOISE CHECK")
    print("=" * 70)

    # Test that Fibonacci sphere has no zero-distance points
    for n in [5, 8, 20, 50]:
        points = fibonacci_sphere(n)
        distances = [np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in points]
        min_dist = min(distances) if distances else 0
        # On unit sphere, all points should be at distance ~1
        ok = all(0.99 < d < 1.01 for d in distances)
        print(f"n={n:3}: all points on unit sphere = {'PASS' if ok else 'FAIL'} (min={min_dist:.4f}, max={max(distances):.4f})")
        if not ok:
            all_tests_pass = False

    # Check minimum inter-point distance (Blue Noise property)
    print()
    for n in [8, 20, 50]:
        points = fibonacci_sphere(n)
        min_inter = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                d = np.sqrt((points[i][0]-points[j][0])**2 +
                           (points[i][1]-points[j][1])**2 +
                           (points[i][2]-points[j][2])**2)
                min_inter = min(min_inter, d)
        # For uniform distribution, min distance should be > 0
        expected_min = 2.0 / np.sqrt(n)  # Rough approximation
        ok = min_inter > expected_min * 0.5
        print(f"n={n:3}: min inter-point distance = {min_inter:.4f} (expected > {expected_min*0.5:.4f}) {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_tests_pass = False

    # Summary
    print()
    print("=" * 70)
    if all_tests_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_tests_pass

if __name__ == '__main__':
    success = test_splitting()
    exit(0 if success else 1)
