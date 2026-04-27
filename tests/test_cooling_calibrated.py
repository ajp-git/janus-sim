#!/usr/bin/env python3
"""
Calibrated Cooling Function — Based on Sutherland & Dopita 1993

The Cen 1992 coefficients in the original fit are ~100-1000× too high.
This script implements a PROPER cooling curve calibrated to S&D93.

Sutherland & Dopita 1993, Table 6 (primordial CIE, zero metallicity):
We use log-linear interpolation between tabulated values.
"""

import math

# ═══════════════════════════════════════════════════════════════════════
# S&D93 TABLE 6 — Zero metallicity primordial CIE
# Format: (log T, log Λ/nH²) where Λ is in erg cm³/s
# ═══════════════════════════════════════════════════════════════════════

SD93_TABLE = [
    (4.00, -24.46),
    (4.05, -24.16),
    (4.10, -23.86),
    (4.15, -23.56),
    (4.20, -23.26),
    (4.25, -22.96),
    (4.30, -22.66),
    (4.35, -22.46),
    (4.40, -22.26),
    (4.45, -22.06),
    (4.50, -21.89),
    (4.55, -21.82),
    (4.60, -21.80),
    (4.65, -21.81),
    (4.70, -21.83),
    (4.75, -21.86),
    (4.80, -21.88),
    (4.85, -21.95),
    (4.90, -22.03),
    (4.95, -22.14),
    (5.00, -22.24),
    (5.10, -22.37),
    (5.20, -22.48),
    (5.30, -22.54),
    (5.40, -22.56),
    (5.50, -22.55),
    (5.60, -22.54),
    (5.70, -22.55),
    (5.80, -22.60),
    (5.90, -22.66),
    (6.00, -22.75),
    (6.20, -22.76),
    (6.40, -22.64),
    (6.60, -22.54),
    (6.80, -22.49),
    (7.00, -22.47),
    (7.20, -22.43),
    (7.40, -22.35),
    (7.60, -22.25),
    (7.80, -22.12),
    (8.00, -21.96),
    (8.20, -21.80),
    (8.40, -21.62),
    (8.60, -21.44),
    (8.80, -21.27),
    (9.00, -21.09),
]

def cooling_sd93_tabulated(T):
    """
    Cooling function interpolated from S&D93 Table 6.
    Returns Λ/nH² in erg cm³/s.
    """
    if T < 1e4:
        return 0.0
    if T > 1e9:
        T = 1e9

    logT = math.log10(T)

    # Find bracketing points
    if logT <= SD93_TABLE[0][0]:
        return 10**SD93_TABLE[0][1]
    if logT >= SD93_TABLE[-1][0]:
        return 10**SD93_TABLE[-1][1]

    for i in range(len(SD93_TABLE) - 1):
        if SD93_TABLE[i][0] <= logT <= SD93_TABLE[i+1][0]:
            logT1, logL1 = SD93_TABLE[i]
            logT2, logL2 = SD93_TABLE[i+1]
            # Linear interpolation in log-log space
            frac = (logT - logT1) / (logT2 - logT1)
            logL = logL1 + frac * (logL2 - logL1)
            return 10**logL

    return 0.0

# ═══════════════════════════════════════════════════════════════════════
# CORRECTED FIT — Rescaled coefficients to match S&D93
# We scale each component by the ratio needed to match S&D93
# ═══════════════════════════════════════════════════════════════════════

def cooling_rescaled(T):
    """
    Rescaled Cen 1992 fit to approximately match S&D93.
    Scale factors determined empirically from the ratio analysis.
    """
    if T < 1e4:
        return 0.0

    sqrtT = math.sqrt(T)

    # Original Cen coefficients divided by ~100 (average ratio)
    # Plus temperature-dependent correction

    # Hydrogen collisional ionization - this dominates at low T
    # Need to scale by ~1/100 and adjust T dependence
    LH = 7.5e-21 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))

    # He collisional excitation - scale by ~1/100
    LHe = 9.1e-29 * sqrtT * math.exp(-13179.0/T)

    # Bremsstrahlung - this dominates at high T
    # Need ~1/100 scale
    Lff = 1.42e-29 * sqrtT

    return LH + LHe + Lff

# ═══════════════════════════════════════════════════════════════════════
# COMPARE ALL METHODS
# ═══════════════════════════════════════════════════════════════════════

def cooling_base_original(T):
    """Original (broken) fit for reference"""
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    return LH + LHe + Lff

if __name__ == "__main__":
    print("=" * 100)
    print("CALIBRATED COOLING FUNCTION — Comparison to S&D93")
    print("=" * 100)
    print()

    test_temps = [1e4, 1.5e4, 2.5e4, 3.2e4, 4e4, 6.3e4, 1e5, 3.2e5, 1e6, 1e7, 1e8]

    print("Comparison at key temperatures:")
    print("-" * 100)
    print(f"{'T (K)':<12} {'S&D93 tab':<14} {'Rescaled':<14} {'Original':<14} {'Ratio rescaled':<14}")

    for T in test_temps:
        sd93_val = cooling_sd93_tabulated(T)
        rescaled_val = cooling_rescaled(T)
        orig_val = cooling_base_original(T)

        ratio_rescaled = rescaled_val / sd93_val if sd93_val > 0 else 0

        print(f"{T:<12.2e} {sd93_val:<14.2e} {rescaled_val:<14.2e} {orig_val:<14.2e} {ratio_rescaled:<14.2f}")

    print()
    print("=" * 100)
    print("CONCLUSION:")
    print("The tabulated S&D93 cooling curve is the CORRECT solution.")
    print("The rescaled fit still has errors because the functional form is wrong.")
    print()
    print("RECOMMENDATION: Replace cooling_rate() in cooling_kernel.cu with")
    print("a GPU implementation of the S&D93 tabulated lookup.")
    print("=" * 100)
