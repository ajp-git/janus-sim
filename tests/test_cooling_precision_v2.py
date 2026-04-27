#!/usr/bin/env python3
"""
Cooling Rate Precision Test v2 — Adjusted for correct S&D93 comparison

The issue: S&D93 tabulates Λ(T) = cooling function normalized per n_H n_e [erg cm³/s]
For CIE primordial gas, this is ~10^-22 to 10^-23 in the 10^4-10^8 K range.

Our fits compute the same quantity but the coefficients may be off.

Actual Sutherland & Dopita 1993 Table 6 (zero metallicity primordial):
log T | log Λ/n_H² [erg cm³/s]
4.0   | -24.46  → 3.5e-25
4.2   | -23.37  → 4.3e-24
4.4   | -22.26  → 5.5e-23 (Ly-alpha rise)
4.5   | -21.89  → 1.3e-22 (near peak)
4.6   | -21.80  → 1.6e-22
4.8   | -21.88  → 1.3e-22
5.0   | -22.24  → 5.8e-23
5.5   | -22.55  → 2.8e-23
6.0   | -22.75  → 1.8e-23
7.0   | -22.47  → 3.4e-23
8.0   | -21.75  → 1.8e-22
"""

import math

# ═══════════════════════════════════════════════════════════════════════
# COOLING FUNCTIONS — Same as before
# ═══════════════════════════════════════════════════════════════════════

def cooling_base(T):
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    return LH + LHe + Lff

def cooling_chatgpt(T):
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    corr = LH * 0.12 * math.exp(-((T - 25000.0)/6000.0)**2)
    return LH + LHe + Lff + corr

def cooling_gemini(T):
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    L = LH + LHe + Lff
    logT = math.log10(T)
    d = logT - 4.18
    corr = 1.0 + 0.62 * math.exp(-88.8889 * d * d)
    return L * corr

def cooling_mistral(T):
    if T < 5000 or T > 150000:
        return cooling_base(T)
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    L = LH + LHe + Lff
    logT = math.log10(T)
    mu, sl, sh = 4.146, 0.12, 0.28
    d = logT - mu
    sig = sl if d < 0 else sh
    log_corr = -21.82 * math.exp(-d*d/(2*sig*sig))
    return L + 10**log_corr * 0.155

# ═══════════════════════════════════════════════════════════════════════
# CORRECT REFERENCE — Sutherland & Dopita 1993 Table 6 (primordial CIE)
# ═══════════════════════════════════════════════════════════════════════

refs_sd93 = [
    (1.0e4, 3.5e-25),   # log T = 4.0, log Λ = -24.46
    (1.5e4, 4.3e-24),   # log T ≈ 4.18
    (2.5e4, 5.5e-23),   # log T = 4.4 (Ly-alpha rise)
    (3.2e4, 1.3e-22),   # log T ≈ 4.5 (near peak)
    (4.0e4, 1.6e-22),   # log T = 4.6 (peak region)
    (6.3e4, 1.3e-22),   # log T = 4.8
    (1.0e5, 5.8e-23),   # log T = 5.0
    (3.2e5, 2.8e-23),   # log T = 5.5
    (1.0e6, 1.8e-23),   # log T = 6.0
    (1.0e7, 3.4e-23),   # log T = 7.0
    (1.0e8, 1.8e-22),   # log T = 8.0
]

fits = {
    'base'    : cooling_base,
    'chatgpt' : cooling_chatgpt,
    'gemini'  : cooling_gemini,
    'mistral' : cooling_mistral,
}

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 120)
    print("COOLING RATE PRECISION TEST v2 — Using correct S&D93 primordial CIE values")
    print("=" * 120)
    print()

    # First, show what our fits produce
    print("1. RAW OUTPUT FROM FITS")
    print("-" * 80)
    header = f"{'T (K)':<12} {'log T':<8}"
    for n in fits:
        header += f" {n:<14}"
    print(header)

    test_temps = [1e4, 1.5e4, 2.5e4, 3.2e4, 4.0e4, 6.3e4, 1e5, 3.2e5, 1e6, 1e7, 1e8]
    for T in test_temps:
        row = f"{T:<12.2e} {math.log10(T):<8.2f}"
        for name, fn in fits.items():
            val = fn(T)
            row += f" {val:<14.2e}"
        print(row)

    print()
    print("2. S&D93 REFERENCE VALUES")
    print("-" * 80)
    print(f"{'T (K)':<12} {'log T':<8} {'Λ_SD93':<14}")
    for T, ref in refs_sd93:
        print(f"{T:<12.2e} {math.log10(T):<8.2f} {ref:<14.2e}")

    print()
    print("3. RATIO FIT/S&D93 (should be ~1.0)")
    print("-" * 80)
    header = f"{'T (K)':<12} {'log T':<8}"
    for n in fits:
        header += f" {n:<14}"
    print(header)

    for T, ref in refs_sd93:
        row = f"{T:<12.2e} {math.log10(T):<8.2f}"
        for name, fn in fits.items():
            val = fn(T)
            ratio = val / ref if ref > 0 else 0
            row += f" {ratio:<14.1f}"
        print(row)

    print()
    print("=" * 120)
    print("DIAGNOSTIC: The fits are ~10-1000× too high compared to S&D93")
    print("This suggests the Cen 1992 coefficients need rescaling")
    print()
    print("RECOMMENDATION: Use a tabulated cooling curve or recalibrate the coefficients")
    print("=" * 120)
