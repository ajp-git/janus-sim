#!/usr/bin/env python3
"""
Cooling Rate Precision Test — Compare 4 fits to Sutherland & Dopita 1993

References:
  - Sutherland & Dopita 1993, ApJS 88, 253
  - Primordial CIE (H + He only, no metals)

Decision criteria:
  - erreur_max < 5%  : EXCELLENT
  - erreur_max < 10% : ACCEPTABLE
  - erreur_max > 10% : REJETER
"""

import math

# ═══════════════════════════════════════════════════════════════════════
# COOLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def cooling_base(T):
    """Version de base — fit original sans correction"""
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    return LH + LHe + Lff

def cooling_chatgpt(T):
    """Version ChatGPT — correction additive gaussienne sur Lambda_H
    Centre T=25000K, amplitude +12%, sigma=6000K"""
    if T < 1e4:
        return 0.0
    sqrtT = math.sqrt(T)
    LH  = 7.5e-19 * math.exp(-118348.0/T) / (1.0 + math.sqrt(T/1e5))
    LHe = 9.1e-27 * sqrtT * math.exp(-13179.0/T)
    Lff = 1.42e-27 * sqrtT
    corr = LH * 0.12 * math.exp(-((T - 25000.0)/6000.0)**2)
    return LH + LHe + Lff + corr

def cooling_gemini(T):
    """Version Gemini — correction multiplicative globale
    Centre logT=4.18 (T~15100K), amplitude +62%, sigma=0.075 en log"""
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
    """Version Mistral — correction additive asymétrique
    Centre logT=4.146 (T~14000K), sigma_low=0.12, sigma_high=0.28"""
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
# REFERENCE TABLE — Sutherland & Dopita 1993
# Primordial CIE (H + He only)
# ═══════════════════════════════════════════════════════════════════════

refs = [
    (1.0e4, 1.5e-25),   # Hydrogène neutre
    (1.5e4, 1.6e-22),   # Montée pic Ly-alpha
    (3.0e4, 3.5e-22),   # PIC Ly-alpha
    (1.0e5, 2.5e-22),   # Hélium + recombinaison
    (1.0e6, 2.5e-22),   # Plateau He
    (1.0e7, 1.5e-22),   # Transition Bremsstrahlung
    (1.0e8, 4.0e-22),   # Bremsstrahlung dominant
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
    print("=" * 100)
    print("COOLING RATE PRECISION TEST — Comparison to Sutherland & Dopita 1993")
    print("=" * 100)
    print()

    # Header
    header = f"{'T (K)':<12} {'Λ_ref':<14}"
    for n in fits:
        header += f" {n:<20}"
    print(header)
    print("-" * 100)

    errors = {n: [] for n in fits}

    for T, ref in refs:
        row = f"{T:<12.2e} {ref:<14.2e}"
        for name, fn in fits.items():
            val = fn(T)
            err = abs(val - ref) / ref * 100 if ref > 0 else 0
            errors[name].append(err)
            row += f" {val:.2e} ({err:5.1f}%)"
        print(row)

    print()
    print("=" * 100)
    print("ERREUR MAXIMALE PAR FIT")
    print("=" * 100)

    best_fit = None
    best_err = float('inf')

    for name, errs in errors.items():
        max_err = max(errs)
        avg_err = sum(errs) / len(errs)

        if max_err < 5:
            status = "EXCELLENT"
        elif max_err < 10:
            status = "ACCEPTABLE"
        else:
            status = "REJETER"

        print(f"  {name:<12} : err_max = {max_err:6.1f}%  err_moy = {avg_err:5.1f}%  -> {status}")

        if max_err < best_err:
            best_err = max_err
            best_fit = name

    print()
    print("=" * 100)
    print("DÉCISION")
    print("=" * 100)

    if best_err < 10:
        print(f"  FIT RETENU : {best_fit} (erreur max = {best_err:.1f}%)")
        if best_err < 5:
            print("  STATUS : EXCELLENT")
        else:
            print("  STATUS : ACCEPTABLE")
    else:
        print(f"  AUCUN FIT ACCEPTABLE (meilleur : {best_fit} avec erreur max = {best_err:.1f}%)")
        print("  -> Demander à ChatGPT une version améliorée avec les valeurs de référence exactes")

    print()

    # Detailed breakdown by temperature zone
    print("=" * 100)
    print("ERREURS PAR ZONE DE TEMPÉRATURE")
    print("=" * 100)
    zones = [
        "Hydrogène neutre (10^4 K)",
        "Montée Ly-alpha (1.5×10^4 K)",
        "PIC Ly-alpha (3×10^4 K)",
        "He + recombinaison (10^5 K)",
        "Plateau He (10^6 K)",
        "Transition Bremss. (10^7 K)",
        "Bremsstrahlung (10^8 K)",
    ]

    for i, zone in enumerate(zones):
        row = f"  {zone:<35}"
        for name in fits:
            row += f"  {name}: {errors[name][i]:5.1f}%"
        print(row)

    print()
    print("=" * 100)
