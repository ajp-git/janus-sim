#!/usr/bin/env python3
"""
Debug the Janus Friedmann equations.

Test the corrected equations from Petit & D'Agostini 2014:
  ä = -1.5 * E / a²
  ā̈ = +1.5 * E / ā²

where E = Ω₊ - Ω₋ is the conserved energy.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("DEBUG: Janus Friedmann Equations")
print("=" * 70)

# Parameters
eta = 2.0  # Density ratio |ρ̄₀|/ρ₀
omega_plus = 1.0 / (1.0 + eta)   # = 1/3
omega_minus = eta / (1.0 + eta)  # = 2/3
e_conserved = omega_plus - omega_minus  # = -1/3 (negative!)

print(f"\nParameters:")
print(f"  η = {eta}")
print(f"  Ω₊ = {omega_plus:.4f}")
print(f"  Ω₋ = {omega_minus:.4f}")
print(f"  E = Ω₊ - Ω₋ = {e_conserved:.4f}")
print(f"  E < 0: {e_conserved < 0} → positive sector should ACCELERATE")

# Initial conditions at τ = 0 (today)
# From standard Friedmann: H² = Ω₊/a³ → ȧ = a·H = √(Ω₊/a) = √Ω₊ at a=1
a0 = 1.0
a_bar0 = 1.0
a_dot0 = np.sqrt(omega_plus)      # ≈ 0.577
a_bar_dot0 = -np.sqrt(omega_minus) # ≈ -0.816 (contracting)

print(f"\nInitial conditions (τ = 0):")
print(f"  a = {a0}")
print(f"  ā = {a_bar0}")
print(f"  ȧ = √Ω₊ = {a_dot0:.4f}")
print(f"  ā̇ = -√Ω₋ = {a_bar_dot0:.4f}")

# Compute initial accelerations
a_ddot0 = -1.5 * e_conserved / (a0 * a0)
a_bar_ddot0 = 1.5 * e_conserved / (a_bar0 * a_bar0)

print(f"\nInitial accelerations:")
print(f"  ä = -1.5 × E / a² = -1.5 × {e_conserved:.4f} / 1 = {a_ddot0:.4f}")
print(f"  ā̈ = +1.5 × E / ā² = +1.5 × {e_conserved:.4f} / 1 = {a_bar_ddot0:.4f}")
print(f"  ä > 0: {a_ddot0 > 0} → positive sector accelerates")
print(f"  ā̈ < 0: {a_bar_ddot0 < 0} → negative sector decelerates")

# RK4 integration BACKWARD in time
def derivatives(a, a_bar, a_dot, a_bar_dot, E):
    """Compute derivatives for Janus Friedmann equations."""
    a_ddot = -1.5 * E / (a * a)
    a_bar_ddot = 1.5 * E / (a_bar * a_bar)
    return a_dot, a_bar_dot, a_ddot, a_bar_ddot

def rk4_step(a, a_bar, a_dot, a_bar_dot, E, dtau):
    """One RK4 step."""
    da1, dab1, dda1, ddab1 = derivatives(a, a_bar, a_dot, a_bar_dot, E)

    a2 = a + 0.5 * dtau * da1
    ab2 = a_bar + 0.5 * dtau * dab1
    ad2 = a_dot + 0.5 * dtau * dda1
    abd2 = a_bar_dot + 0.5 * dtau * ddab1
    da2, dab2, dda2, ddab2 = derivatives(a2, ab2, ad2, abd2, E)

    a3 = a + 0.5 * dtau * da2
    ab3 = a_bar + 0.5 * dtau * dab2
    ad3 = a_dot + 0.5 * dtau * dda2
    abd3 = a_bar_dot + 0.5 * dtau * ddab2
    da3, dab3, dda3, ddab3 = derivatives(a3, ab3, ad3, abd3, E)

    a4 = a + dtau * da3
    ab4 = a_bar + dtau * dab3
    ad4 = a_dot + dtau * dda3
    abd4 = a_bar_dot + dtau * ddab3
    da4, dab4, dda4, ddab4 = derivatives(a4, ab4, ad4, abd4, E)

    a_new = a + dtau / 6.0 * (da1 + 2*da2 + 2*da3 + da4)
    a_bar_new = a_bar + dtau / 6.0 * (dab1 + 2*dab2 + 2*dab3 + dab4)
    a_dot_new = a_dot + dtau / 6.0 * (dda1 + 2*dda2 + 2*dda3 + dda4)
    a_bar_dot_new = a_bar_dot + dtau / 6.0 * (ddab1 + 2*ddab2 + 2*ddab3 + ddab4)

    return a_new, a_bar_new, a_dot_new, a_bar_dot_new

# Integrate backward
print("\n" + "=" * 70)
print("BACKWARD INTEGRATION (τ < 0, going to past)")
print("=" * 70)

a, a_bar = a0, a_bar0
a_dot, a_bar_dot = a_dot0, a_bar_dot0
tau = 0.0

n_steps = 1000
dtau = -0.01  # Backward

history = {
    'tau': [tau],
    'a': [a],
    'a_bar': [a_bar],
    'a_dot': [a_dot],
    'a_bar_dot': [a_bar_dot],
    'z': [1.0/a - 1.0],
}

for step in range(n_steps):
    a, a_bar, a_dot, a_bar_dot = rk4_step(a, a_bar, a_dot, a_bar_dot, e_conserved, dtau)
    tau += dtau

    # Safety
    if a <= 0.01 or a_bar <= 0.01 or a > 100 or a_bar > 100:
        print(f"  Step {step}: STOPPED (a={a:.4f}, ā={a_bar:.4f})")
        break

    history['tau'].append(tau)
    history['a'].append(a)
    history['a_bar'].append(a_bar)
    history['a_dot'].append(a_dot)
    history['a_bar_dot'].append(a_bar_dot)
    history['z'].append(1.0/a - 1.0)

    if step < 10 or step % 100 == 0:
        z = 1.0/a - 1.0
        print(f"  Step {step:4d}: τ={tau:7.3f}, a={a:.4f}, ā={a_bar:.4f}, z={z:.4f}")

# Convert to arrays
for key in history:
    history[key] = np.array(history[key])

print(f"\nFinal state:")
print(f"  τ = {history['tau'][-1]:.3f}")
print(f"  a = {history['a'][-1]:.4f}")
print(f"  ā = {history['a_bar'][-1]:.4f}")
print(f"  z = {history['z'][-1]:.4f}")
print(f"  ȧ = {history['a_dot'][-1]:.4f}")
print(f"  ā̇ = {history['a_bar_dot'][-1]:.4f}")

# Check if we reached high z
max_z = history['z'].max()
print(f"\nMax redshift reached: z = {max_z:.2f}")
if max_z > 1.5:
    print("✓ PASS: Reached z > 1.5")
else:
    print(f"✗ FAIL: Did not reach z > 1.5")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scale factors vs time
ax = axes[0, 0]
ax.plot(history['tau'], history['a'], 'b-', lw=2, label='a (positive)')
ax.plot(history['tau'], history['a_bar'], 'r--', lw=2, label='ā (negative)')
ax.axhline(1, color='k', ls=':', alpha=0.5)
ax.set_xlabel('τ = H₀t')
ax.set_ylabel('Scale factor')
ax.set_title('Scale Factors vs Time')
ax.legend()
ax.grid(alpha=0.3)

# 2. Velocities vs time
ax = axes[0, 1]
ax.plot(history['tau'], history['a_dot'], 'b-', lw=2, label='ȧ')
ax.plot(history['tau'], history['a_bar_dot'], 'r--', lw=2, label='ā̇')
ax.axhline(0, color='k', ls='-', alpha=0.5)
ax.set_xlabel('τ = H₀t')
ax.set_ylabel('da/dτ')
ax.set_title('Velocities vs Time')
ax.legend()
ax.grid(alpha=0.3)

# 3. Redshift vs time
ax = axes[1, 0]
ax.plot(history['tau'], history['z'], 'g-', lw=2)
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('τ = H₀t')
ax.set_ylabel('z = 1/a - 1')
ax.set_title('Redshift vs Time')
ax.grid(alpha=0.3)

# 4. Phase space a vs ȧ
ax = axes[1, 1]
ax.plot(history['a'], history['a_dot'], 'b-', lw=2, label='positive sector')
ax.plot(history['a'][0], history['a_dot'][0], 'bo', ms=10, label='today')
ax.set_xlabel('a')
ax.set_ylabel('ȧ')
ax.set_title('Phase Space (a, ȧ)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output/debug_friedmann.png', dpi=150)
print(f"\n✓ Saved output/debug_friedmann.png")

# Analyze the problem
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

# Check: with E < 0, ä > 0 always
# Going backward, ȧ should DECREASE (we're unwinding the acceleration)
# Eventually ȧ could become negative if acceleration was always positive

# At what point does ȧ become negative?
if np.any(history['a_dot'] < 0):
    idx = np.where(history['a_dot'] < 0)[0][0]
    print(f"\nȧ becomes negative at:")
    print(f"  τ = {history['tau'][idx]:.3f}")
    print(f"  a = {history['a'][idx]:.4f}")
    print(f"  z = {history['z'][idx]:.4f}")
else:
    print("\nȧ remains positive throughout integration")

# The issue: with ä > 0 always (when E < 0), we have perpetual acceleration
# This means going backward, ȧ was SMALLER in the past
# The universe was expanding slower in the past
# But standard cosmology has: ȧ LARGER in the past (deceleration)
print(f"""
ISSUE IDENTIFIED:
  With E < 0, the equation ä = -1.5E/a² gives ä > 0 (always accelerating)
  Going backward in time, ȧ should have been SMALLER (unwinding acceleration)

  But the initial condition ȧ₀ = √Ω₊ comes from H² = Ω₊/a³ (Friedmann)
  This assumes DECELERATION (ä < 0) from matter domination.

  There's a mismatch between:
    1. Janus acceleration equations (ä > 0 when E < 0)
    2. Standard Friedmann initial condition (derived assuming ä < 0)

  We need to reconsider the initial conditions for Janus cosmology.
""")
