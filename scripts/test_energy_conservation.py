#!/usr/bin/env python3
"""
TEST 1: Energy Conservation with 2 Particles

Tests the Leapfrog integrator with Plummer softening on a simple
2-body system: 1 positive mass + 1 negative mass.

In Janus physics: opposite masses REPEL each other.
So we expect them to fly apart, but TOTAL ENERGY should be conserved.

Energy:
  KE = 0.5 * m * v²  (both particles)
  PE = +m₁*m₂/r  (positive because they repel: opposite signs)
  E_total = KE + PE = constant (should be conserved by Leapfrog)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("TEST 1: Energy Conservation (2 Particles)")
print("=" * 70)

# Parameters
dt = 0.005
n_steps = 1000
softening = 0.1  # Same as used in simulation

# Initial conditions: 2 particles at distance 1, at rest
# Particle 1: positive mass at (-0.5, 0, 0)
# Particle 2: negative mass at (+0.5, 0, 0)
m1, m2 = 1.0, 1.0
sign1, sign2 = +1, -1  # Positive and negative mass

pos1 = np.array([-0.5, 0.0, 0.0])
pos2 = np.array([+0.5, 0.0, 0.0])
vel1 = np.array([0.0, 0.0, 0.0])
vel2 = np.array([0.0, 0.0, 0.0])

def compute_acceleration(pos1, pos2, m1, m2, sign1, sign2, eps):
    """Compute acceleration on particle 1 from particle 2 using Plummer softening"""
    r_vec = pos2 - pos1  # Points from 1 to 2
    r2 = np.sum(r_vec**2)
    r2_soft = r2 + eps**2

    # Janus: same sign attract, opposite sign repel
    interaction = 1.0 if sign1 == sign2 else -1.0

    # Plummer force: a = G*m*r_vec / (r² + ε²)^(3/2)
    inv_r3_soft = 1.0 / (r2_soft * np.sqrt(r2_soft))
    acc1 = interaction * m2 * r_vec * inv_r3_soft

    return acc1

def compute_ke(vel, m):
    """Kinetic energy"""
    return 0.5 * m * np.sum(vel**2)

def compute_pe(pos1, pos2, m1, m2, sign1, sign2, eps):
    """Potential energy with Plummer softening"""
    r_vec = pos2 - pos1
    r2 = np.sum(r_vec**2)
    r_soft = np.sqrt(r2 + eps**2)

    # Janus: same sign → attractive potential (negative)
    #        opposite sign → repulsive potential (positive)
    interaction = -1.0 if sign1 == sign2 else +1.0

    # Plummer potential: φ = G*m/sqrt(r² + ε²)
    return interaction * m1 * m2 / r_soft

# Storage
times = []
ke1_arr, ke2_arr = [], []
pe_arr = []
e_total_arr = []
r_arr = []

# Initial energy
ke1 = compute_ke(vel1, m1)
ke2 = compute_ke(vel2, m2)
pe = compute_pe(pos1, pos2, m1, m2, sign1, sign2, softening)
e0 = ke1 + ke2 + pe

times.append(0.0)
ke1_arr.append(ke1)
ke2_arr.append(ke2)
pe_arr.append(pe)
e_total_arr.append(e0)
r_arr.append(np.linalg.norm(pos2 - pos1))

print(f"\nInitial state:")
print(f"  r = {r_arr[0]:.4f}")
print(f"  KE1 = {ke1:.6f}, KE2 = {ke2:.6f}")
print(f"  PE = {pe:.6f}")
print(f"  E_total = {e0:.6f}")

# Leapfrog integration
print(f"\nRunning {n_steps} steps with dt={dt}...")

for step in range(n_steps):
    # Compute accelerations
    acc1 = compute_acceleration(pos1, pos2, m1, m2, sign1, sign2, softening)
    acc2 = -compute_acceleration(pos2, pos1, m2, m1, sign2, sign1, softening)  # Newton's 3rd law (sort of)

    # Actually, for Janus, let's compute both properly
    acc2 = compute_acceleration(pos2, pos1, m2, m1, sign2, sign1, softening)

    # Half-step velocity update
    vel1 = vel1 + 0.5 * dt * acc1
    vel2 = vel2 + 0.5 * dt * acc2

    # Full-step position update
    pos1 = pos1 + dt * vel1
    pos2 = pos2 + dt * vel2

    # Recompute accelerations at new positions
    acc1 = compute_acceleration(pos1, pos2, m1, m2, sign1, sign2, softening)
    acc2 = compute_acceleration(pos2, pos1, m2, m1, sign2, sign1, softening)

    # Half-step velocity update
    vel1 = vel1 + 0.5 * dt * acc1
    vel2 = vel2 + 0.5 * dt * acc2

    # Compute energies
    ke1 = compute_ke(vel1, m1)
    ke2 = compute_ke(vel2, m2)
    pe = compute_pe(pos1, pos2, m1, m2, sign1, sign2, softening)
    e_total = ke1 + ke2 + pe

    times.append((step + 1) * dt)
    ke1_arr.append(ke1)
    ke2_arr.append(ke2)
    pe_arr.append(pe)
    e_total_arr.append(e_total)
    r_arr.append(np.linalg.norm(pos2 - pos1))

# Convert to arrays
times = np.array(times)
ke1_arr = np.array(ke1_arr)
ke2_arr = np.array(ke2_arr)
pe_arr = np.array(pe_arr)
e_total_arr = np.array(e_total_arr)
r_arr = np.array(r_arr)

# Compute energy drift
e_drift = (e_total_arr - e0) / np.abs(e0) * 100  # Percentage drift

print(f"\nFinal state:")
print(f"  r = {r_arr[-1]:.4f}")
print(f"  KE1 = {ke1_arr[-1]:.6f}, KE2 = {ke2_arr[-1]:.6f}")
print(f"  PE = {pe_arr[-1]:.6f}")
print(f"  E_total = {e_total_arr[-1]:.6f}")
print(f"\nEnergy conservation:")
print(f"  E_initial = {e0:.6f}")
print(f"  E_final = {e_total_arr[-1]:.6f}")
print(f"  Max |ΔE/E₀| = {np.max(np.abs(e_drift)):.4f}%")
print(f"  Final ΔE/E₀ = {e_drift[-1]:.4f}%")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distance over time
ax = axes[0, 0]
ax.plot(times, r_arr, 'b-', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Distance r', fontsize=12)
ax.set_title('Separation Distance', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Energies over time
ax = axes[0, 1]
ax.plot(times, ke1_arr + ke2_arr, 'r-', linewidth=2, label='KE')
ax.plot(times, pe_arr, 'b--', linewidth=2, label='PE')
ax.plot(times, e_total_arr, 'k-', linewidth=3, label='E_total')
ax.axhline(e0, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('Energy Components', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Energy drift
ax = axes[1, 0]
ax.plot(times, e_drift, 'r-', linewidth=2)
ax.axhline(0, color='k', linestyle='-')
ax.axhline(1, color='g', linestyle='--', alpha=0.5, label='±1% threshold')
ax.axhline(-1, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('ΔE/E₀ (%)', fontsize=12)
ax.set_title('Energy Conservation', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Summary
ax = axes[1, 1]
ax.axis('off')
summary = f"""
═══════════════════════════════════════════════════
    TEST 1: ENERGY CONSERVATION (2 PARTICLES)
═══════════════════════════════════════════════════

SETUP
─────
  Particle 1: mass=1, sign=POSITIVE
  Particle 2: mass=1, sign=NEGATIVE
  Initial distance: 1.0
  Initial velocity: 0 (at rest)
  Softening ε: {softening}

INTEGRATION
───────────
  Method: Leapfrog (symplectic)
  Steps: {n_steps}
  dt: {dt}
  Total time: {times[-1]:.2f}

RESULTS
───────
  Final distance: {r_arr[-1]:.2f}

  E_initial: {e0:.6f}
  E_final:   {e_total_arr[-1]:.6f}

  Max |ΔE/E₀|: {np.max(np.abs(e_drift)):.4f}%

VERDICT
───────
  {'✓ PASS: Energy conserved < 1%' if np.max(np.abs(e_drift)) < 1 else '✗ FAIL: Energy drift > 1%'}
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('output/test1_energy_conservation.png', dpi=150)
print(f"\n✓ Saved output/test1_energy_conservation.png")

# Verdict
print("\n" + "=" * 70)
if np.max(np.abs(e_drift)) < 1.0:
    print("VERDICT: ✓ PASS — Energy conserved to < 1%")
else:
    print(f"VERDICT: ✗ FAIL — Energy drift = {np.max(np.abs(e_drift)):.2f}%")
print("=" * 70)
