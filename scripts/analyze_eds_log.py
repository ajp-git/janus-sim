#!/usr/bin/env python3
"""
Analyze test_eds_growing_mode log: compute R_local for adjacent snapshots.
"""
import sys
import re

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/app/output/eds_growing_mode.log"
    rows = []
    with open(log_path) as f:
        for line in f:
            # Match "step a z t_gyr sigma_filt sigma_8 D_meas D_th R"
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                step = int(m.group(1))
                a = float(m.group(2))
                z = float(m.group(3))
                sig_filt = float(m.group(5))
                sig_8 = float(m.group(6))
                R = float(m.group(9))
                rows.append((step, a, z, sig_filt, sig_8, R))

    print(f"# EdS log analysis: {log_path}")
    print(f"# {len(rows)} snapshots loaded")
    print()
    print(f"{'step':>5} {'a':>7} {'z':>7} {'sigma_8':>8} {'R_local(σ8)':>12}")
    last_a, last_s8 = None, None
    for r in rows:
        step, a, z, sf, s8, R = r
        if last_a is None:
            R_local = 1.0
        else:
            R_local = (s8 / last_s8) / (a / last_a)
        print(f"{step:>5} {a:.5f} {z:>7.3f} {s8:.4e} {R_local:>12.4f}")
        last_a, last_s8 = a, s8
    print()
    # Compute mean R_local for late times (a > 3 a_init)
    a_init = rows[0][1]
    late_R = []
    last_a, last_s8 = None, None
    for r in rows:
        step, a, z, sf, s8, R = r
        if last_a is not None and a > 3.0 * a_init:
            R_local = (s8 / last_s8) / (a / last_a)
            late_R.append((a, z, R_local))
        last_a, last_s8 = a, s8

    if late_R:
        import statistics
        rs = [r for (_, _, r) in late_R]
        print(f"# Late-time (a > {3.0*a_init:.4f}) R_local statistics:")
        print(f"#   N      = {len(rs)}")
        print(f"#   mean   = {statistics.mean(rs):.4f}")
        print(f"#   median = {statistics.median(rs):.4f}")
        print(f"#   stdev  = {statistics.stdev(rs) if len(rs)>1 else 0:.4f}")
        print(f"#   min    = {min(rs):.4f}")
        print(f"#   max    = {max(rs):.4f}")
        # Specific z checkpoints
        for z_target in [10, 5, 3, 2]:
            best = min(late_R, key=lambda t: abs(t[1] - z_target))
            print(f"#   R_local at z≈{z_target}: {best[2]:.4f} (actual z={best[1]:.2f}, a={best[0]:.4f})")

if __name__ == "__main__":
    main()
