/*
 * Grackle Bridge - Simplified C interface for Rust FFI
 */

#ifndef GRACKLE_BRIDGE_H
#define GRACKLE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Grackle with data file path (HM2012 tables)
// Returns 1 on success, 0 on failure
int grackle_bridge_init(const char* data_file_path);

// Compute cooling rate: Λ(T, ρ, z) → dE/dt [erg/s/cm³]
// temperature_K: gas temperature in Kelvin
// density_cgs: gas density in g/cm³
// redshift: cosmological redshift
// Returns cooling rate in erg/s/cm³
double grackle_bridge_cooling_rate(double temperature_K, double density_cgs, double redshift);

// Get normalized cooling function Λ/n_H² [erg·cm³/s]
// This is the standard astrophysical form
double grackle_bridge_lambda_norm(double temperature_K, double redshift);

// Clean up Grackle resources
void grackle_bridge_cleanup(void);

// Test function - returns 1 if Λ(10^4.5 K) ≈ 1.6e-22
int grackle_bridge_test(void);

#ifdef __cplusplus
}
#endif

#endif /* GRACKLE_BRIDGE_H */
