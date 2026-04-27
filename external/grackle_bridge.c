/*
 * Grackle Bridge - Simplified C interface for Rust FFI
 * Exposes cooling_rate(T, rho, z) → dU/dt [erg/s/g]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRACKLE_FLOAT_8
#include "grackle.h"

static chemistry_data *my_chemistry = NULL;
static chemistry_data_storage my_rates;
static code_units my_units;
static int grackle_initialized = 0;

// Initialize Grackle with HM2012 UV background
int grackle_bridge_init(const char* data_file_path) {
    if (grackle_initialized) return 1;

    grackle_verbose = 0;

    // Allocate chemistry data
    my_chemistry = malloc(sizeof(chemistry_data));
    if (my_chemistry == NULL) {
        fprintf(stderr, "grackle_bridge: Failed to allocate chemistry_data\n");
        return 0;
    }

    // Set default parameters
    if (set_default_chemistry_parameters(my_chemistry) == 0) {
        fprintf(stderr, "grackle_bridge: set_default_chemistry_parameters failed\n");
        free(my_chemistry);
        return 0;
    }

    // Configure for tabulated cooling with metals and UV background
    my_chemistry->use_grackle = 1;
    my_chemistry->with_radiative_cooling = 1;
    my_chemistry->primordial_chemistry = 0;  // tabulated cooling (no chemistry evolution)
    my_chemistry->metal_cooling = 1;          // include metals
    my_chemistry->UVbackground = 1;           // HM2012 UV background
    my_chemistry->grackle_data_file = data_file_path;
    my_chemistry->Gamma = 5.0 / 3.0;          // monatomic gas
    my_chemistry->HydrogenFractionByMass = -1.0;  // let Grackle calculate

    // Set code units (CGS for simplicity)
    my_units.comoving_coordinates = 0;
    my_units.density_units = 1.0;     // g/cm^3
    my_units.length_units = 1.0;      // cm
    my_units.time_units = 1.0;        // s
    my_units.velocity_units = 1.0;    // cm/s
    my_units.a_units = 1.0;
    my_units.a_value = 1.0;           // will be set per call

    // Initialize chemistry data
    if (local_initialize_chemistry_data(my_chemistry, &my_rates, &my_units) == 0) {
        fprintf(stderr, "grackle_bridge: local_initialize_chemistry_data failed\n");
        free(my_chemistry);
        return 0;
    }

    grackle_initialized = 1;
    return 1;
}

// Compute cooling rate: Λ(T, rho, z) → dE/dt [erg/s/cm³]
// Returns cooling rate in erg/s/cm³ (positive = cooling)
double grackle_bridge_cooling_rate(double temperature_K, double density_cgs, double redshift) {
    if (!grackle_initialized) {
        fprintf(stderr, "grackle_bridge: not initialized\n");
        return 0.0;
    }

    // Scale factor
    double a_value = 1.0 / (1.0 + redshift);
    my_units.a_value = a_value;

    // Compute internal energy from temperature
    // e = (3/2) k_B T / (μ m_H), where μ ≈ 0.6 for ionized primordial
    double k_B = 1.3807e-16;      // erg/K
    double m_H = 1.6726e-24;      // g
    double mu = 0.6;              // mean molecular weight (ionized)
    double internal_energy = (3.0/2.0) * k_B * temperature_K / (mu * m_H);  // erg/g

    // Set up field data for a single cell
    grackle_field_data my_fields;
    gr_initialize_field_data(&my_fields);

    int grid_dim[3] = {1, 1, 1};
    int grid_start[3] = {0, 0, 0};
    int grid_end[3] = {0, 0, 0};

    my_fields.grid_rank = 3;
    my_fields.grid_dimension = grid_dim;
    my_fields.grid_start = grid_start;
    my_fields.grid_end = grid_end;
    my_fields.grid_dx = 1.0;

    // Single cell data
    gr_float density_field = density_cgs;
    gr_float energy_field = internal_energy;
    gr_float metal_field = density_cgs * 0.02;  // 2% solar metallicity
    gr_float vel_x = 0.0, vel_y = 0.0, vel_z = 0.0;

    my_fields.density = &density_field;
    my_fields.internal_energy = &energy_field;
    my_fields.metal_density = &metal_field;
    my_fields.x_velocity = &vel_x;
    my_fields.y_velocity = &vel_y;
    my_fields.z_velocity = &vel_z;

    // Calculate cooling time
    gr_float cooling_time;
    if (local_calculate_cooling_time(my_chemistry, &my_rates, &my_units,
                                      &my_fields, &cooling_time) == 0) {
        fprintf(stderr, "grackle_bridge: calculate_cooling_time failed\n");
        return 0.0;
    }

    // Cooling rate = energy / cooling_time
    // Λ [erg/s/cm³] = ρ * e / t_cool
    double cooling_rate = density_cgs * internal_energy / fabs(cooling_time);

    return cooling_rate;
}

// Get normalized cooling function Λ/n_H² [erg·cm³/s]
// This is the standard form used in astrophysics
double grackle_bridge_lambda_norm(double temperature_K, double redshift) {
    // Use reference density n_H = 1 cm^-3
    double n_H = 1.0;  // cm^-3
    double m_H = 1.6726e-24;  // g
    double X_H = 0.76;  // hydrogen mass fraction
    double rho = n_H * m_H / X_H;  // g/cm³

    double cooling_rate = grackle_bridge_cooling_rate(temperature_K, rho, redshift);

    // Λ_norm = Λ / n_H²
    return cooling_rate / (n_H * n_H);
}

// Clean up
void grackle_bridge_cleanup(void) {
    if (grackle_initialized) {
        local_free_chemistry_data(my_chemistry, &my_rates);
        free(my_chemistry);
        my_chemistry = NULL;
        grackle_initialized = 0;
    }
}

// Test function
int grackle_bridge_test(void) {
    double T = pow(10.0, 4.5);  // 10^4.5 K ≈ 31623 K
    double lambda = grackle_bridge_lambda_norm(T, 0.0);
    printf("Λ(10^4.5 K) = %.3e erg·cm³/s\n", lambda);
    printf("Expected:   ≈ 1.6e-22 erg·cm³/s\n");
    return (lambda > 1e-23 && lambda < 1e-21) ? 1 : 0;
}
