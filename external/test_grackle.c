#include <stdio.h>
#include <math.h>
#include "grackle_bridge.h"

int main() {
    const char* data_file = "/usr/local/share/grackle/input/CloudyData_UVB=HM2012.h5";

    printf("Initializing Grackle with HM2012...\n");
    if (!grackle_bridge_init(data_file)) {
        fprintf(stderr, "Failed to initialize Grackle\n");
        return 1;
    }
    printf("Grackle initialized successfully.\n\n");

    // Test at 10^4.5 K
    double T = pow(10.0, 4.5);
    double lambda = grackle_bridge_lambda_norm(T, 0.0);
    printf("=== Test: Λ(10^4.5 K) ===\n");
    printf("Temperature: %.1f K (10^4.5 K)\n", T);
    printf("Λ_norm = %.3e erg·cm³/s\n", lambda);
    printf("Expected: ≈ 1.6e-22 erg·cm³/s\n\n");

    // Test cooling curve at z=0
    printf("=== Cooling Curve z=0 ===\n");
    printf("log(T/K)   Λ [erg·cm³/s]\n");
    for (double logT = 4.0; logT <= 8.0; logT += 0.5) {
        double T_test = pow(10.0, logT);
        double L = grackle_bridge_lambda_norm(T_test, 0.0);
        printf("  %.1f      %.3e\n", logT, L);
    }

    // Test redshift dependence
    printf("\n=== Λ(10^5 K) vs Redshift ===\n");
    T = 1e5;
    for (double z = 0.0; z <= 4.0; z += 1.0) {
        double L = grackle_bridge_lambda_norm(T, z);
        printf("z=%.0f: Λ = %.3e erg·cm³/s\n", z, L);
    }

    grackle_bridge_cleanup();
    printf("\nDone.\n");
    return 0;
}
