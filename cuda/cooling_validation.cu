// Cooling Rate Validation — Comparing 4 fits to Sutherland & Dopita 1993
// Base version + 3 AI-proposed corrections for Lyman-alpha peak
//
// References:
//   - Sutherland & Dopita 1993, ApJS 88, 253
//   - Cen 1992 (H collisional ionization)
//   - Primordial CIE (H + He only, no metals)

#include <math.h>

extern "C" {

// ═══════════════════════════════════════════════════════════════════════
// VERSION BASE — Fit original sans correction
// ═══════════════════════════════════════════════════════════════════════

__device__ double cooling_base(double T)
{
    if (T < 1e4) return 0.0;
    double sqrtT = sqrt(T);
    double Lambda_H  = 7.5e-19 * exp(-118348.0 / T)
                       / (1.0 + sqrt(T / 1e5));
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0 / T);
    double Lambda_ff = 1.42e-27 * sqrtT;
    return Lambda_H + Lambda_He + Lambda_ff;
}

// ═══════════════════════════════════════════════════════════════════════
// VERSION CHATGPT — Correction additive gaussienne sur Lambda_H
// Centre T=25000K, amplitude +12%, sigma=6000K
// ═══════════════════════════════════════════════════════════════════════

__device__ double cooling_chatgpt(double T)
{
    if (T < 1e4) return 0.0;
    double sqrtT = sqrt(T);
    double Lambda_H  = 7.5e-19 * exp(-118348.0 / T)
                       / (1.0 + sqrt(T / 1e5));
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0 / T);
    double Lambda_ff = 1.42e-27 * sqrtT;
    // Correction : gaussienne additive sur Lambda_H
    double correction = Lambda_H * 0.12
                      * exp(-pow((T - 25000.0) / 6000.0, 2.0));
    return Lambda_H + Lambda_He + Lambda_ff + correction;
}

// ═══════════════════════════════════════════════════════════════════════
// VERSION GEMINI — Correction multiplicative globale
// Centre logT=4.18 (T~15100K), amplitude +62%, sigma=0.075 en log
// ═══════════════════════════════════════════════════════════════════════

__device__ double cooling_gemini(double T)
{
    if (T < 1e4) return 0.0;
    double sqrtT = sqrt(T);
    double Lambda_H  = 7.5e-19 * exp(-118348.0 / T)
                       / (1.0 + sqrt(T / 1e5));
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0 / T);
    double Lambda_ff = 1.42e-27 * sqrtT;
    double Lambda = Lambda_H + Lambda_He + Lambda_ff;
    // Correction : multiplicative gaussienne en log(T)
    double logT = log10(T);
    double d = logT - 4.18;
    double correction = 1.0 + 0.62 * exp(-88.8889 * d * d);
    return Lambda * correction;
}

// ═══════════════════════════════════════════════════════════════════════
// VERSION MISTRAL — Correction additive asymétrique
// Centre logT=4.146 (T~14000K), sigma_low=0.12, sigma_high=0.28
// ═══════════════════════════════════════════════════════════════════════

__device__ double cooling_mistral(double T)
{
    if (T < 5000.0 || T > 150000.0) return cooling_base(T);
    double sqrtT = sqrt(T);
    double Lambda_H  = 7.5e-19 * exp(-118348.0 / T)
                       / (1.0 + sqrt(T / 1e5));
    double Lambda_He = 9.1e-27 * sqrtT * exp(-13179.0 / T);
    double Lambda_ff = 1.42e-27 * sqrtT;
    double Lambda = Lambda_H + Lambda_He + Lambda_ff;
    float logT = log10f((float)T);
    const float A = -21.82f;
    const float mu = 4.146f;
    const float sigma_low = 0.12f;
    const float sigma_high = 0.28f;
    float delta = logT - mu;
    float sigma = (delta < 0.0f) ? sigma_low : sigma_high;
    float log_corr = A * expf(-(delta * delta)
                    / (2.0f * sigma * sigma));
    double corr_add = pow(10.0, (double)log_corr) * 0.155;
    return Lambda + corr_add;
}

// ═══════════════════════════════════════════════════════════════════════
// TEST KERNEL — Compare all 4 versions at given temperatures
// ═══════════════════════════════════════════════════════════════════════

__global__ void test_cooling_fits(
    const double* __restrict__ temperatures,  // [N] input temperatures
    double* __restrict__ lambda_base,         // [N] output base
    double* __restrict__ lambda_chatgpt,      // [N] output ChatGPT
    double* __restrict__ lambda_gemini,       // [N] output Gemini
    double* __restrict__ lambda_mistral,      // [N] output Mistral
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double T = temperatures[i];
    lambda_base[i] = cooling_base(T);
    lambda_chatgpt[i] = cooling_chatgpt(T);
    lambda_gemini[i] = cooling_gemini(T);
    lambda_mistral[i] = cooling_mistral(T);
}

} // extern "C"
