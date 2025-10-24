#ifndef SPHERICAL_HARMONICS_CUH
#define SPHERICAL_HARMONICS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.cuh"

/**
 * Spherical Harmonics Evaluation for NLOS Gaussian Rendering
 * 
 * Provides view-dependent albedo computation using spherical harmonics
 * up to degree 3 (16 coefficients).
 * 
 * Usage:
 *   float3 view_dir = normalize(gaussian_pos - camera_pos);
 *   float albedo = eval_sh(degree, sh_coeffs, view_dir);
 */

// Spherical harmonics evaluation (simplified for degree 0-3)
__device__ inline float eval_sh_basis(int l, int m, const float3& dir) {
    const float C0 = 0.28209479177387814f;  // 1 / (2 * sqrt(pi))
    
    if (l == 0) {
        return C0;
    }
    
    float x = dir.x, y = dir.y, z = dir.z;
    
    if (l == 1) {
        if (m == -1) return 0.4886025119029199f * y;       // sqrt(3/(4pi)) * y
        if (m ==  0) return 0.4886025119029199f * z;       // sqrt(3/(4pi)) * z
        if (m ==  1) return 0.4886025119029199f * x;       // sqrt(3/(4pi)) * x
    }
    
    if (l == 2) {
        if (m == -2) return 1.0925484305920792f * x * y;
        if (m == -1) return 1.0925484305920792f * y * z;
        if (m ==  0) return 0.31539156525252005f * (3.0f * z * z - 1.0f);
        if (m ==  1) return 1.0925484305920792f * x * z;
        if (m ==  2) return 0.5462742152960396f * (x * x - y * y);
    }
    
    if (l == 3) {
        if (m == -3) return 0.5900435899266435f * y * (3.0f * x * x - y * y);
        if (m == -2) return 2.890611442640554f * x * y * z;
        if (m == -1) return 0.4570457994644658f * y * (5.0f * z * z - 1.0f);
        if (m ==  0) return 0.3731763325901154f * z * (5.0f * z * z - 3.0f);
        if (m ==  1) return 0.4570457994644658f * x * (5.0f * z * z - 1.0f);
        if (m ==  2) return 1.445305721320277f * z * (x * x - y * y);
        if (m ==  3) return 0.5900435899266435f * x * (x * x - 3.0f * y * y);
    }
    
    return 0.0f;
}

/**
 * Evaluate spherical harmonics with given coefficients
 * 
 * @param degree Maximum SH degree (0-3 supported)
 * @param sh_coeffs SH coefficients [(degree+1)^2 values]
 * @param dir Direction vector (should be normalized)
 * @return Evaluated SH value
 */
__device__ inline float eval_sh(
    int degree,
    const float* sh_coeffs,  // [(degree+1)^2] coefficients
    const float3& dir
) {
    float result = 0.0f;
    int idx = 0;
    
    for (int l = 0; l <= degree; l++) {
        for (int m = -l; m <= l; m++) {
            result += sh_coeffs[idx] * eval_sh_basis(l, m, dir);
            idx++;
        }
    }
    
    return result;
}

#endif // SPHERICAL_HARMONICS_CUH