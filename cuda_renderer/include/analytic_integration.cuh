#ifndef ANALYTIC_INTEGRATION_CUH
#define ANALYTIC_INTEGRATION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Analytic Integration for Gaussian Primitives
 * Based on "Don't Splat your Gaussians" (Condor et al., 2024)
 * 
 * This provides closed-form solutions for transmittance computation
 * along a ray through Gaussian primitives.
 */

// Structure to hold ray-Gaussian intersection information
struct GaussianSection {
    int gaussian_idx;      // Index of the Gaussian
    float t_enter;         // Ray parameter where Gaussian influence starts
    float t_exit;          // Ray parameter where Gaussian influence ends
    float t_peak;          // Ray parameter at closest point to Gaussian center
};

/**
 * Compute ray-Gaussian intersection interval
 * Returns the t_min and t_max where Gaussian has significant contribution
 * 
 * @param ray_o Ray origin
 * @param ray_d Ray direction (normalized)
 * @param mean Gaussian center
 * @param scale Gaussian scales (after exp)
 * @param quat Gaussian rotation (quaternion)
 * @param sigma_threshold Threshold in standard deviations (e.g., 3.0)
 * @param t_enter [out] Entry point
 * @param t_exit [out] Exit point
 * @param t_peak [out] Closest point to center
 * @return true if intersection exists
 */
__device__ inline bool compute_gaussian_section(
    const float3& ray_o,
    const float3& ray_d,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float sigma_threshold,
    float& t_enter,
    float& t_exit,
    float& t_peak
) {
    // Build rotation matrix
    float R[9];
    quat_to_rotmat(quat, R);
    
    // Transform ray to Gaussian local space
    // Local ray origin: o' = R^T * (ray_o - mean)
    float3 diff = ray_o - mean;
    float3 local_o = make_float3(
        R[0]*diff.x + R[3]*diff.y + R[6]*diff.z,
        R[1]*diff.x + R[4]*diff.y + R[7]*diff.z,
        R[2]*diff.x + R[5]*diff.y + R[8]*diff.z
    );
    
    // Local ray direction: d' = R^T * ray_d
    float3 local_d = make_float3(
        R[0]*ray_d.x + R[3]*ray_d.y + R[6]*ray_d.z,
        R[1]*ray_d.x + R[4]*ray_d.y + R[7]*ray_d.z,
        R[2]*ray_d.x + R[5]*ray_d.y + R[8]*ray_d.z
    );
    
    // Solve for intersection with sigma_threshold * scale ellipsoid
    // ||((o' + t*d') / scale)||^2 = sigma_threshold^2
    // This is a quadratic equation in t
    
    float3 scaled_d = make_float3(
        local_d.x / scale.x,
        local_d.y / scale.y,
        local_d.z / scale.z
    );
    
    float3 scaled_o = make_float3(
        local_o.x / scale.x,
        local_o.y / scale.y,
        local_o.z / scale.z
    );
    
    float a = dot(scaled_d, scaled_d);
    float b = 2.0f * dot(scaled_o, scaled_d);
    float c = dot(scaled_o, scaled_o) - sigma_threshold * sigma_threshold;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0.0f) {
        return false;  // No intersection
    }
    
    float sqrt_disc = sqrtf(discriminant);
    t_enter = (-b - sqrt_disc) / (2.0f * a);
    t_exit = (-b + sqrt_disc) / (2.0f * a);
    
    // Find closest point to center (minimum distance)
    // d/dt ||o' + t*d'||^2 = 0
    t_peak = -dot(scaled_o, scaled_d) / dot(scaled_d, scaled_d);
    
    return true;
}

/**
 * Compute analytic transmittance integral through a single Gaussian
 * 
 * This computes: τ_i = ∫[t0 to t1] σ_i * exp(-0.5 * ||R^T(x(t) - μ) / s||^2) dt
 * 
 * Based on closed-form solution from paper Appendix B.
 * 
 * @param ray_o Ray origin
 * @param ray_d Ray direction (normalized)
 * @param t0 Integration start
 * @param t1 Integration end
 * @param mean Gaussian center
 * @param scale Gaussian scales
 * @param quat Gaussian rotation
 * @param opacity Gaussian opacity (σ_i)
 * @return Integrated optical depth τ_i
 */
__device__ inline float compute_analytic_transmittance(
    const float3& ray_o,
    const float3& ray_d,
    const float t0,
    const float t1,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float opacity
) {
    // Build rotation matrix
    float R[9];
    quat_to_rotmat(quat, R);
    
    // Transform to Gaussian local space
    float3 diff = ray_o - mean;
    float3 v = make_float3(
        R[0]*diff.x + R[3]*diff.y + R[6]*diff.z,
        R[1]*diff.x + R[4]*diff.y + R[7]*diff.z,
        R[2]*diff.x + R[5]*diff.y + R[8]*diff.z
    );
    
    float3 omega = make_float3(
        R[0]*ray_d.x + R[3]*ray_d.y + R[6]*ray_d.z,
        R[1]*ray_d.x + R[4]*ray_d.y + R[7]*ray_d.z,
        R[2]*ray_d.x + R[5]*ray_d.y + R[8]*ray_d.z
    );
    
    // Scale by covariance
    float3 v_scaled = make_float3(v.x / scale.x, v.y / scale.y, v.z / scale.z);
    float3 omega_scaled = make_float3(omega.x / scale.x, omega.y / scale.y, omega.z / scale.z);
    
    float a = dot(v_scaled, v_scaled);
    float b = 2.0f * dot(v_scaled, omega_scaled);
    float c = dot(omega_scaled, omega_scaled);
    
    // Compute CDF using closed-form solution
    // This is a simplified version - full implementation in paper Appendix B
    float G = opacity * sqrtf(2.0f * M_PI / c) * scale.x * scale.y * scale.z;
    
    float exp_factor = expf(-0.5f * (a - b * b / (4.0f * c)));
    
    // Error function for integration bounds
    float erf_t1 = erff((b + 2.0f * c * t1) / (2.0f * sqrtf(c)));
    float erf_t0 = erff((b + 2.0f * c * t0) / (2.0f * sqrtf(c)));
    
    float tau = G * exp_factor * (erf_t1 - erf_t0);
    
    return fmaxf(tau, 0.0f);
}

/**
 * Sort Gaussian sections along the ray
 * Uses insertion sort (efficient for small arrays)
 */
__device__ inline void sort_gaussian_sections(
    GaussianSection* sections,
    int num_sections
) {
    for (int i = 1; i < num_sections; i++) {
        GaussianSection key = sections[i];
        int j = i - 1;
        
        while (j >= 0 && sections[j].t_enter > key.t_enter) {
            sections[j + 1] = sections[j];
            j--;
        }
        sections[j + 1] = key;
    }
}

#endif // ANALYTIC_INTEGRATION_CUH


