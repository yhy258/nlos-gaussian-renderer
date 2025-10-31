#ifndef BBOX_COMPUTE_CUH
#define BBOX_COMPUTE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.cuh"

/**
 * Compute Gaussian AABBs directly on GPU
 * This is much faster than computing on CPU and transferring!
 */

/**
 * Compute AABB for a single Gaussian
 * 
 * @param mean Gaussian center [3]
 * @param scale Gaussian scales (after exp) [3]
 * @param quat Gaussian rotation (quaternion) [4]
 * @param sigma_scale Number of standard deviations (e.g., 3.0)
 * @param bbox_min [out] AABB minimum corner [3]
 * @param bbox_max [out] AABB maximum corner [3]
 */
__device__ inline void compute_gaussian_bbox(
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float sigma_scale,
    float3& bbox_min,
    float3& bbox_max
) {
    // Build rotation matrix
    float R[9];
    quat_to_rotmat(quat, R);
    
    // Compute AABB by transforming the corners of the scaled ellipsoid
    // The ellipsoid in local space is: ||x / scale|| <= sigma_scale
    // We need to find the AABB in world space
    
    // Simplified approach: Use axis-aligned extent after rotation
    // For each axis i, find max |R_i * (scale * sigma_scale)|
    
    float extent_x = sigma_scale * sqrtf(
        (R[0] * scale.x) * (R[0] * scale.x) +
        (R[1] * scale.y) * (R[1] * scale.y) +
        (R[2] * scale.z) * (R[2] * scale.z)
    );
    
    float extent_y = sigma_scale * sqrtf(
        (R[3] * scale.x) * (R[3] * scale.x) +
        (R[4] * scale.y) * (R[4] * scale.y) +
        (R[5] * scale.z) * (R[5] * scale.z)
    );
    
    float extent_z = sigma_scale * sqrtf(
        (R[6] * scale.x) * (R[6] * scale.x) +
        (R[7] * scale.y) * (R[7] * scale.y) +
        (R[8] * scale.z) * (R[8] * scale.z)
    );
    
    bbox_min = make_float3(
        mean.x - extent_x,
        mean.y - extent_y,
        mean.z - extent_z
    );
    
    bbox_max = make_float3(
        mean.x + extent_x,
        mean.y + extent_y,
        mean.z + extent_z
    );
}

/**
 * Kernel to compute AABBs for all Gaussians in parallel
 * Declaration only - definition is in bbox_compute.cu
 */
__global__ void compute_gaussian_bboxes_kernel(
    const float* __restrict__ gaussian_means,     // [N, 3]
    const float* __restrict__ gaussian_scales,    // [N, 3]
    const float* __restrict__ gaussian_rotations, // [N, 4]
    const int N_gaussians,
    const float scaling_modifier,
    const float sigma_scale,
    float* __restrict__ bboxes_out                // [N, 6] (min_x, min_y, min_z, max_x, max_y, max_z)
);

#endif // BBOX_COMPUTE_CUH

