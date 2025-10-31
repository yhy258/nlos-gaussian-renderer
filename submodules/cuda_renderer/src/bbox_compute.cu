#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bbox_compute.cuh"

#define THREADS_PER_BLOCK 256

/**
 * Kernel to compute AABBs for all Gaussians in parallel
 * Implementation - definition moved from header to avoid multiple definition errors
 */
__global__ void compute_gaussian_bboxes_kernel(
    const float* __restrict__ gaussian_means,     // [N, 3]
    const float* __restrict__ gaussian_scales,    // [N, 3]
    const float* __restrict__ gaussian_rotations, // [N, 4]
    const int N_gaussians,
    const float scaling_modifier,
    const float sigma_scale,
    float* __restrict__ bboxes_out                // [N, 6] (min_x, min_y, min_z, max_x, max_y, max_z)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N_gaussians) return;
    
    // Load Gaussian parameters
    float3 mean = make_float3(
        gaussian_means[idx * 3 + 0],
        gaussian_means[idx * 3 + 1],
        gaussian_means[idx * 3 + 2]
    );
    
    float3 scale = make_float3(
        expf(gaussian_scales[idx * 3 + 0]) * scaling_modifier,
        expf(gaussian_scales[idx * 3 + 1]) * scaling_modifier,
        expf(gaussian_scales[idx * 3 + 2]) * scaling_modifier
    );
    
    float4 quat = make_float4(
        gaussian_rotations[idx * 4 + 0],
        gaussian_rotations[idx * 4 + 1],
        gaussian_rotations[idx * 4 + 2],
        gaussian_rotations[idx * 4 + 3]
    );
    
    // Compute AABB
    float3 bbox_min, bbox_max;
    compute_gaussian_bbox(mean, scale, quat, sigma_scale, bbox_min, bbox_max);
    
    // Store result
    bboxes_out[idx * 6 + 0] = bbox_min.x;
    bboxes_out[idx * 6 + 1] = bbox_min.y;
    bboxes_out[idx * 6 + 2] = bbox_min.z;
    bboxes_out[idx * 6 + 3] = bbox_max.x;
    bboxes_out[idx * 6 + 4] = bbox_max.y;
    bboxes_out[idx * 6 + 5] = bbox_max.z;
}

