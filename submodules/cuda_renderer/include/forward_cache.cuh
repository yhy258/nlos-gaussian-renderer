#ifndef FORWARD_CACHE_CUH
#define FORWARD_CACHE_CUH

#include <cuda_runtime.h>

/**
 * Forward Cache Structure
 * 
 * Stores intermediate results from forward pass for backward pass reuse.
 * This eliminates expensive recomputation in backward pass.
 * 
 * Memory layout: [N_rays, N_samples, MAX_GAUSSIANS_PER_RAY]
 */

struct ForwardCache {
    float pdf;         // Gaussian PDF value
    float opacity;     // Sigmoid(opacity_logit)
    float rho;         // Albedo from SH evaluation
    float contrib;     // pdf * opacity
    float alpha;       // 1 - exp(-contrib) [for occlusion mode]
    
    // Gaussian parameters (avoid reloading)
    float mean_x, mean_y, mean_z;
    float scale_x, scale_y, scale_z;
    float quat_x, quat_y, quat_z, quat_w;
};

// Compact version (less memory, recompute some values)
struct ForwardCacheCompact {
    float pdf;
    float opacity;
    float rho;
    // Gaussian params can be reloaded (trade-off)
};

#endif // FORWARD_CACHE_CUH

