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

// COMPACT VERSION: Only cache expensive computations
// Memory: 3 floats = 12 bytes per entry (5x smaller!)
// Trade-off: Reload Gaussian params in backward (cheap memory access)
struct ForwardCache {
    float pdf;         // Gaussian PDF value (EXPENSIVE: eval_gaussian_pdf)
    float opacity;     // Sigmoid(opacity_logit) (cheap but needed)
    float rho;         // Albedo from SH evaluation (EXPENSIVE: eval_sh)
    
    // NOT stored (reload in backward):
    // - Gaussian params (mean, scale, quat): Cheap memory loads
    // - contrib, alpha: Can recompute from pdf/opacity (trivial)
};

// Full version (DISABLED due to OOM):
// 15 floats = 60 bytes → 20GB for 4096×300×256
// Compact version: 3 floats = 12 bytes → 4GB for 4096×300×256

#endif // FORWARD_CACHE_CUH

