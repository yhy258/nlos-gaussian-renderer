#ifndef RAY_AABB_H
#define RAY_AABB_H

#include <torch/extension.h>

// Ray-AABB intersection test and Gaussian filtering
torch::Tensor filter_gaussians_per_ray(
    const torch::Tensor& ray_origins,      // [N_rays, 3]
    const torch::Tensor& ray_directions,   // [N_rays, 3]
    const torch::Tensor& gaussian_means,   // [N_gaussians, 3]
    const torch::Tensor& gaussian_bboxes,  // [N_gaussians, 2, 3] (min, max)
    const float sigma_threshold            // typically 3.0
);

#endif // RAY_AABB_H


