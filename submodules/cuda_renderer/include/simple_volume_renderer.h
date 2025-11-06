#ifndef SIMPLE_VOLUME_RENDERER_H
#define SIMPLE_VOLUME_RENDERER_H

#include <torch/extension.h>
#include <tuple>

// Per-ray volume rendering with transmittance (default: shared memory + forward cache)
// Returns: (rho_density, density, transmittance, gaussian_bboxes, gaussian_filter, forward_cache)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> simple_render_rays(
    const torch::Tensor& ray_origins,        // [N_rays, 3]
    const torch::Tensor& ray_directions,     // [N_rays, 3]
    const torch::Tensor& t_samples,          // [N_samples] - ray parameter values (r values)
    const torch::Tensor& gaussian_means,     // [N_gaussians, 3]
    const torch::Tensor& gaussian_scales,    // [N_gaussians, 3]
    const torch::Tensor& gaussian_rotations, // [N_gaussians, 4] quaternions
    const torch::Tensor& gaussian_opacities, // [N_gaussians, 1]
    const torch::Tensor& gaussian_features,  // [N_gaussians, K] - SH features
    const torch::Tensor& camera_pos,         // [3] - for view-dependent effects
    const int active_sh_degree,
    const float c,                           // speed of light
    const float deltaT,                      // time interval
    const float scaling_modifier,
    const bool use_occlusion
);

// OPTIMIZED: Shared memory version with forward cache
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> simple_render_rays_shared(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const torch::Tensor& t_samples,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_scales,
    const torch::Tensor& gaussian_rotations,
    const torch::Tensor& gaussian_opacities,
    const torch::Tensor& gaussian_features,
    const torch::Tensor& camera_pos,
    const int active_sh_degree,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion
);

// BASELINE: Global memory version (for comparison)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> simple_render_rays_global(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const torch::Tensor& t_samples,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_scales,
    const torch::Tensor& gaussian_rotations,
    const torch::Tensor& gaussian_opacities,
    const torch::Tensor& gaussian_features,
    const torch::Tensor& camera_pos,
    const int active_sh_degree,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion
);

#endif // SIMPLE_VOLUME_RENDERER_H


