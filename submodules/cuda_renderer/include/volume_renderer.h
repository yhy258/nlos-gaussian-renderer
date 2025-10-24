#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include <torch/extension.h>
#include <tuple>

// Per-ray volume rendering with transmittance
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_rays(
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
    const bool use_occlusion,
    const std::string& rendering_type        // "netf" or "nlos-neus"
);

#endif // VOLUME_RENDERER_H


