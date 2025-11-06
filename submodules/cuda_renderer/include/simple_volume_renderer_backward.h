#ifndef SIMPLE_VOLUME_RENDERER_BACKWARD_H
#define SIMPLE_VOLUME_RENDERER_BACKWARD_H

#include <torch/extension.h>
#include <tuple>

/**
 * Backward pass for volume rendering
 * 
 * Computes gradients of loss w.r.t. Gaussian parameters
 * given gradients w.r.t. outputs (rho_density, density, transmittance)
 * 
 * This is the adjoint/backward operator for the simple_render_rays forward pass.
 */
std::tuple<
    torch::Tensor,  // grad_gaussian_means
    torch::Tensor,  // grad_gaussian_scales
    torch::Tensor,  // grad_gaussian_rotations
    torch::Tensor,  // grad_gaussian_opacities
    torch::Tensor   // grad_gaussian_features
> simple_render_rays_backward(
    // Forward pass outputs (for recomputation if needed)
    const torch::Tensor& rho_density,            // [N_rays, N_samples
    
    // Gradient inputs (from upstream)
    const torch::Tensor& grad_rho_density,       // [N_rays, N_samples]
    const torch::Tensor& grad_density,           // [N_rays, N_samples]
    const torch::Tensor& grad_transmittance,     // [N_rays, N_samples]
    
    // Forward pass inputs (needed for recomputation)
    const torch::Tensor& ray_origins,            // [N_rays, 3]
    const torch::Tensor& ray_directions,         // [N_rays, 3]
    const torch::Tensor& t_samples,              // [N_samples]
    const torch::Tensor& gaussian_filter,        // [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    const torch::Tensor& gaussian_means,         // [N_gaussians, 3]
    const torch::Tensor& gaussian_scales,        // [N_gaussians, 3]
    const torch::Tensor& gaussian_rotations,     // [N_gaussians, 4]
    const torch::Tensor& gaussian_opacities,     // [N_gaussians, 1]
    const torch::Tensor& gaussian_features,      // [N_gaussians, K]
    const torch::Tensor& camera_pos,             // [3]
    
    // Forward pass outputs (for recomputation)
    const torch::Tensor& forward_cache,          // [N_rays, N_samples, MAX_GAUSSIANS_PER_RAY] - NEW!
    
    // Hyperparameters
    const int active_sh_degree,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion
);

#endif // SIMPLE_VOLUME_RENDERER_BACKWARD_H

