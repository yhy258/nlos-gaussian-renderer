#include <torch/extension.h>
#include "ray_aabb.h"
#include "volume_renderer.h"
#include "volume_renderer_backward.h"

// Forward declarations for coordinate-based rendering
torch::Tensor compute_albedo_at_coords(
    const torch::Tensor& coords,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_scales,
    const torch::Tensor& gaussian_rotations,
    const torch::Tensor& gaussian_opacities,
    const torch::Tensor& gaussian_features,
    const torch::Tensor& camera_pos,
    const int active_sh_degree,
    const float scaling_modifier,
    const float sigma_threshold = 3.0f
);

torch::Tensor compute_density_at_coords(
    const torch::Tensor& coords,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_scales,
    const torch::Tensor& gaussian_rotations,
    const torch::Tensor& gaussian_opacities,
    const float scaling_modifier,
    const float sigma_threshold = 3.0f
);

// Forward declaration for analytic renderer
torch::Tensor render_rays_analytic(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const float t_min,
    const float t_max,
    const torch::Tensor& gaussian_filter,
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
    const float sigma_threshold,
    const std::string& rendering_type
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_gaussians_per_ray", &filter_gaussians_per_ray, 
          "Filter Gaussians per ray using AABB intersection (CUDA)");
    
    m.def("render_rays", &render_rays,
          "Per-ray volume rendering with transmittance (CUDA)");

    m.def("albedo_render_rays", &albedo_render_rays,
            "Per-ray albedo rendering (CUDA)");
    
    m.def("render_rays_analytic", &render_rays_analytic,
          "Section-based analytic volume rendering (CUDA)");

    m.def("render_rays_backward", &render_rays_backward,
          "Backward pass for volume rendering (CUDA)");
    
    m.def("compute_albedo_at_coords", &compute_albedo_at_coords,
          "Compute albedo at given 3D coordinates (CUDA)");
    
    m.def("compute_density_at_coords", &compute_density_at_coords,
          "Compute density at given 3D coordinates (CUDA)");
}

