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



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_gaussians_per_ray", &filter_gaussians_per_ray, 
          "Filter Gaussians per ray using AABB intersection (CUDA)");
    
    m.def("render_rays", &render_rays,
          "Per-ray volume rendering with transmittance (CUDA) - Default: Shared Memory");
    
    m.def("render_rays_shared", &render_rays_shared,
          "Per-ray volume rendering with SHARED MEMORY optimization (CUDA)");
    
    m.def("render_rays_global", &render_rays_global,
          "Per-ray volume rendering with GLOBAL MEMORY baseline (CUDA)");
    
    m.def("render_rays_backward", &render_rays_backward,
          "Backward pass for volume rendering (CUDA)");
    
    m.def("compute_albedo_at_coords", &compute_albedo_at_coords,
          "Compute albedo at given 3D coordinates (CUDA)");
    
    m.def("compute_density_at_coords", &compute_density_at_coords,
          "Compute density at given 3D coordinates (CUDA)");
}

