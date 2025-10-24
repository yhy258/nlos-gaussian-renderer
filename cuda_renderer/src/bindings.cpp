#include <torch/extension.h>
#include "ray_aabb.h"
#include "volume_renderer.h"

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
    
    m.def("render_rays_analytic", &render_rays_analytic,
          "Section-based analytic volume rendering (CUDA)");
}

