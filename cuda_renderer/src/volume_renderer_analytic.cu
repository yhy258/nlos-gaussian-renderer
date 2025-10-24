#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "analytic_integration.cuh"
#include <tuple>
#include "spherical_harmonics.cuh"

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256
#define MAX_SECTIONS_PER_RAY 128

/**
 * Analytic Volume Rendering Kernel
 * Based on "Don't Splat your Gaussians" section-based rendering
 * 
 * This kernel:
 * 1. Computes Gaussian sections along each ray
 * 2. Sorts sections by entry point
 * 3. Analytically integrates transmittance through each section
 * 4. Accumulates color weighted by transmittance
 */
__global__ void volume_render_analytic_kernel(
    const float* __restrict__ ray_origins,        // [N_rays, 3]
    const float* __restrict__ ray_directions,     // [N_rays, 3]
    const float t_min,                            // Start of ray sampling
    const float t_max,                            // End of ray sampling
    const int* __restrict__ gaussian_filter,      // [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    const float* __restrict__ gaussian_means,     // [N_gaussians, 3]
    const float* __restrict__ gaussian_scales,    // [N_gaussians, 3]
    const float* __restrict__ gaussian_rotations, // [N_gaussians, 4]
    const float* __restrict__ gaussian_opacities, // [N_gaussians, 1]
    const float* __restrict__ gaussian_features,  // [N_gaussians, K]
    const float* __restrict__ camera_pos,         // [3]
    const int N_rays,
    const int N_gaussians,
    const int active_sh_degree,
    const int sh_dim,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const float sigma_threshold,
    const int rendering_type,  // 0: netf, 1: nlos-neus
    float* __restrict__ histogram_out             // [N_rays]
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= N_rays) return;
    
    // Load ray
    float3 ray_o = make_float3(
        ray_origins[ray_idx * 3 + 0],
        ray_origins[ray_idx * 3 + 1],
        ray_origins[ray_idx * 3 + 2]
    );
    
    float3 ray_d = make_float3(
        ray_directions[ray_idx * 3 + 0],
        ray_directions[ray_idx * 3 + 1],
        ray_directions[ray_idx * 3 + 2]
    );
    
    float3 cam_pos_3d = make_float3(camera_pos[0], camera_pos[1], camera_pos[2]);
    
    // Get filtered Gaussians for this ray
    int num_gaussians = gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1)];
    const int* valid_gaussian_indices = &gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1) + 1];
    
    // Compute sections for each Gaussian
    GaussianSection sections[MAX_SECTIONS_PER_RAY];
    int num_sections = 0;
    
    for (int i = 0; i < num_gaussians && num_sections < MAX_SECTIONS_PER_RAY; i++) {
        int g = valid_gaussian_indices[i];
        if (g < 0 || g >= N_gaussians) continue;
        
        // Load Gaussian parameters
        float3 mean = make_float3(
            gaussian_means[g * 3 + 0],
            gaussian_means[g * 3 + 1],
            gaussian_means[g * 3 + 2]
        );
        
        float3 scale = make_float3(
            expf(gaussian_scales[g * 3 + 0]) * scaling_modifier,
            expf(gaussian_scales[g * 3 + 1]) * scaling_modifier,
            expf(gaussian_scales[g * 3 + 2]) * scaling_modifier
        );
        
        float4 quat = make_float4(
            gaussian_rotations[g * 4 + 0],
            gaussian_rotations[g * 4 + 1],
            gaussian_rotations[g * 4 + 2],
            gaussian_rotations[g * 4 + 3]
        );
        
        float t_enter, t_exit, t_peak;
        if (compute_gaussian_section(ray_o, ray_d, mean, scale, quat, 
                                     sigma_threshold, t_enter, t_exit, t_peak)) {
            // Clip to ray bounds
            t_enter = fmaxf(t_enter, t_min);
            t_exit = fminf(t_exit, t_max);
            
            if (t_enter < t_exit) {
                sections[num_sections].gaussian_idx = g;
                sections[num_sections].t_enter = t_enter;
                sections[num_sections].t_exit = t_exit;
                sections[num_sections].t_peak = t_peak;
                num_sections++;
            }
        }
    }
    
    // Sort sections by entry point
    sort_gaussian_sections(sections, num_sections);
    
    // Analytic integration through sections
    float transmittance = 1.0f;
    float accumulated_radiance = 0.0f;
    
    for (int s = 0; s < num_sections; s++) {
        int g = sections[s].gaussian_idx;
        float t_enter = sections[s].t_enter;
        float t_exit = sections[s].t_exit;
        
        // Load Gaussian parameters
        float3 mean = make_float3(
            gaussian_means[g * 3 + 0],
            gaussian_means[g * 3 + 1],
            gaussian_means[g * 3 + 2]
        );
        
        float3 scale = make_float3(
            expf(gaussian_scales[g * 3 + 0]) * scaling_modifier,
            expf(gaussian_scales[g * 3 + 1]) * scaling_modifier,
            expf(gaussian_scales[g * 3 + 2]) * scaling_modifier
        );
        
        float4 quat = make_float4(
            gaussian_rotations[g * 4 + 0],
            gaussian_rotations[g * 4 + 1],
            gaussian_rotations[g * 4 + 2],
            gaussian_rotations[g * 4 + 3]
        );
        
        float opacity = 1.0f / (1.0f + expf(-gaussian_opacities[g]));
        
        // View-dependent albedo
        float3 view_dir = normalize(mean - cam_pos_3d);
        float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
        rho = fmaxf(rho + 0.5f, 0.0f);
        
        // Analytic transmittance through this section
        float tau = compute_analytic_transmittance(
            ray_o, ray_d, t_enter, t_exit,
            mean, scale, quat, opacity
        );
        
        // Accumulate radiance
        float section_transmittance = expf(-tau);
        float alpha = 1.0f - section_transmittance;
        
        accumulated_radiance += transmittance * alpha * rho;
        
        // Update transmittance
        transmittance *= section_transmittance;
        
        // Early termination
        if (transmittance < 1e-4f) break;
    }
    
    histogram_out[ray_idx] = accumulated_radiance;
}

/**
 * Python interface for analytic rendering
 */
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
) {
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(gaussian_means);
    
    const int N_rays = ray_origins.size(0);
    const int N_gaussians = gaussian_means.size(0);
    const int sh_dim = gaussian_features.size(1);
    
    // Allocate output
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(ray_origins.device());
    torch::Tensor histogram = torch::zeros({N_rays}, options);
    
    int render_type = (rendering_type == "netf") ? 0 : 1;
    
    // Launch kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    volume_render_analytic_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        ray_origins.data_ptr<float>(),
        ray_directions.data_ptr<float>(),
        t_min,
        t_max,
        gaussian_filter.data_ptr<int>(),
        gaussian_means.data_ptr<float>(),
        gaussian_scales.data_ptr<float>(),
        gaussian_rotations.data_ptr<float>(),
        gaussian_opacities.data_ptr<float>(),
        gaussian_features.data_ptr<float>(),
        camera_pos.data_ptr<float>(),
        N_rays,
        N_gaussians,
        active_sh_degree,
        sh_dim,
        c,
        deltaT,
        scaling_modifier,
        sigma_threshold,
        render_type,
        histogram.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    return histogram;
}


