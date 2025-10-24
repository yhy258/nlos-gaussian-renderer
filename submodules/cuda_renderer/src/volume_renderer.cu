#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "bbox_compute.cuh"
#include "spherical_harmonics.cuh"
#include "ray_aabb.h"
#include <tuple>

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256


// Unified volume rendering kernel with integrated transmittance
// Single-pass rendering for efficiency and correctness
__global__ void volume_render_kernel(
    const float* __restrict__ ray_origins,        // [N_rays, 3]
    const float* __restrict__ ray_directions,     // [N_rays, 3]
    const float* __restrict__ t_samples,          // [N_samples]
    const int* __restrict__ gaussian_filter,      // [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    const float* __restrict__ gaussian_means,     // [N_gaussians, 3]
    const float* __restrict__ gaussian_scales,    // [N_gaussians, 3]
    const float* __restrict__ gaussian_rotations, // [N_gaussians, 4]
    const float* __restrict__ gaussian_opacities, // [N_gaussians, 1]
    const float* __restrict__ gaussian_features,  // [N_gaussians, K]
    const float* __restrict__ camera_pos,         // [3]
    const int N_rays,
    const int N_samples,
    const int N_gaussians,
    const int active_sh_degree,
    const int sh_dim,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion,
    const int rendering_type,  // 0: netf, 1: nlos-neus
    float* __restrict__ rho_density_out,          // [N_rays, N_samples] - FINAL OUTPUT
    float* __restrict__ density_out,              // [N_rays, N_samples] - for debugging
    float* __restrict__ transmittance_out         // [N_rays, N_samples] - for debugging
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
    
    float3 cam_pos = make_float3(camera_pos[0], camera_pos[1], camera_pos[2]);
    
    // Get filtered Gaussians for this ray
    int num_gaussians = gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1)];
    const int* valid_gaussian_indices = &gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1) + 1];
    
    // Initialize transmittance
    float T = 1.0f;

    // March along ray (sequential - correct volume rendering!)
    for (int s = 0; s < N_samples; s++) {
        float t = t_samples[s];
        float3 pos = ray_o + ray_d * t;
        
        // Accumulate contributions from ALL Gaussians at this sample
        float density = 0.0f;
        float weighted_radiance = 0.0f;  // rho * density
        float weighted_alphas = 0.0f;
        // Sum over all filtered Gaussians


        // NEW
        if (use_occlusion) {
            for (int i = 0; i < num_gaussians; i++) {
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
                
                float opacity = 1.0f / (1.0f + expf(-gaussian_opacities[g])); // sigmoid
                float pdf = eval_gaussian_pdf(pos, mean, scale, quat);
                
                // View-dependent albedo (SH evaluation)
                float3 view_dir = normalize(mean - cam_pos);
                float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
                rho = fmaxf(rho + 0.5f, 0.0f);  // clamp_min(sh2rho + 0.5, 0.0)


                float contrib = pdf * opacity;
                float alpha = 1.0f - expf(-contrib * c * deltaT);
                weighted_alphas += alpha * rho;
                density += contrib;
            }

            int out_idx = ray_idx * N_samples + s;
            density_out[out_idx] = density;
            transmittance_out[out_idx] = T;


        rho_density_out[out_idx] = T * weighted_alphas;
        T *= expf(-density * c * deltaT);
        
        // Early ray termination for efficiency
        if (T < 1e-4f) {
            // Fill remaining samples with zero
            for (int s_rest = s + 1; s_rest < N_samples; s_rest++) {
                int idx_rest = ray_idx * N_samples + s_rest;
                rho_density_out[idx_rest] = 0.0f;
                density_out[idx_rest] = 0.0f;
                transmittance_out[idx_rest] = 0.0f;
            }
            break;
        }
    } else {
        for (int i = 0; i < num_gaussians; i++) {
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
            
            float opacity = 1.0f / (1.0f + expf(-gaussian_opacities[g])); // sigmoid
            
            // Evaluate Gaussian PDF
            float pdf = eval_gaussian_pdf(pos, mean, scale, quat);
            
            // View-dependent albedo (SH evaluation)
            float3 view_dir = normalize(mean - cam_pos);
            float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
            rho = fmaxf(rho + 0.5f, 0.0f);  // clamp_min(sh2rho + 0.5, 0.0)
            
            // Accumulate
            float contrib = pdf * opacity;
            density += contrib;
            weighted_radiance += contrib * rho;
        }
        int out_idx = ray_idx * N_samples + s;
        density_out[out_idx] = density;
        transmittance_out[out_idx] = T;

        rho_density_out[out_idx] = weighted_radiance * c * deltaT;
    }
}

// NOTE: compute_transmittance_kernel is now integrated into volume_render_kernel
// for single-pass efficiency and correctness. Keeping this comment for reference.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_rays(
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
    const bool use_occlusion,
    const std::string& rendering_type
) {
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(t_samples);
    CHECK_INPUT(gaussian_means);
    CHECK_INPUT(gaussian_scales);
    CHECK_INPUT(gaussian_rotations);
    CHECK_INPUT(gaussian_opacities);
    CHECK_INPUT(gaussian_features);
    
    const int N_rays = ray_origins.size(0);
    const int N_samples = t_samples.size(0);
    const int N_gaussians = gaussian_means.size(0);
    const int sh_dim = gaussian_features.size(1);
    
    // Compute Gaussian bounding boxes on GPU (much faster!)
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(ray_origins.device());
    torch::Tensor gaussian_bboxes = torch::empty({N_gaussians, 6}, float_options);
    
    {
        const int blocks = (N_gaussians + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_gaussian_bboxes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            gaussian_means.data_ptr<float>(),
            gaussian_scales.data_ptr<float>(),
            gaussian_rotations.data_ptr<float>(),
            N_gaussians,
            scaling_modifier,
            3.0f,  // sigma_threshold
            gaussian_bboxes.data_ptr<float>()
        );
        cudaDeviceSynchronize();
    }
    
    // Filter gaussians per ray using computed bboxes
    torch::Tensor gaussian_filter = filter_gaussians_per_ray(
        ray_origins,
        ray_directions,
        gaussian_means,
        gaussian_bboxes,  // Keep as [N, 6] flat layout
        3.0f  // sigma_threshold
    );

    // When testing this CUDA based code, I want to use the simplified version (all gaussians for all rays)
    // ** For training and advanced testing, I will use the bbox computation + filtering. (uncomment the code above) **
    // Allocate temporary filter (simplified: all gaussians for all rays)
    // auto options = torch::TensorOptions().dtype(torch::kInt32).device(ray_origins.device());
    // torch::Tensor gaussian_filter = torch::zeros({N_rays, MAX_GAUSSIANS_PER_RAY + 1}, options);
    
    // // Simplified version (all gaussians for all rays)
    // for (int i = 0; i < N_rays; i++) {
    //     gaussian_filter[i][0] = N_gaussians;
    //     for (int j = 0; j < N_gaussians && j < MAX_GAUSSIANS_PER_RAY; j++) {
    //         gaussian_filter[i][j + 1] = j;
    //     }
    // }
    
    
    // Allocate output tensors (reuse float_options from above)
    torch::Tensor rho_density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor transmittance = torch::zeros({N_rays, N_samples}, float_options);
    
    int render_type = (rendering_type == "netf") ? 0 : 1;
    
    // Launch rendering kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    volume_render_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        ray_origins.data_ptr<float>(),
        ray_directions.data_ptr<float>(),
        t_samples.data_ptr<float>(),
        gaussian_filter.data_ptr<int>(),
        gaussian_means.data_ptr<float>(),
        gaussian_scales.data_ptr<float>(),
        gaussian_rotations.data_ptr<float>(),
        gaussian_opacities.data_ptr<float>(),
        gaussian_features.data_ptr<float>(),
        camera_pos.data_ptr<float>(),
        N_rays,
        N_samples,
        N_gaussians,
        active_sh_degree,
        sh_dim,
        c,
        deltaT,
        scaling_modifier,
        use_occlusion,
        render_type,
        rho_density.data_ptr<float>(),
        density.data_ptr<float>(),
        transmittance.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    // Single-pass rendering complete! 
    // Transmittance is now computed inside volume_render_kernel
    // No second pass needed - more efficient and correct!
    
    return std::make_tuple(rho_density, density, transmittance);
}


