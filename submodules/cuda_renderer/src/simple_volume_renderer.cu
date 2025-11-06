#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "bbox_compute.cuh"
#include "spherical_harmonics.cuh"
#include "ray_aabb.h"
#include "forward_cache.cuh"
#include <tuple>

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256
#define MAX_T_SAMPLES_SHARED 512  // Shared memory limit for t_samples

// ============================================================
// BASELINE: Global Memory Version (for comparison)
// ============================================================
__global__ void simple_volume_render_kernel_global(
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
    float* __restrict__ rho_density_out,          // [N_rays, N_samples]
    float* __restrict__ density_out,              // [N_rays, N_samples]
    float* __restrict__ transmittance_out         // [N_rays, N_samples]
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= N_rays) return;
    
    // Load ray (from GLOBAL MEMORY)
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
        float t = t_samples[s];  // GLOBAL MEMORY ACCESS
        float3 pos = ray_o + ray_d * t;
        
        // Accumulate contributions from ALL Gaussians at this sample
        float density = 0.0f;
        float weighted_radiance = 0.0f;
        float weighted_alphas = 0.0f;

        if (use_occlusion) {
            break;
        } else {
            for (int i = 0; i < num_gaussians; i++) {
                int g = valid_gaussian_indices[i];
                if (g < 0 || g >= N_gaussians) continue;
                
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
                float pdf = eval_gaussian_pdf(pos, mean, scale, quat);
                
                float3 view_dir = normalize(mean - cam_pos);
                float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
                rho = fmaxf(rho + 0.5f, 0.0f);
                
                float contrib = pdf * opacity;
                density += contrib;
                weighted_radiance += contrib * rho;
            }
            int out_idx = ray_idx * N_samples + s;
            density_out[out_idx] = density;
            transmittance_out[out_idx] = T;

            rho_density_out[out_idx] = weighted_radiance / (density + 1e-8f);
        }
    }
}

// ============================================================
// OPTIMIZED: Shared Memory Version
// ============================================================
__global__ void volume_render_kernel_shared(
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
    float* __restrict__ rho_density_out,          // [N_rays, N_samples] - FINAL OUTPUT
    float* __restrict__ density_out,              // [N_rays, N_samples] - for debugging
    float* __restrict__ transmittance_out,        // [N_rays, N_samples] - for debugging
    ForwardCache* __restrict__ cache_out          // [N_rays, N_samples, MAX_GAUSSIANS_PER_RAY] - NEW!
) {
    // ============================================================
    // SHARED MEMORY OPTIMIZATION
    // ============================================================
    // Shared memory for t_samples (all rays use the same samples)
    __shared__ float s_t_samples[MAX_T_SAMPLES_SHARED];
    
    // Shared memory for camera position (all rays use the same camera)
    __shared__ float3 s_cam_pos;
    
    // Cooperative loading: all threads in block help load t_samples
    int tid = threadIdx.x;
    int load_iterations = (N_samples + blockDim.x - 1) / blockDim.x;
    
    for (int iter = 0; iter < load_iterations; iter++) {
        int sample_idx = tid + iter * blockDim.x;
        if (sample_idx < N_samples && sample_idx < MAX_T_SAMPLES_SHARED) {
            s_t_samples[sample_idx] = t_samples[sample_idx];
        }
    }
    
    // First thread loads camera position
    if (tid == 0) {
        s_cam_pos = make_float3(camera_pos[0], camera_pos[1], camera_pos[2]);
    }
    
    // Synchronize to ensure all shared memory is loaded
    __syncthreads();
    
    // ============================================================
    // PER-RAY COMPUTATION
    // ============================================================
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= N_rays) return;
    
    // Load ray from global memory (coalesced access)
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
    
    // Use cached camera position from shared memory
    float3 cam_pos = s_cam_pos;
    
    // Get filtered Gaussians for this ray
    int num_gaussians = gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1)];
    const int* valid_gaussian_indices = &gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1) + 1];
    
    // Initialize transmittance
    float T = 1.0f;

    // March along ray (sequential - correct volume rendering!)
    // Use shared memory for t_samples (faster access)
    for (int s = 0; s < N_samples; s++) {
        // Read from shared memory instead of global memory
        float t = (s < MAX_T_SAMPLES_SHARED) ? s_t_samples[s] : t_samples[s];
        float3 pos = ray_o + ray_d * t;
        
        // Accumulate contributions from ALL Gaussians at this sample
        float density = 0.0f;
        float weighted_radiance = 0.0f;  // rho * density
        float weighted_alphas = 0.0f;
        // Sum over all filtered Gaussians


        // NEW
        if (use_occlusion) {
            break;
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
                
                // ============================================================
                // FORWARD CACHE: Save expensive computations (COMPACT!)
                // ============================================================
                if (cache_out != nullptr) {
                    int cache_idx = (ray_idx * N_samples + s) * MAX_GAUSSIANS_PER_RAY + i;
                    cache_out[cache_idx].pdf = pdf;
                    cache_out[cache_idx].opacity = opacity;
                    cache_out[cache_idx].rho = rho;
                }
            }
            int out_idx = ray_idx * N_samples + s;
            density_out[out_idx] = density;
            transmittance_out[out_idx] = T;
            rho_density_out[out_idx] = weighted_radiance / (density + 1e-8f);
        }
    }
}
// NOTE: compute_transmittance_kernel is now integrated into volume_render_kernel
// for single-pass efficiency and correctness. Keeping this comment for reference.

// ============================================================
// Wrapper Functions
// ============================================================

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
    
    // Compute Gaussian bounding boxes
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
            3.0f,
            gaussian_bboxes.data_ptr<float>()
        );
        cudaDeviceSynchronize();
    }
    
    // Filter gaussians per ray
    torch::Tensor gaussian_filter = filter_gaussians_per_ray(
        ray_origins,
        ray_directions,
        gaussian_means,
        gaussian_bboxes,
        3.0f
    );
    
    // Allocate output tensors
    torch::Tensor rho_density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor transmittance = torch::zeros({N_rays, N_samples}, float_options);
    
    // ============================================================
    // FORWARD CACHE: Allocate cache for backward pass
    // Size: [N_rays, N_samples, MAX_GAUSSIANS_PER_RAY] × sizeof(ForwardCache)
    // ============================================================
    const size_t cache_size = static_cast<size_t>(N_rays) * N_samples * MAX_GAUSSIANS_PER_RAY;
    auto cache_options = torch::TensorOptions()
        .dtype(torch::kUInt8)  // Raw bytes
        .device(ray_origins.device());
    
    torch::Tensor forward_cache = torch::empty({static_cast<int64_t>(cache_size * sizeof(ForwardCache))}, cache_options);
    
    // Launch SHARED MEMORY kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    volume_render_kernel_shared<<<blocks, THREADS_PER_BLOCK>>>(
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
        rho_density.data_ptr<float>(),
        density.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        reinterpret_cast<ForwardCache*>(forward_cache.data_ptr<uint8_t>())  // NEW!
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(rho_density, density, transmittance, gaussian_bboxes, gaussian_filter, forward_cache);
}

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
    
    // Compute Gaussian bounding boxes
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
            3.0f,
            gaussian_bboxes.data_ptr<float>()
        );
        cudaDeviceSynchronize();
    }
    
    // Filter gaussians per ray
    torch::Tensor gaussian_filter = filter_gaussians_per_ray(
        ray_origins,
        ray_directions,
        gaussian_means,
        gaussian_bboxes,
        3.0f
    );
    
    // Allocate output tensors
    torch::Tensor rho_density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor density = torch::zeros({N_rays, N_samples}, float_options);
    torch::Tensor transmittance = torch::zeros({N_rays, N_samples}, float_options);
    
    // Launch GLOBAL MEMORY kernel (baseline)
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    simple_volume_render_kernel_global<<<blocks, THREADS_PER_BLOCK>>>(
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
        rho_density.data_ptr<float>(),
        density.data_ptr<float>(),
        transmittance.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(rho_density, density, transmittance, gaussian_bboxes, gaussian_filter);
}

// Default: use shared memory version WITH forward cache
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> simple_render_rays(
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
) {
    // Default behavior: use optimized shared memory version with forward caching
    return simple_render_rays_shared(
        ray_origins, ray_directions, t_samples,
        gaussian_means, gaussian_scales, gaussian_rotations,
        gaussian_opacities, gaussian_features, camera_pos,
        active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
    );
}


