#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "backward_utils.cuh"
#include "spherical_harmonics.cuh"
#include "forward_cache.cuh"
#include "simple_volume_renderer_backward.h"

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256
#define MAX_T_SAMPLES_SHARED 512  // Shared memory limit for t_samples

/**
 * Simple Volume Rendering Backward Pass Kernel
 * 
 * This kernel computes gradients w.r.t. Gaussian parameters
 * given gradients w.r.t. the output (rho_density).
 * 
 * Key algorithmic features:
 * 1. Processes rays in parallel (one ray per thread/block)
 * 2. Marches BACKWARD along each ray (reverse order) to handle transmittance dependencies
 * 3. Accumulates gradients using atomicAdd for thread safety
 * 
 */
__global__ void simple_volume_render_backward_kernel(
    // Gradient inputs (from upstream loss)
    const float* __restrict__ grad_rho_density,   // [N_rays, N_samples]
    
    // Forward pass inputs (needed for gradient computation)
    const float* __restrict__ ray_origins,        // [N_rays, 3]
    const float* __restrict__ ray_directions,     // [N_rays, 3]
    const float* __restrict__ t_samples,          // [N_samples]
    const int* __restrict__ gaussian_filter,      // [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    const float* __restrict__ gaussian_means,     // [N_gaussians, 3]
    const float* __restrict__ gaussian_scales,    // [N_gaussians, 3] (log scale)
    const float* __restrict__ gaussian_rotations, // [N_gaussians, 4] (quaternion)
    const float* __restrict__ gaussian_opacities, // [N_gaussians, 1] (logit)
    const float* __restrict__ gaussian_features,  // [N_gaussians, K] (SH coeffs)
    const float* __restrict__ camera_pos,         // [3]

    const float* __restrict__ density_fwd,        // [N_rays, N_samples]
    const float* __restrict__ transmittance_fwd,  // [N_rays, N_samples]
    const ForwardCache* __restrict__ cache_in,    // [N_rays, N_samples, MAX_GAUSSIANS_PER_RAY]
    
    // Dimensions
    const int N_rays,
    const int N_samples,
    const int N_gaussians,
    const int active_sh_degree,
    const int sh_dim,
    
    // Hyperparameters
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion,
    
    // Gradient outputs (accumulated via atomicAdd)
    float* __restrict__ grad_means,       // [N_gaussians, 3]
    float* __restrict__ grad_scales,  // [N_gaussians, 3]
    float* __restrict__ grad_rotations,   // [N_gaussians, 4]
    float* __restrict__ grad_opacities,   // [N_gaussians, 1]
    float* __restrict__ grad_features     // [N_gaussians, K]
) {
    // ============================================================
    // SHARED MEMORY OPTIMIZATION
    // ============================================================
    __shared__ float s_t_samples[MAX_T_SAMPLES_SHARED];
    __shared__ float3 s_cam_pos;
    
    int tid = threadIdx.x;
    int load_iterations = (N_samples + blockDim.x - 1) / blockDim.x;
    
    // Cooperative loading of t_samples
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
    
    __syncthreads();
    
    // ============================================================
    // PER-RAY BACKWARD COMPUTATION
    // ============================================================
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= N_rays) return;
    
    // Load ray (coalesced access)
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
    
    // ============================================================
    // OPTIMIZATION: Local gradient accumulation
    // Instead of atomicAdd on every sample, accumulate locally first
    // ============================================================
    
    // Allocate local gradient buffers for each Gaussian
    // We only allocate for filtered Gaussians (not all N_gaussians)
    float local_grad_means[MAX_GAUSSIANS_PER_RAY * 3];
    float local_grad_scales[MAX_GAUSSIANS_PER_RAY * 3];
    float local_grad_rotations[MAX_GAUSSIANS_PER_RAY * 4];
    float local_grad_opacities[MAX_GAUSSIANS_PER_RAY];
    float local_grad_features[MAX_GAUSSIANS_PER_RAY * 16];  // Assuming max SH degree 3

    // Store per-Gaussian intermediate values for gradient computation
    float pdf_values[MAX_GAUSSIANS_PER_RAY];
    float opacity_values[MAX_GAUSSIANS_PER_RAY];
    float rho_values[MAX_GAUSSIANS_PER_RAY];
    float alpha_values[MAX_GAUSSIANS_PER_RAY];
    float contrib_values[MAX_GAUSSIANS_PER_RAY];
    
    // Initialize to zero
    for (int i = 0; i < num_gaussians; i++) {
        local_grad_means[i * 3 + 0] = 0.0f;
        local_grad_means[i * 3 + 1] = 0.0f;
        local_grad_means[i * 3 + 2] = 0.0f;
        
        local_grad_scales[i * 3 + 0] = 0.0f;
        local_grad_scales[i * 3 + 1] = 0.0f;
        local_grad_scales[i * 3 + 2] = 0.0f;
        
        local_grad_rotations[i * 4 + 0] = 0.0f;
        local_grad_rotations[i * 4 + 1] = 0.0f;
        local_grad_rotations[i * 4 + 2] = 0.0f;
        local_grad_rotations[i * 4 + 3] = 0.0f;
        
        local_grad_opacities[i] = 0.0f;
        
        for (int k = 0; k < 16 && k < sh_dim; k++) {
            local_grad_features[i * 16 + k] = 0.0f;
        }
    }
    
    
    // March BACKWARD along ray (CRITICAL for transmittance dependencies!)
    for (int s = N_samples - 1; s >= 0; s--) {
        // Read from shared memory instead of global memory
        float t = (s < MAX_T_SAMPLES_SHARED) ? s_t_samples[s] : t_samples[s];
        float3 pos = ray_o + ray_d * t;
        
        int out_idx = ray_idx * N_samples + s;
        
        // Load forward pass values
        // Load gradient from upstream
        float grad_output_s = grad_rho_density[out_idx];
        
        // ============================================================
        // Step 1: Recompute forward values (needed for gradient)
        // ============================================================
        
        float weighted_alphas_s = 0.0f;
        
    
        
        // ============================================================
        // Step 3: Backpropagate to Gaussian parameters
        // ============================================================
        
        for (int i = 0; i < num_gaussians; i++) {
            int g = valid_gaussian_indices[i];
            if (g < 0 || g >= N_gaussians) continue;
            
            // ============================================================
            // Load Gaussian params (COMPACT CACHE: Always reload params)
            // Trade-off: Small memory access vs 5x less cache memory
            // ============================================================
            float3 mean = make_float3(
                gaussian_means[g * 3 + 0],
                gaussian_means[g * 3 + 1],
                gaussian_means[g * 3 + 2]
            );
            
            float3 scale = make_float3(
                gaussian_scales[g * 3 + 0],
                gaussian_scales[g * 3 + 1],
                gaussian_scales[g * 3 + 2]
            );
            
            float4 quat = make_float4(
                gaussian_rotations[g * 4 + 0],
                gaussian_rotations[g * 4 + 1],
                gaussian_rotations[g * 4 + 2],
                gaussian_rotations[g * 4 + 3]
            );
            float opacity = gaussian_opacities[g];
            float pdf = eval_gaussian_pdf(pos, mean, scale, quat);
                
            float3 view_dir = normalize(mean - cam_pos);
            float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
            rho = fmaxf(rho + 0.5f, 0.0f);

            
            // ============================================================
            // CRITICAL: Separate time-independent and time-dependent paths!
            // ============================================================
            
            // global grad = local grad * grad_output_s
            // --- Local gradient ---
            float3 grad_mean_from_sh = make_float3(0.0f, 0.0f, 0.0f);
            float3 grad_mean_from_pdf = make_float3(0.0f, 0.0f, 0.0f);
            if (use_occlusion) {
                continue;
            } else {
                // mean gradient
                // 1. Gradient w.r.t. mean (via PDF)
                grad_mean_from_pdf = grad_gaussian_pdf_wrt_mean(pos, mean, scale, quat, pdf);

                // 2. Gradient w.r.t. mean (via view_dir → view-dependent reflectance path)
                if (rho > 0.0f) {
                    float jacobian[9];
                    float3 view_dir = normalize(mean - cam_pos);
                    grad_view_dir_wrt_mean(mean, cam_pos, jacobian);
                    
                    float3 grad_sh_wrt_dir = grad_sh_wrt_direction(
                        active_sh_degree,
                        &gaussian_features[g * sh_dim],
                        view_dir
                    );                
                    
                    grad_mean_from_sh = make_float3(
                        jacobian[0] * grad_sh_wrt_dir.x + jacobian[3] * grad_sh_wrt_dir.y + jacobian[6] * grad_sh_wrt_dir.z,
                        jacobian[1] * grad_sh_wrt_dir.x + jacobian[4] * grad_sh_wrt_dir.y + jacobian[7] * grad_sh_wrt_dir.z,
                        jacobian[2] * grad_sh_wrt_dir.x + jacobian[5] * grad_sh_wrt_dir.y + jacobian[8] * grad_sh_wrt_dir.z
                    );
                }

                float3 mean_gradient = grad_output_s * opacity * (rho * grad_mean_from_pdf + pdf * grad_mean_from_sh);

                local_grad_means[i * 3 + 0] += mean_gradient.x;
                local_grad_means[i * 3 + 1] += mean_gradient.y;
                local_grad_means[i * 3 + 2] += mean_gradient.z;


                // scale, quaternion gradient
                float3 grad_scale = grad_output_s * opacity * rho * grad_gaussian_pdf_wrt_scale(pos, mean, scale, quat, pdf);
                local_grad_scales[i * 3 + 0] += grad_scale.x;
                local_grad_scales[i * 3 + 1] += grad_scale.y;
                local_grad_scales[i * 3 + 2] += grad_scale.z;

                float4 grad_quat = grad_output_s * opacity * rho * grad_gaussian_pdf_wrt_quaternion(pos, mean, scale, quat, pdf);
                local_grad_rotations[i * 4 + 0] += grad_quat.x;
                local_grad_rotations[i * 4 + 1] += grad_quat.y;
                local_grad_rotations[i * 4 + 2] += grad_quat.z;
                local_grad_rotations[i * 4 + 3] += grad_quat.w;

                // opacity gradient
                float opacity_gradient = grad_output_s * rho * pdf;
                local_grad_opacities[i] += opacity_gradient;

                // feature gradient
                // Compute SH basis gradients
                if (rho > 0.0f){
                    float sh_basis_grads[16];  // Max (degree+1)^2 for degree 3
                    int max_coeffs = (active_sh_degree + 1) * (active_sh_degree + 1);
                    
                    float3 view_dir = normalize(mean - cam_pos);
                    grad_sh_wrt_coeffs(active_sh_degree, view_dir, sh_basis_grads);
                
                    for (int k = 0; k < max_coeffs && k < sh_dim; k++) {
                        local_grad_features[i * 16 + k] += grad_output_s * opacity * pdf * sh_basis_grads[k];
                    }   
                }
            }
        }
    }
    for (int i = 0; i < num_gaussians; i++) {
        int g = valid_gaussian_indices[i];
        if (g < 0 || g >= N_gaussians) continue;
        
        // Means
        if (local_grad_means[i * 3 + 0] != 0.0f || local_grad_means[i * 3 + 1] != 0.0f || local_grad_means[i * 3 + 2] != 0.0f) {
            atomicAdd(&grad_means[g * 3 + 0], local_grad_means[i * 3 + 0]);
            atomicAdd(&grad_means[g * 3 + 1], local_grad_means[i * 3 + 1]);
            atomicAdd(&grad_means[g * 3 + 2], local_grad_means[i * 3 + 2]);
        }
        
        // Log scales
        if (local_grad_scales[i * 3 + 0] != 0.0f || local_grad_scales[i * 3 + 1] != 0.0f || local_grad_scales[i * 3 + 2] != 0.0f) {
            atomicAdd(&grad_scales[g * 3 + 0], local_grad_scales[i * 3 + 0]);
            atomicAdd(&grad_scales[g * 3 + 1], local_grad_scales[i * 3 + 1]);
            atomicAdd(&grad_scales[g * 3 + 2], local_grad_scales[i * 3 + 2]);
        }
        
        // Rotations
        if (local_grad_rotations[i * 4 + 0] != 0.0f || local_grad_rotations[i * 4 + 1] != 0.0f || 
            local_grad_rotations[i * 4 + 2] != 0.0f || local_grad_rotations[i * 4 + 3] != 0.0f) {
            atomicAdd(&grad_rotations[g * 4 + 0], local_grad_rotations[i * 4 + 0]);
            atomicAdd(&grad_rotations[g * 4 + 1], local_grad_rotations[i * 4 + 1]);
            atomicAdd(&grad_rotations[g * 4 + 2], local_grad_rotations[i * 4 + 2]);
            atomicAdd(&grad_rotations[g * 4 + 3], local_grad_rotations[i * 4 + 3]);
        }
        
        // Opacity
        if (local_grad_opacities[i] != 0.0f) {
            atomicAdd(&grad_opacities[g], local_grad_opacities[i]);
        }
        
        // Features
        for (int k = 0; k < 16 && k < sh_dim; k++) {
            if (local_grad_features[i * 16 + k] != 0.0f) {
                atomicAdd(&grad_features[g * sh_dim + k], local_grad_features[i * 16 + k]);
            }
        }
    }
}

// ============================================================
// C++ Wrapper Function
// ============================================================

std::tuple<
    torch::Tensor,  // grad_gaussian_means
    torch::Tensor,  // grad_gaussian_scales
    torch::Tensor,  // grad_gaussian_rotations
    torch::Tensor,  // grad_gaussian_opacities
    torch::Tensor   // grad_gaussian_features
> simple_render_rays_backward(
    const torch::Tensor& rho_density,
    const torch::Tensor& density,
    const torch::Tensor& transmittance,
    const torch::Tensor& grad_rho_density,
    const torch::Tensor& grad_density,
    const torch::Tensor& grad_transmittance,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const torch::Tensor& t_samples,
    const torch::Tensor& gaussian_filter,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_scales,
    const torch::Tensor& gaussian_rotations,
    const torch::Tensor& gaussian_opacities,
    const torch::Tensor& gaussian_features,
    const torch::Tensor& camera_pos,
    const torch::Tensor& forward_cache,
    const int active_sh_degree,
    const float c,
    const float deltaT,
    const float scaling_modifier,
    const bool use_occlusion
) {
    // Input validation
    CHECK_INPUT(grad_rho_density);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(gaussian_means);
    
    const int N_rays = ray_origins.size(0);
    const int N_samples = t_samples.size(0);
    const int N_gaussians = gaussian_means.size(0);
    const int sh_dim = gaussian_features.size(1);
    
    // Allocate gradient tensors (initialized to zero)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(ray_origins.device());
    
    torch::Tensor grad_means = torch::zeros({N_gaussians, 3}, options);
    torch::Tensor grad_scales = torch::zeros({N_gaussians, 3}, options);
    torch::Tensor grad_rotations = torch::zeros({N_gaussians, 4}, options);
    torch::Tensor grad_opacities = torch::zeros({N_gaussians, 1}, options);
    torch::Tensor grad_features = torch::zeros({N_gaussians, sh_dim}, options);

    const ForwardCache* cache_ptr = nullptr;
    if (forward_cache.defined() && forward_cache.numel() > 0) {
        cache_ptr = reinterpret_cast<const ForwardCache*>(forward_cache.data_ptr<uint8_t>());
    }
    // Launch backward kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    simple_volume_render_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        grad_rho_density.data_ptr<float>(),
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
        density.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        cache_ptr,  // NEW! Forwar
        N_rays,
        N_samples,
        N_gaussians,
        active_sh_degree,
        sh_dim,
        c,
        deltaT,
        scaling_modifier,
        use_occlusion,
        grad_means.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_features.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(
        grad_means,
        grad_scales,
        grad_rotations,
        grad_opacities,
        grad_features
    );
}

