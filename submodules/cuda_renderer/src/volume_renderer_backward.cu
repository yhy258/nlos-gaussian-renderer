#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "backward_utils.cuh"
#include "spherical_harmonics.cuh"
#include "volume_renderer_backward.h"

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256
#define MAX_T_SAMPLES_SHARED 512  // Shared memory limit for t_samples

/**
 * Volume Rendering Backward Pass Kernel
 * 
 * This kernel computes gradients w.r.t. Gaussian parameters
 * given gradients w.r.t. the output (rho_density).
 * 
 * Key algorithmic features:
 * 1. Processes rays in parallel (one ray per thread/block)
 * 2. Marches BACKWARD along each ray (reverse order) to handle transmittance dependencies
 * 3. Accumulates gradients using atomicAdd for thread safety
 * 
 * Mathematical derivation: See BACKWARD_PASS_DERIVATION.md
 * OPTIMIZED VERSION with Shared Memory
 */
__global__ void volume_render_backward_kernel(
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
    
    // Forward pass outputs (for recomputation)
    const float* __restrict__ density_fwd,        // [N_rays, N_samples]
    const float* __restrict__ transmittance_fwd,  // [N_rays, N_samples]
    
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
    float* __restrict__ grad_log_scales,  // [N_gaussians, 3]
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
    float local_grad_log_scales[MAX_GAUSSIANS_PER_RAY * 3];
    float local_grad_rotations[MAX_GAUSSIANS_PER_RAY * 4];
    float local_grad_opacities[MAX_GAUSSIANS_PER_RAY];
    float local_grad_features[MAX_GAUSSIANS_PER_RAY * 16];  // Assuming max SH degree 3
    
    // Initialize to zero
    for (int i = 0; i < num_gaussians; i++) {
        local_grad_means[i * 3 + 0] = 0.0f;
        local_grad_means[i * 3 + 1] = 0.0f;
        local_grad_means[i * 3 + 2] = 0.0f;
        
        local_grad_log_scales[i * 3 + 0] = 0.0f;
        local_grad_log_scales[i * 3 + 1] = 0.0f;
        local_grad_log_scales[i * 3 + 2] = 0.0f;
        
        local_grad_rotations[i * 4 + 0] = 0.0f;
        local_grad_rotations[i * 4 + 1] = 0.0f;
        local_grad_rotations[i * 4 + 2] = 0.0f;
        local_grad_rotations[i * 4 + 3] = 0.0f;
        
        local_grad_opacities[i] = 0.0f;
        
        for (int k = 0; k < 16 && k < sh_dim; k++) {
            local_grad_features[i * 16 + k] = 0.0f;
        }
    }
    
    // Accumulated gradient for transmittance (carries backward through samples)
    float grad_T_accumulated = 0.0f;

    /*
    
    Transmittance Gradient Analysis
    Tr = exp (- \sum_{k=1}^{r-1} \sum_{G in Gs} o_G * p_G * delta r)
    And this term can be also represented as :
    Tr = T1 * Contrib1 * Contrib2 * ... * Contrib_{r-1} where Contrib_k = exp (- \sum_{G in Gk} o_G * p_G * delta r) ; (IN OUR CODE, this would be same with the exponential of contrib)
    This means:
    Tr = T_{r-1} * Contrib_{r-1}

    In this case, how to compute the gradient for Tr (∂Lr / ∂Tr * ∂Tr/∂pn)? (Lr: Loss for r-th sample and pn: n-th Gaussian's pdf)
    # ASSUME: r is the end sample index.
    ∂L/∂T_{r-1} = ∂L/∂C_{r-1} * ∂C_{r-1}/∂T_{r-1} + ∂L/∂Tr * ∂T_{r}/∂T_{r-1}
    ∂L/∂T_{r-2} = ∂L/∂C_{r-2} * ∂C_{r-2}/∂T_{r-2} + ∂L/∂T_{r-1} * ∂T_{r-1}/∂T_{r-2} <CHAIN RULE>
    ...
    */
    
    // March BACKWARD along ray (CRITICAL for transmittance dependencies!)
    for (int s = N_samples - 1; s >= 0; s--) {
        // Read from shared memory instead of global memory
        float t = (s < MAX_T_SAMPLES_SHARED) ? s_t_samples[s] : t_samples[s];
        float3 pos = ray_o + ray_d * t;
        
        int out_idx = ray_idx * N_samples + s;
        
        // Load forward pass values
        float density_s = density_fwd[out_idx];
        float T_s = transmittance_fwd[out_idx];
        
        // Load gradient from upstream
        float grad_output_s = grad_rho_density[out_idx];
        
        // ============================================================
        // Step 1: Recompute forward values (needed for gradient)
        // ============================================================
        
        float weighted_alphas_s = 0.0f;
        
        // Store per-Gaussian intermediate values for gradient computation
        float pdf_values[MAX_GAUSSIANS_PER_RAY];
        float opacity_values[MAX_GAUSSIANS_PER_RAY];
        float rho_values[MAX_GAUSSIANS_PER_RAY];
        float alpha_values[MAX_GAUSSIANS_PER_RAY];
        float contrib_values[MAX_GAUSSIANS_PER_RAY];
        
        if (use_occlusion) {
            // Recompute weighted_alphas
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
                
                float opacity = 1.0f / (1.0f + expf(-gaussian_opacities[g]));
                float pdf = eval_gaussian_pdf(pos, mean, scale, quat);
                
                // View-dependent albedo
                float3 view_dir = normalize(mean - cam_pos);
                float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
                rho = fmaxf(rho + 0.5f, 0.0f);
                
                float contrib = pdf * opacity;
                // should we product the dr=c*deltaT? 
                // In NLOS-NeuS, they producted this factor since this term would be the discretized version of the integral for the ray samples points.
                // float alpha = 1.0f - expf(-contrib * c * deltaT); 
                float alpha = 1.0f - expf(-contrib);
                
                // Store for gradient computation
                pdf_values[i] = pdf;
                opacity_values[i] = opacity;
                rho_values[i] = rho;
                alpha_values[i] = alpha;
                contrib_values[i] = contrib;
                
                weighted_alphas_s += alpha * rho;
            }
        } else {
            // No occlusion case (simpler)
            for (int i = 0; i < num_gaussians; i++) {
                int g = valid_gaussian_indices[i];
                if (g < 0 || g >= N_gaussians) continue;
                
                // Similar recomputation but without alpha
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
                
                pdf_values[i] = pdf;
                opacity_values[i] = opacity;
                rho_values[i] = rho;
                alpha_values[i] = 0.0f;  // Not used in no-occlusion case
                contrib_values[i] = contrib;
            }
        }
        
        // ============================================================
        // Step 2: Compute local gradients
        // ============================================================
        
        float grad_weighted_alphas_s = 0.0f;
        float grad_T_s = 0.0f;
        float grad_density_s = 0.0f;
        
        if (use_occlusion) {
            // Gradient w.r.t. weighted_alphas (TIME-INDEPENDENT!)
            // Only from current sample's output, no future influence.
            grad_weighted_alphas_s = grad_output_s * T_s;

            float T_next = T_s * expf(-density_s * c * deltaT);
            grad_density_s = grad_T_accumulated * T_next * (-c * deltaT); // for the last ray (first traversal), this automatically become zero.

            float grad_T_s_local = grad_output_s * weighted_alphas_s;

            float grad_T_s_future = grad_T_accumulated * expf(-density_s * c * deltaT);
            float grad_T_s_total  = grad_T_s_local + grad_T_s_future;

            grad_T_accumulated = grad_T_s_total;
        } else {
            // No occlusion: simpler gradient flow
            grad_weighted_alphas_s = grad_output_s * c * deltaT;
            grad_density_s = 0.0f;
            grad_T_s = 0.0f;
        }
        
        // ============================================================
        // Step 3: Backpropagate to Gaussian parameters
        // ============================================================
        
        for (int i = 0; i < num_gaussians; i++) {
            int g = valid_gaussian_indices[i];
            if (g < 0 || g >= N_gaussians) continue;
            
            // Reload Gaussian parameters (needed for gradient computation)
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
            
            float pdf = pdf_values[i];
            float opacity = opacity_values[i];
            float rho = rho_values[i];
            float alpha = alpha_values[i];
            float contrib = contrib_values[i];
            
            // ============================================================
            // CRITICAL: Separate time-independent and time-dependent paths!
            // ============================================================
            
            float grad_pdf = 0.0f;
            float grad_opacity_local = 0.0f;
            
            if (use_occlusion) {
                // PATH 1: weighted_alphas → alpha → contrib (TIME-INDEPENDENT!)
                // weighted_alphas_s = Σ(alpha_g * rho_g)
                // ∂weighted_alphas_s/∂μ_g via alpha path only
                float grad_alpha = grad_weighted_alphas_s * rho;
                
                // alpha = 1 - exp(-contrib)
                // ∂alpha/∂contrib = exp(-contrib) = (1 - alpha)
                float grad_contrib_via_alpha = grad_alpha * (1.0f - alpha);
                
                // contrib = pdf * opacity
                float grad_pdf_via_alpha = grad_contrib_via_alpha * opacity;
                float grad_opacity_via_alpha = grad_contrib_via_alpha * pdf;
                
                grad_pdf += grad_pdf_via_alpha;
                grad_opacity_local += grad_opacity_via_alpha;
                
                // PATH 2: transmittance → density → contrib (TIME-DEPENDENT!)
                // density_s = Σ contrib_g
                // This affects T_{s+1}, T_{s+2}, ... (future samples)
                if (grad_density_s != 0.0f) {
                    // ∂density_s/∂contrib_g = 1.0
                    float grad_contrib_via_density = grad_density_s * 1.0f;
                    
                    // contrib = pdf * opacity
                    float grad_pdf_via_density = grad_contrib_via_density * opacity;
                    float grad_opacity_via_density = grad_contrib_via_density * pdf;
                    
                    grad_pdf += grad_pdf_via_density;
                    grad_opacity_local += grad_opacity_via_density;
                }
            } else {
                // No occlusion: only weighted_alphas path
                // weighted_radiance = Σ(contrib * rho)
                float grad_weighted_radiance = grad_weighted_alphas_s;
                float grad_contrib = grad_weighted_radiance * rho;
                grad_pdf = grad_contrib * opacity;
                grad_opacity_local = grad_contrib * pdf;
            }
            
            // --- Gradient w.r.t. rho ---
            float grad_rho = 0.0f;
            if (use_occlusion) {
                grad_rho = grad_weighted_alphas_s * alpha;
            } else {
                grad_rho = grad_weighted_alphas_s * contrib;
            }
            
            // Apply ReLU gradient (rho = max(sh_eval + 0.5, 0))
            if (rho <= 0.0f) {
                grad_rho = 0.0f;
            }
            
            // ============================================================
            // OPTIMIZATION: Accumulate to LOCAL buffers (no atomicAdd yet!)
            // ============================================================
            
            // 1. Gradient w.r.t. mean (via PDF)
            float3 grad_mean_from_pdf = grad_gaussian_pdf_wrt_mean(pos, mean, scale, quat, pdf);
            float3 grad_mean_total = grad_mean_from_pdf * grad_pdf; // ∂L/∂alpha * ∂alpha/∂mean
            
            // 1b. Gradient w.r.t. mean (via view_dir → rho path)
            if (grad_rho != 0.0f) {
                // Compute ∂view_dir/∂mean (3x3 Jacobian)
                float jacobian[9];
                float3 view_dir = normalize(mean - cam_pos);
                grad_view_dir_wrt_mean(mean, cam_pos, jacobian);
                
                // Compute ∂rho/∂view_dir (3D gradient vector)
                float3 grad_rho_wrt_dir = grad_sh_wrt_direction(
                    active_sh_degree,
                    &gaussian_features[g * sh_dim],
                    view_dir
                );
                
                // Chain rule: ∂L/∂mean = ∂L/∂rho * ∂rho/∂view_dir * ∂view_dir/∂mean
                float3 grad_mean_from_rho = make_float3(
                    jacobian[0] * grad_rho_wrt_dir.x + jacobian[3] * grad_rho_wrt_dir.y + jacobian[6] * grad_rho_wrt_dir.z,
                    jacobian[1] * grad_rho_wrt_dir.x + jacobian[4] * grad_rho_wrt_dir.y + jacobian[7] * grad_rho_wrt_dir.z,
                    jacobian[2] * grad_rho_wrt_dir.x + jacobian[5] * grad_rho_wrt_dir.y + jacobian[8] * grad_rho_wrt_dir.z
                );
                grad_mean_from_rho = grad_mean_from_rho * grad_rho;
                
                // Add to total
                grad_mean_total.x += grad_mean_from_rho.x;
                grad_mean_total.y += grad_mean_from_rho.y;
                grad_mean_total.z += grad_mean_from_rho.z;
            }
            
            // Accumulate to LOCAL buffer
            local_grad_means[i * 3 + 0] += grad_mean_total.x;
            local_grad_means[i * 3 + 1] += grad_mean_total.y;
            local_grad_means[i * 3 + 2] += grad_mean_total.z;
            
            // 2. Gradient w.r.t. log-scale (via PDF)
            float3 grad_log_scale_local = grad_gaussian_pdf_wrt_log_scale(pos, mean, scale, quat, pdf);
            grad_log_scale_local = grad_log_scale_local * grad_pdf;
            
            local_grad_log_scales[i * 3 + 0] += grad_log_scale_local.x;
            local_grad_log_scales[i * 3 + 1] += grad_log_scale_local.y;
            local_grad_log_scales[i * 3 + 2] += grad_log_scale_local.z;
            
            // 3. Gradient w.r.t. logit-opacity (via contribution)
            float grad_logit_opacity = grad_opacity_local * grad_sigmoid(opacity);
            local_grad_opacities[i] += grad_logit_opacity;
            
            // 4. Gradient w.r.t. quaternion (via PDF)
            float4 grad_quat_local = grad_gaussian_pdf_wrt_quaternion(pos, mean, scale, quat, pdf);
            grad_quat_local = grad_quat_local * grad_pdf;
            
            local_grad_rotations[i * 4 + 0] += grad_quat_local.x;
            local_grad_rotations[i * 4 + 1] += grad_quat_local.y;
            local_grad_rotations[i * 4 + 2] += grad_quat_local.z;
            local_grad_rotations[i * 4 + 3] += grad_quat_local.w;
            
            // 5. Gradient w.r.t. SH features (via rho)
            if (grad_rho != 0.0f) {
                // Compute SH basis gradients
                float sh_basis_grads[16];  // Max (degree+1)^2 for degree 3
                int max_coeffs = (active_sh_degree + 1) * (active_sh_degree + 1);
                
                float3 view_dir = normalize(mean - cam_pos);
                grad_sh_wrt_coeffs(active_sh_degree, view_dir, sh_basis_grads);
                
                for (int k = 0; k < max_coeffs && k < sh_dim; k++) {
                    local_grad_features[i * 16 + k] += grad_rho * sh_basis_grads[k];
                }
            }
        }
    }
    
    // ============================================================
    // OPTIMIZATION: Write local gradients to global memory ONCE per ray
    // This reduces atomicAdd calls from O(N_samples * N_gaussians) to O(N_gaussians)
    // Expected speedup: ~N_samples times (e.g., 256x fewer atomicAdd calls!)
    // ============================================================
    
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
        if (local_grad_log_scales[i * 3 + 0] != 0.0f || local_grad_log_scales[i * 3 + 1] != 0.0f || local_grad_log_scales[i * 3 + 2] != 0.0f) {
            atomicAdd(&grad_log_scales[g * 3 + 0], local_grad_log_scales[i * 3 + 0]);
            atomicAdd(&grad_log_scales[g * 3 + 1], local_grad_log_scales[i * 3 + 1]);
            atomicAdd(&grad_log_scales[g * 3 + 2], local_grad_log_scales[i * 3 + 2]);
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
> render_rays_backward(
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
    torch::Tensor grad_log_scales = torch::zeros({N_gaussians, 3}, options);
    torch::Tensor grad_rotations = torch::zeros({N_gaussians, 4}, options);
    torch::Tensor grad_opacities = torch::zeros({N_gaussians, 1}, options);
    torch::Tensor grad_features = torch::zeros({N_gaussians, sh_dim}, options);

    
    // Launch backward kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    volume_render_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
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
        grad_log_scales.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_features.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(
        grad_means,
        grad_log_scales,
        grad_rotations,
        grad_opacities,
        grad_features
    );
}

