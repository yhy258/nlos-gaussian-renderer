#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "bbox_compute.cuh"
#include <tuple>

#define THREADS_PER_BLOCK 256
#define MAX_GAUSSIANS_PER_RAY 256

// Spherical harmonics evaluation (simplified for degree 0-3)
__device__ float eval_sh_basis(int l, int m, const float3& dir) {
    const float C0 = 0.28209479177387814f;  // 1 / (2 * sqrt(pi))
    
    if (l == 0) {
        return C0;
    }
    
    float x = dir.x, y = dir.y, z = dir.z;
    
    if (l == 1) {
        if (m == -1) return 0.4886025119029199f * y;       // sqrt(3/(4pi)) * y
        if (m ==  0) return 0.4886025119029199f * z;       // sqrt(3/(4pi)) * z
        if (m ==  1) return 0.4886025119029199f * x;       // sqrt(3/(4pi)) * x
    }
    
    if (l == 2) {
        if (m == -2) return 1.0925484305920792f * x * y;
        if (m == -1) return 1.0925484305920792f * y * z;
        if (m ==  0) return 0.31539156525252005f * (3.0f * z * z - 1.0f);
        if (m ==  1) return 1.0925484305920792f * x * z;
        if (m ==  2) return 0.5462742152960396f * (x * x - y * y);
    }
    
    if (l == 3) {
        if (m == -3) return 0.5900435899266435f * y * (3.0f * x * x - y * y);
        if (m == -2) return 2.890611442640554f * x * y * z;
        if (m == -1) return 0.4570457994644658f * y * (5.0f * z * z - 1.0f);
        if (m ==  0) return 0.3731763325901154f * z * (5.0f * z * z - 3.0f);
        if (m ==  1) return 0.4570457994644658f * x * (5.0f * z * z - 1.0f);
        if (m ==  2) return 1.445305721320277f * z * (x * x - y * y);
        if (m ==  3) return 0.5900435899266435f * x * (x * x - 3.0f * y * y);
    }
    
    return 0.0f;
}

__device__ float eval_sh(
    int degree,
    const float* sh_coeffs,  // [(degree+1)^2] coefficients
    const float3& dir
) {
    float result = 0.0f;
    int idx = 0;
    
    for (int l = 0; l <= degree; l++) {
        for (int m = -l; m <= l; m++) {
            result += sh_coeffs[idx] * eval_sh_basis(l, m, dir);
            idx++;
        }
    }
    
    return result;
}

// Volume rendering kernel for a batch of rays
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
    float* __restrict__ rho_density_out,          // [N_rays, N_samples]
    float* __restrict__ density_out,              // [N_rays, N_samples]
    float* __restrict__ transmittance_out         // [N_rays, N_samples]
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
    // gaussian_filter:First column is count, rest are indices
    int num_gaussians = gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1)]; // This is the number of valid gaussians for this ray.
    const int* valid_gaussian_indices = &gaussian_filter[ray_idx * (MAX_GAUSSIANS_PER_RAY + 1) + 1]; // This is the address of starting Gaussian index for this ray.
    // g = -1 means invalid Gaussian.

    // For each sample along the ray
    for (int s = 0; s < N_samples; s++) {
        float t = t_samples[s];
        float3 pos = ray_o + ray_d * t;
        
        float density = 0.0f;
        float rho_density = 0.0f;
        
        // Accumulate contribution from filtered Gaussians
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
            
            // Accumulate density
            float contrib = pdf * opacity;
            density += contrib;
            rho_density += contrib * rho;
        }
        
        // Store results
        int out_idx = ray_idx * N_samples + s;
        density_out[out_idx] = density;
        
        // Compute transmittance (will be done in a second pass or in Python)
        rho_density_out[out_idx] = rho_density;
        transmittance_out[out_idx] = 1.0f;  // placeholder
    }
}

// Second pass: compute transmittance and final rho_density
__global__ void compute_transmittance_kernel(
    const float* __restrict__ density,      // [N_rays, N_samples]
    const float* __restrict__ rho_density,  // [N_rays, N_samples]
    const int N_rays,
    const int N_samples,
    const float c,
    const float deltaT,
    const int rendering_type,               // 0: netf, 1: nlos-neus
    float* __restrict__ rho_density_out,    // [N_rays, N_samples]
    float* __restrict__ transmittance_out   // [N_rays, N_samples]
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= N_rays) return;
    
    float T = 1.0f;  // transmittance
    
    for (int s = 0; s < N_samples; s++) {
        int idx = ray_idx * N_samples + s;
        float dens = density[idx];
        float rho_dens = rho_density[idx];
        
        if (rendering_type == 0) {  // netf
            float occlusion = expf(-dens * c * deltaT);
            transmittance_out[idx] = T;
            rho_density_out[idx] = dens * T * (rho_dens / (dens + 1e-8f)) * c * deltaT;
            T *= occlusion;
        } else {  // nlos-neus
            float alpha = 1.0f - expf(-dens * c * deltaT);
            transmittance_out[idx] = T;
            rho_density_out[idx] = alpha * T * (rho_dens / (dens + 1e-8f));
            T *= (1.0f - alpha + 1e-7f);
        }
    }
}

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
    /*auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(ray_origins.device());
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
        gaussian_bboxes.view({N_gaussians, 2, 3}),
        3.0f  // sigma_threshold
    );
    */

    // When testing this CUDA based code, I want to use the simplified version (all gaussians for all rays)
    // ** For training and advanced testing, I will use the bbox computation + filtering. (uncomment the code above) **
    // Allocate temporary filter (simplified: all gaussians for all rays)
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(ray_origins.device());
    torch::Tensor gaussian_filter = torch::zeros({N_rays, MAX_GAUSSIANS_PER_RAY + 1}, options);
    
    // Simplified version (all gaussians for all rays)
    for (int i = 0; i < N_rays; i++) {
        gaussian_filter[i][0] = N_gaussians;
        for (int j = 0; j < N_gaussians && j < MAX_GAUSSIANS_PER_RAY; j++) {
            gaussian_filter[i][j + 1] = j;
        }
    }
    
    
    // Allocate output tensors
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(ray_origins.device());
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
    
    // Second pass: compute transmittance
    if (use_occlusion) {
        compute_transmittance_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            density.data_ptr<float>(),
            rho_density.data_ptr<float>(),
            N_rays,
            N_samples,
            c,
            deltaT,
            render_type,
            rho_density.data_ptr<float>(),
            transmittance.data_ptr<float>()
        );
        
        cudaDeviceSynchronize();
    }
    
    return std::make_tuple(rho_density, density, transmittance);
}


