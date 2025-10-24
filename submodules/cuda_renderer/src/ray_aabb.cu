#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

#define MAX_GAUSSIANS_PER_RAY 256
#define THREADS_PER_BLOCK 256

// Kernel: Filter gaussians per ray based on AABB intersection
__global__ void filter_gaussians_kernel(
    const float* __restrict__ ray_origins,      // [N_rays, 3]
    const float* __restrict__ ray_directions,   // [N_rays, 3]
    const float* __restrict__ gaussian_bboxes,  // [N_gaussians, 2, 3]
    const int N_rays,
    const int N_gaussians,
    int* __restrict__ gaussian_indices,         // [N_rays, MAX_GAUSSIANS_PER_RAY]
    int* __restrict__ num_gaussians_per_ray     // [N_rays]
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
    
    int count = 0;
    
    // Test intersection with all Gaussians
    for (int g = 0; g < N_gaussians && count < MAX_GAUSSIANS_PER_RAY; g++) {
        // Load bbox
        float3 bbox_min = make_float3(
            gaussian_bboxes[g * 6 + 0],
            gaussian_bboxes[g * 6 + 1],
            gaussian_bboxes[g * 6 + 2]
        );
        
        float3 bbox_max = make_float3(
            gaussian_bboxes[g * 6 + 3],
            gaussian_bboxes[g * 6 + 4],
            gaussian_bboxes[g * 6 + 5]
        );
        
        float t_min, t_max;
        if (ray_aabb_intersect(ray_o, ray_d, bbox_min, bbox_max, t_min, t_max)) {
            gaussian_indices[ray_idx * MAX_GAUSSIANS_PER_RAY + count] = g;
            count++;
        }
    }
    
    num_gaussians_per_ray[ray_idx] = count;
}

torch::Tensor filter_gaussians_per_ray(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const torch::Tensor& gaussian_means,
    const torch::Tensor& gaussian_bboxes,
    const float sigma_threshold
) {
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(gaussian_bboxes);
    
    const int N_rays = ray_origins.size(0);
    const int N_gaussians = gaussian_bboxes.size(0);
    
    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(ray_origins.device());
    torch::Tensor gaussian_indices = torch::full({N_rays, MAX_GAUSSIANS_PER_RAY}, -1, options);
    torch::Tensor num_gaussians = torch::zeros({N_rays}, options);
    
    // Launch kernel
    const int blocks = (N_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    filter_gaussians_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        ray_origins.data_ptr<float>(),
        ray_directions.data_ptr<float>(),
        gaussian_bboxes.data_ptr<float>(),
        N_rays,
        N_gaussians,
        gaussian_indices.data_ptr<int>(),
        num_gaussians.data_ptr<int>()
    );
    
    cudaDeviceSynchronize();
    
    // Return concatenated tensor [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    // First column is count, rest are indices
    torch::Tensor result = torch::cat({num_gaussians.unsqueeze(1), gaussian_indices}, 1);
    
    return result;
}


