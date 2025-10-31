#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "bbox_compute.cuh"
#include "spherical_harmonics.cuh"
#include <tuple>

#define THREADS_PER_BLOCK 256

// Kernel for computing albedo at given 3D coordinates with AABB filtering
__global__ void albedo_coords_kernel(
    const float* __restrict__ coords,             // [N_coords, 3] - 3D coordinates to evaluate
    const float* __restrict__ gaussian_means,     // [N_gaussians, 3]
    const float* __restrict__ gaussian_scales,    // [N_gaussians, 3]
    const float* __restrict__ gaussian_rotations, // [N_gaussians, 4]
    const float* __restrict__ gaussian_opacities, // [N_gaussians, 1]
    const float* __restrict__ gaussian_features,  // [N_gaussians, K]
    const float* __restrict__ gaussian_bboxes,    // [N_gaussians, 6] - precomputed AABBs
    const float* __restrict__ camera_pos,         // [3]
    const int N_coords,
    const int N_gaussians,
    const int active_sh_degree,
    const int sh_dim,
    const float scaling_modifier,
    const float sigma_threshold,                  // Sigma threshold for culling (e.g., 3.0)
    float* __restrict__ albedo_out               // [N_coords] - output albedo values
) {
    int coord_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (coord_idx >= N_coords) return;
    
    // Load 3D coordinate
    float3 pos = make_float3(
        coords[coord_idx * 3 + 0],
        coords[coord_idx * 3 + 1],
        coords[coord_idx * 3 + 2]
    );
    
    float3 cam_pos = make_float3(camera_pos[0], camera_pos[1], camera_pos[2]);
    
    // Accumulate albedo from all Gaussians at this coordinate
    float total_weighted_albedo = 0.0f;
    float total_weight = 0.0f;
    
    for (int g = 0; g < N_gaussians; g++) {
        // AABB filtering: Check if point is inside Gaussian's bounding box
        float bbox_min_x = gaussian_bboxes[g * 6 + 0];
        float bbox_min_y = gaussian_bboxes[g * 6 + 1];
        float bbox_min_z = gaussian_bboxes[g * 6 + 2];
        float bbox_max_x = gaussian_bboxes[g * 6 + 3];
        float bbox_max_y = gaussian_bboxes[g * 6 + 4];
        float bbox_max_z = gaussian_bboxes[g * 6 + 5];
        
        // Skip if point is outside AABB
        if (pos.x < bbox_min_x || pos.x > bbox_max_x ||
            pos.y < bbox_min_y || pos.y > bbox_max_y ||
            pos.z < bbox_min_z || pos.z > bbox_max_z) {
            continue;
        }
        // Get Gaussian parameters
        float3 mean = make_float3(
            gaussian_means[g * 3 + 0],
            gaussian_means[g * 3 + 1],
            gaussian_means[g * 3 + 2]
        );
        
        float3 scale = make_float3(
            gaussian_scales[g * 3 + 0] * scaling_modifier,
            gaussian_scales[g * 3 + 1] * scaling_modifier,
            gaussian_scales[g * 3 + 2] * scaling_modifier
        );
        
        float4 quat = make_float4(
            gaussian_rotations[g * 4 + 0],
            gaussian_rotations[g * 4 + 1],
            gaussian_rotations[g * 4 + 2],
            gaussian_rotations[g * 4 + 3]
        );
        
        float opacity = gaussian_opacities[g];
        
        // Compute rotation matrix from quaternion
        float r = quat.x;
        float x = quat.y;
        float y = quat.z;
        float z = quat.w;
        
        float R[3][3];
        R[0][0] = 1.f - 2.f * (y * y + z * z);
        R[0][1] = 2.f * (x * y - r * z);
        R[0][2] = 2.f * (x * z + r * y);
        R[1][0] = 2.f * (x * y + r * z);
        R[1][1] = 1.f - 2.f * (x * x + z * z);
        R[1][2] = 2.f * (y * z - r * x);
        R[2][0] = 2.f * (x * z - r * y);
        R[2][1] = 2.f * (y * z + r * x);
        R[2][2] = 1.f - 2.f * (x * x + y * y);
        
        // Compute difference vector
        float3 diff = pos - mean;
        
        // Apply rotation (R^T * diff)
        float3 transformed_diff;
        transformed_diff.x = R[0][0] * diff.x + R[1][0] * diff.y + R[2][0] * diff.z;
        transformed_diff.y = R[0][1] * diff.x + R[1][1] * diff.y + R[2][1] * diff.z;
        transformed_diff.z = R[0][2] * diff.x + R[1][2] * diff.y + R[2][2] * diff.z;
        
        // Compute Gaussian weight (without normalization factor)
        float exponent = -0.5f * (
            (transformed_diff.x * transformed_diff.x) / (scale.x * scale.x) +
            (transformed_diff.y * transformed_diff.y) / (scale.y * scale.y) +
            (transformed_diff.z * transformed_diff.z) / (scale.z * scale.z)
        );
        
        // Skip if too far from Gaussian center
        if (exponent < -10.0f) continue;
        
        float gaussian_weight = expf(exponent);
        float density = gaussian_weight * opacity;
        
        // Compute view-dependent albedo using SH
        float3 view_dir = normalize(mean - cam_pos);
        float rho = eval_sh(active_sh_degree, &gaussian_features[g * sh_dim], view_dir);
        rho = fmaxf(rho + 0.5f, 0.0f);  // clamp_min(sh2rho + 0.5, 0.0)
        
        // Accumulate weighted albedo
        total_weighted_albedo += rho * density;
        total_weight += density;
    }
    
    // Store final albedo (weighted average)
    albedo_out[coord_idx] = (total_weight > 1e-6f) ? (total_weighted_albedo / total_weight) : 0.0f;
}

// Kernel for computing density at given 3D coordinates with AABB filtering
__global__ void density_coords_kernel(
    const float* __restrict__ coords,             // [N_coords, 3]
    const float* __restrict__ gaussian_means,     // [N_gaussians, 3]
    const float* __restrict__ gaussian_scales,    // [N_gaussians, 3]
    const float* __restrict__ gaussian_rotations, // [N_gaussians, 4]
    const float* __restrict__ gaussian_opacities, // [N_gaussians, 1]
    const float* __restrict__ gaussian_bboxes,    // [N_gaussians, 6] - precomputed AABBs
    const int N_coords,
    const int N_gaussians,
    const float scaling_modifier,
    const float sigma_threshold,
    float* __restrict__ density_out              // [N_coords]
) {
    int coord_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (coord_idx >= N_coords) return;
    
    float3 pos = make_float3(
        coords[coord_idx * 3 + 0],
        coords[coord_idx * 3 + 1],
        coords[coord_idx * 3 + 2]
    );
    
    float total_density = 0.0f;
    
    for (int g = 0; g < N_gaussians; g++) {
        // AABB filtering
        float bbox_min_x = gaussian_bboxes[g * 6 + 0];
        float bbox_min_y = gaussian_bboxes[g * 6 + 1];
        float bbox_min_z = gaussian_bboxes[g * 6 + 2];
        float bbox_max_x = gaussian_bboxes[g * 6 + 3];
        float bbox_max_y = gaussian_bboxes[g * 6 + 4];
        float bbox_max_z = gaussian_bboxes[g * 6 + 5];
        
        if (pos.x < bbox_min_x || pos.x > bbox_max_x ||
            pos.y < bbox_min_y || pos.y > bbox_max_y ||
            pos.z < bbox_min_z || pos.z > bbox_max_z) {
            continue;
        }
        float3 mean = make_float3(
            gaussian_means[g * 3 + 0],
            gaussian_means[g * 3 + 1],
            gaussian_means[g * 3 + 2]
        );
        
        float3 scale = make_float3(
            gaussian_scales[g * 3 + 0] * scaling_modifier,
            gaussian_scales[g * 3 + 1] * scaling_modifier,
            gaussian_scales[g * 3 + 2] * scaling_modifier
        );
        
        float4 quat = make_float4(
            gaussian_rotations[g * 4 + 0],
            gaussian_rotations[g * 4 + 1],
            gaussian_rotations[g * 4 + 2],
            gaussian_rotations[g * 4 + 3]
        );
        
        float opacity = gaussian_opacities[g];
        
        // Compute rotation matrix
        float r = quat.x;
        float x = quat.y;
        float y = quat.z;
        float z = quat.w;
        
        float R[3][3];
        R[0][0] = 1.f - 2.f * (y * y + z * z);
        R[0][1] = 2.f * (x * y - r * z);
        R[0][2] = 2.f * (x * z + r * y);
        R[1][0] = 2.f * (x * y + r * z);
        R[1][1] = 1.f - 2.f * (x * x + z * z);
        R[1][2] = 2.f * (y * z - r * x);
        R[2][0] = 2.f * (x * z - r * y);
        R[2][1] = 2.f * (y * z + r * x);
        R[2][2] = 1.f - 2.f * (x * x + y * y);
        
        float3 diff = pos - mean;
        
        // Apply rotation
        float3 transformed_diff;
        transformed_diff.x = R[0][0] * diff.x + R[1][0] * diff.y + R[2][0] * diff.z;
        transformed_diff.y = R[0][1] * diff.x + R[1][1] * diff.y + R[2][1] * diff.z;
        transformed_diff.z = R[0][2] * diff.x + R[1][2] * diff.y + R[2][2] * diff.z;
        
        float exponent = -0.5f * (
            (transformed_diff.x * transformed_diff.x) / (scale.x * scale.x) +
            (transformed_diff.y * transformed_diff.y) / (scale.y * scale.y) +
            (transformed_diff.z * transformed_diff.z) / (scale.z * scale.z)
        );
        
        if (exponent < -10.0f) continue;
        
        float gaussian_weight = expf(exponent);
        total_density += gaussian_weight * opacity;
    }
    
    density_out[coord_idx] = total_density;
}

// Python binding function for albedo computation at coordinates with AABB filtering
torch::Tensor compute_albedo_at_coords(
    const torch::Tensor& coords,              // [N_coords, 3] or [H, W, D, 3]
    const torch::Tensor& gaussian_means,      // [N_gaussians, 3]
    const torch::Tensor& gaussian_scales,     // [N_gaussians, 3]
    const torch::Tensor& gaussian_rotations,  // [N_gaussians, 4]
    const torch::Tensor& gaussian_opacities,  // [N_gaussians, 1]
    const torch::Tensor& gaussian_features,   // [N_gaussians, K]
    const torch::Tensor& camera_pos,          // [3]
    const int active_sh_degree,
    const float scaling_modifier,
    const float sigma_threshold = 3.0f        // Default sigma threshold for AABB
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(gaussian_means);
    CHECK_INPUT(gaussian_scales);
    CHECK_INPUT(gaussian_rotations);
    CHECK_INPUT(gaussian_opacities);
    CHECK_INPUT(gaussian_features);
    CHECK_INPUT(camera_pos);
    
    // Handle multi-dimensional coords input
    auto original_shape = coords.sizes();
    auto coords_flat = coords.reshape({-1, 3}).contiguous();
    
    const int N_coords = coords_flat.size(0);
    const int N_gaussians = gaussian_means.size(0);
    const int sh_dim = gaussian_features.size(1);
    
    // Allocate output
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(coords.device());
    torch::Tensor albedo_out = torch::zeros({N_coords}, options);
    
    // Compute Gaussian bounding boxes on GPU
    torch::Tensor gaussian_bboxes = torch::empty({N_gaussians, 6}, options);
    
    {
        const int blocks = (N_gaussians + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_gaussian_bboxes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            gaussian_means.data_ptr<float>(),
            gaussian_scales.data_ptr<float>(),
            gaussian_rotations.data_ptr<float>(),
            N_gaussians,
            scaling_modifier,
            sigma_threshold,
            gaussian_bboxes.data_ptr<float>()
        );
        cudaDeviceSynchronize();
    }
    
    // Launch albedo kernel with AABB filtering
    const int blocks = (N_coords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    albedo_coords_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        coords_flat.data_ptr<float>(),
        gaussian_means.data_ptr<float>(),
        gaussian_scales.data_ptr<float>(),
        gaussian_rotations.data_ptr<float>(),
        gaussian_opacities.data_ptr<float>(),
        gaussian_features.data_ptr<float>(),
        gaussian_bboxes.data_ptr<float>(),  // Pass precomputed AABBs
        camera_pos.data_ptr<float>(),
        N_coords,
        N_gaussians,
        active_sh_degree,
        sh_dim,
        scaling_modifier,
        sigma_threshold,
        albedo_out.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    // Reshape output to match input shape (except last dimension)
    std::vector<int64_t> output_shape(original_shape.begin(), original_shape.end() - 1);
    return albedo_out.reshape(output_shape);
}

// Python binding function for density computation at coordinates with AABB filtering
torch::Tensor compute_density_at_coords(
    const torch::Tensor& coords,              // [N_coords, 3] or [H, W, D, 3]
    const torch::Tensor& gaussian_means,      // [N_gaussians, 3]
    const torch::Tensor& gaussian_scales,     // [N_gaussians, 3]
    const torch::Tensor& gaussian_rotations,  // [N_gaussians, 4]
    const torch::Tensor& gaussian_opacities,  // [N_gaussians, 1]
    const float scaling_modifier,
    const float sigma_threshold = 3.0f        // Default sigma threshold for AABB
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(gaussian_means);
    CHECK_INPUT(gaussian_scales);
    CHECK_INPUT(gaussian_rotations);
    CHECK_INPUT(gaussian_opacities);
    
    auto original_shape = coords.sizes();
    auto coords_flat = coords.reshape({-1, 3}).contiguous();
    
    const int N_coords = coords_flat.size(0);
    const int N_gaussians = gaussian_means.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(coords.device());
    torch::Tensor density_out = torch::zeros({N_coords}, options);
    
    // Compute Gaussian bounding boxes on GPU
    torch::Tensor gaussian_bboxes = torch::empty({N_gaussians, 6}, options);
    
    {
        const int blocks = (N_gaussians + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_gaussian_bboxes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            gaussian_means.data_ptr<float>(),
            gaussian_scales.data_ptr<float>(),
            gaussian_rotations.data_ptr<float>(),
            N_gaussians,
            scaling_modifier,
            sigma_threshold,
            gaussian_bboxes.data_ptr<float>()
        );
        cudaDeviceSynchronize();
    }
    
    // Launch density kernel with AABB filtering
    const int blocks = (N_coords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    density_coords_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        coords_flat.data_ptr<float>(),
        gaussian_means.data_ptr<float>(),
        gaussian_scales.data_ptr<float>(),
        gaussian_rotations.data_ptr<float>(),
        gaussian_opacities.data_ptr<float>(),
        gaussian_bboxes.data_ptr<float>(),  // Pass precomputed AABBs
        N_coords,
        N_gaussians,
        scaling_modifier,
        sigma_threshold,
        density_out.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    std::vector<int64_t> output_shape(original_shape.begin(), original_shape.end() - 1);
    return density_out.reshape(output_shape);
}
