#ifndef BACKWARD_UTILS_CUH
#define BACKWARD_UTILS_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

/**
 * Backward utilities for volume rendering gradients
 * 
 * These functions compute gradients of intermediate quantities
 * w.r.t. learnable Gaussian parameters.
 */

// ============================================================
// Gradient of Gaussian PDF w.r.t. Mean
// ============================================================

/**
 * Compute gradient of Gaussian PDF w.r.t. mean position
 * 
 * PDF = exp(-0.5 * (pos - mean)^T * Sigma^{-1} * (pos - mean))
 * 
 * Where Sigma^{-1} = R * diag(1/scale^2) * R^T
 * 
 * Returns: ∂PDF/∂mean
 */
__device__ __forceinline__ float3 grad_gaussian_pdf_wrt_mean(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float pdf_value  // Pre-computed PDF value
) {
    // Transform to local frame
    float R[9];
    quat_to_rotmat(quat, R);
    
    float3 delta = pos - mean;
    
    // delta_local = R^T * delta
    float3 delta_local = make_float3(
        R[0] * delta.x + R[3] * delta.y + R[6] * delta.z,
        R[1] * delta.x + R[4] * delta.y + R[7] * delta.z,
        R[2] * delta.x + R[5] * delta.y + R[8] * delta.z
    );
    
    // Normalized delta: delta_local / scale
    const float eps = 1e-8f;
    float3 normalized = make_float3(
        delta_local.x / (scale.x + eps),
        delta_local.y / (scale.y + eps),
        delta_local.z / (scale.z + eps)
    );
    
    // Gradient in local frame: pdf * normalized / scale
    float3 grad_local = make_float3(
        pdf_value * normalized.x / (scale.x + eps),
        pdf_value * normalized.y / (scale.y + eps),
        pdf_value * normalized.z / (scale.z + eps)
    );
    
    // Transform back to world frame: grad_mean = -R * grad_local
    // (negative because ∂/∂mean of (pos - mean) = -1)
    float3 grad_mean = make_float3(
        -(R[0] * grad_local.x + R[1] * grad_local.y + R[2] * grad_local.z),
        -(R[3] * grad_local.x + R[4] * grad_local.y + R[5] * grad_local.z),
        -(R[6] * grad_local.x + R[7] * grad_local.y + R[8] * grad_local.z)
    );
    
    return grad_mean;
}

// ============================================================
// Gradient of Gaussian PDF w.r.t. Covariance Matrix
// ============================================================

/**
 * Compute gradient of Gaussian PDF w.r.t. Covariance matrix Σ
 * 
 * Mathematical derivation:
 * ========================
 * 
 * Given: PDF = exp(-1/2 * d^T * Σ^{-1} * d) where d = pos - mean
 * 
 * Goal: Compute ∂PDF/∂Σ
 * 
 * Step 1: Gradient w.r.t. Σ^{-1} (easier to compute)
 * ---------------------------------------------------
 * Let f = -1/2 * d^T * Σ^{-1} * d
 * 
 * ∂f/∂Σ^{-1} = -1/2 * d * d^T  (matrix derivative)
 * 
 * ∂PDF/∂Σ^{-1} = PDF * ∂f/∂Σ^{-1} = -1/2 * PDF * (d * d^T)
 * 
 * Step 2: Convert to gradient w.r.t. Σ
 * -------------------------------------
 * Using matrix inversion derivative: ∂Σ^{-1}/∂Σ = -Σ^{-1} ⊗ Σ^{-1}
 * 
 * Result: ∂PDF/∂Σ = 1/2 * PDF * Σ^{-1} * (d * d^T) * Σ^{-1}
 * 
 * Step 3: Efficient computation for Σ = R S^2 R^T
 * ------------------------------------------------
 * Σ^{-1} = R S^{-2} R^T
 * 
 * Let v = Σ^{-1} * d = R S^{-2} R^T * d
 * Then: ∂PDF/∂Σ = 1/2 * PDF * (v * v^T)
 * 
 * This is a symmetric 3x3 matrix stored as 6 unique elements.
 * 
 * @param pos      Sample position
 * @param mean     Gaussian mean
 * @param scale    Gaussian scale (diagonal of S)
 * @param quat     Rotation quaternion (defines R)
 * @param pdf_value Pre-computed PDF value
 * @param grad_Sigma Output: gradient w.r.t. Σ as 6 elements [Σ₀₀, Σ₀₁, Σ₀₂, Σ₁₁, Σ₁₂, Σ₂₂]
 */
__device__ __forceinline__ void grad_gaussian_pdf_wrt_covariance(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float pdf_value,
    float* grad_Sigma  // Output: 6 elements
) {
    const float eps = 1e-8f;
    
    // Build rotation matrix from quaternion
    float R[9];
    quat_to_rotmat(quat, R);
    
    // Compute deviation vector: d = pos - mean
    float3 d = pos - mean;
    
    // Step 1: Transform to local frame: d_local = R^T * d
    float3 d_local = make_float3(
        R[0] * d.x + R[3] * d.y + R[6] * d.z,
        R[1] * d.x + R[4] * d.y + R[7] * d.z,
        R[2] * d.x + R[5] * d.y + R[8] * d.z
    );
    
    // Step 2: Apply S^{-2} in local frame: w = S^{-2} * d_local
    float3 w = make_float3(
        d_local.x / (scale.x * scale.x + eps),
        d_local.y / (scale.y * scale.y + eps),
        d_local.z / (scale.z * scale.z + eps)
    );
    
    // Step 3: Transform back to world frame: v = R * w = Σ^{-1} * d
    float3 v = make_float3(
        R[0] * w.x + R[1] * w.y + R[2] * w.z,
        R[3] * w.x + R[4] * w.y + R[5] * w.z,
        R[6] * w.x + R[7] * w.y + R[8] * w.z
    );
    
    // Step 4: Compute outer product: v * v^T (symmetric 3x3 matrix)
    // ∂PDF/∂Σ = 1/2 * PDF * (v * v^T)
    float coeff = 0.5f * pdf_value;
    
    // Store as 6 unique elements (upper triangular including diagonal)
    // Order: [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
    grad_Sigma[0] = coeff * v.x * v.x;  // Σ₀₀
    grad_Sigma[1] = coeff * v.x * v.y;  // Σ₀₁
    grad_Sigma[2] = coeff * v.x * v.z;  // Σ₀₂
    grad_Sigma[3] = coeff * v.y * v.y;  // Σ₁₁
    grad_Sigma[4] = coeff * v.y * v.z;  // Σ₁₂
    grad_Sigma[5] = coeff * v.z * v.z;  // Σ₂₂
}

/**
 * Alternative version that returns gradient as full 3x3 symmetric matrix
 * 
 * This is useful when you need to chain with other matrix operations.
 * 
 * @param grad_Sigma_mat Output: 9 elements stored row-major (symmetric)
 */
__device__ __forceinline__ void grad_gaussian_pdf_wrt_covariance_full(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float pdf_value,
    float* grad_Sigma_mat  // Output: 9 elements (3x3, row-major)
) {
    const float eps = 1e-8f;
    
    // Build rotation matrix
    float R[9];
    quat_to_rotmat(quat, R);
    
    // d = pos - mean
    float3 d = pos - mean;
    
    // d_local = R^T * d
    float3 d_local = make_float3(
        R[0] * d.x + R[3] * d.y + R[6] * d.z,
        R[1] * d.x + R[4] * d.y + R[7] * d.z,
        R[2] * d.x + R[5] * d.y + R[8] * d.z
    );
    
    // w = S^{-2} * d_local
    float3 w = make_float3(
        d_local.x / (scale.x * scale.x + eps),
        d_local.y / (scale.y * scale.y + eps),
        d_local.z / (scale.z * scale.z + eps)
    );
    
    // v = R * w = Σ^{-1} * d
    float3 v = make_float3(
        R[0] * w.x + R[1] * w.y + R[2] * w.z,
        R[3] * w.x + R[4] * w.y + R[5] * w.z,
        R[6] * w.x + R[7] * w.y + R[8] * w.z
    );
    
    // Compute symmetric outer product: 1/2 * PDF * (v * v^T)
    float coeff = 0.5f * pdf_value;
    
    // Row 0
    grad_Sigma_mat[0] = coeff * v.x * v.x;
    grad_Sigma_mat[1] = coeff * v.x * v.y;
    grad_Sigma_mat[2] = coeff * v.x * v.z;
    
    // Row 1 (symmetric)
    grad_Sigma_mat[3] = grad_Sigma_mat[1];  // v.x * v.y
    grad_Sigma_mat[4] = coeff * v.y * v.y;
    grad_Sigma_mat[5] = coeff * v.y * v.z;
    
    // Row 2 (symmetric)
    grad_Sigma_mat[6] = grad_Sigma_mat[2];  // v.x * v.z
    grad_Sigma_mat[7] = grad_Sigma_mat[5];  // v.y * v.z
    grad_Sigma_mat[8] = coeff * v.z * v.z;
}



// ============================================================
// Gradient of Gaussian PDF w.r.t. Scale (log-scale)
// ============================================================

/**
 * Compute gradient of Gaussian PDF w.r.t. log-scale
 * 
 * Since scale = exp(log_scale), we apply chain rule:
 * ∂PDF/∂log_scale = ∂PDF/∂scale * ∂scale/∂log_scale
 *                 = ∂PDF/∂scale * scale
 * 
 * Returns: ∂PDF/∂log_scale (3D vector)
 */
__device__ __forceinline__ float3 grad_gaussian_pdf_wrt_log_scale(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float pdf_value
) {
    // Transform to local frame
    float R[9];
    quat_to_rotmat(quat, R);
    
    float3 delta = pos - mean;
    
    // delta_local = R^T * delta
    float3 delta_local = make_float3(
        R[0] * delta.x + R[3] * delta.y + R[6] * delta.z,
        R[1] * delta.x + R[4] * delta.y + R[7] * delta.z,
        R[2] * delta.x + R[5] * delta.y + R[8] * delta.z
    );
    
    // Normalized delta
    const float eps = 1e-8f;
    float3 normalized = make_float3(
        delta_local.x / (scale.x + eps),
        delta_local.y / (scale.y + eps),
        delta_local.z / (scale.z + eps)
    );
    
    // Mahalanobis distance squared components
    float3 mahal_sq_components = make_float3(
        normalized.x * normalized.x,
        normalized.y * normalized.y,
        normalized.z * normalized.z
    );
    
    // Gradient w.r.t. scale: pdf * mahal_sq / scale
    float3 grad_scale = make_float3(
        pdf_value * mahal_sq_components.x / (scale.x + eps),
        pdf_value * mahal_sq_components.y / (scale.y + eps),
        pdf_value * mahal_sq_components.z / (scale.z + eps)
    );
    
    // Chain rule: ∂scale/∂log_scale = scale
    float3 grad_log_scale = make_float3(
        grad_scale.x * scale.x,
        grad_scale.y * scale.y,
        grad_scale.z * scale.z
    );
    
    return grad_log_scale;
}

/**
 * Compute gradient of Gaussian PDF w.r.t. quaternion
 * 
 * PDF = exp(-0.5 * (pos - mean)^T * Sigma^{-1} * (pos - mean))
 * where Sigma = R * S * S^T * R^T
 * 
 * We need to compute d(PDF)/d(quat) using chain rule through covariance matrix.
 * 
 * Note: This returns gradient w.r.t. the 4 quaternion components (qx, qy, qz, qw)
 * but we only return a float3 for the last 3 components since the gradient computation
 * treats the quaternion as (r, x, y, z) = (qx, qy, qz, qw) and we pack the result.
 * 
 * Returns: gradient as float3 containing derivatives w.r.t. (qy, qz, qw)
 *          The derivative w.r.t. qx can be computed separately if needed.
 */
__device__ __forceinline__ float4 grad_gaussian_pdf_wrt_quaternion(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat,
    const float pdf_value
){
    const float eps = 1e-8f;
    
    // Build rotation matrix from quaternion
    float R[9];
    quat_to_rotmat(quat, R);
    
    float3 delta = pos - mean; // location difference
    float3 delta_local = make_float3( // R^T (x- µ)
        R[0] * delta.x + R[3] * delta.y + R[6] * delta.z,
        R[1] * delta.x + R[4] * delta.y + R[7] * delta.z,
        R[2] * delta.x + R[5] * delta.y + R[8] * delta.z
    );
    float3 normalized = make_float3(
        delta_local.x / (scale.x * scale.x + eps),
        delta_local.y / (scale.y * scale.y + eps),
        delta_local.z / (scale.z * scale.z + eps)
    );

    float3 v = make_float3(
        R[0] * normalized.x + R[1] * normalized.y + R[2] * normalized.z,
        R[3] * normalized.x + R[4] * normalized.y + R[5] * normalized.z,
        R[6] * normalized.x + R[7] * normalized.y + R[8] * normalized.z
    );
    const float scale_factor = 0.5f * pdf_value;
    float3 v_scaled = make_float3(
        v.x * scale_factor,
        v.y * scale_factor,
        v.z * scale_factor
    );


    float dpdcov[9];
    outer_product(v, v_scaled, dpdcov); // dpdcov = vv^T * 1/2 * pdf


    // ∂p/∂M =  2M*∂p/∂∑ (dpdcov)
    float M[9];
    M[0] = R[0] * scale.x; M[1] = R[1] * scale.y; M[2] = R[2] * scale.z;
    M[3] = R[3] * scale.x; M[4] = R[4] * scale.y; M[5] = R[5] * scale.z;
    M[6] = R[6] * scale.x; M[7] = R[7] * scale.y; M[8] = R[8] * scale.z;

    // J_M = 2.0f * J_Sigma * M
    // dpdM = 2.0f * dpdcov * M
    float dpdM[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // (dpdM)_ij = 2.0 * sum_k (dpdcov)_ik * M_kj
            dpdM[i * 3 + j] = 2.0f * (
                dpdcov[i * 3 + 0] * M[0 * 3 + j] +
                dpdcov[i * 3 + 1] * M[1 * 3 + j] +
                dpdcov[i * 3 + 2] * M[2 * 3 + j]
            );
        }
    }

    
    // ∂p/∂R = ∂p/∂M*S
    float dpdR[9];
    float scales[3] = {scale.x, scale.y, scale.z};
    for (int i = 0; i < 3; i++) { // i = row
        for (int j = 0; j < 3; j++) { // j = column
            // (J_R)_ij = sum_k (J_M)_ik * S_kj = (J_M)_ij * S_jj
            dpdR[i * 3 + j] = dpdM[i * 3 + j] * scales[j];
        }
    }


    float dpdRt[9]; // dpdR Transposed
    dpdRt[0] = dpdR[0]; dpdRt[1] = dpdR[3]; dpdRt[2] = dpdR[6];
    dpdRt[3] = dpdR[1]; dpdRt[4] = dpdR[4]; dpdRt[5] = dpdR[7];
    dpdRt[6] = dpdR[2]; dpdRt[7] = dpdR[5]; dpdRt[8] = dpdR[8];

    float r = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;
    

    float4 dpdq;

    // dp/dr (w.r.t. quat.x)
    dpdq.x = 2.0f * z * (dpdRt[1] - dpdRt[3]) +   // 2z * (J_R^T_01 - J_R^T_10)
             2.0f * y * (dpdRt[6] - dpdRt[2]) +   // 2y * (J_R^T_20 - J_R^T_02)
             2.0f * x * (dpdRt[5] - dpdRt[7]);    // 2x * (J_R^T_12 - J_R^T_21)

    // dp/dx (w.r.t. quat.y)
    dpdq.y = 2.0f * y * (dpdRt[3] + dpdRt[1]) +   // 2y * (J_R^T_10 + J_R^T_01)
             2.0f * z * (dpdRt[6] + dpdRt[2]) +   // 2z * (J_R^T_20 + J_R^T_02)
             2.0f * r * (dpdRt[5] - dpdRt[7]) -   // 2r * (J_R^T_12 - J_R^T_21)
             4.0f * x * (dpdRt[8] + dpdRt[4]);    // 4x * (J_R^T_22 + J_R^T_11)

    // dp/dy (w.r.t. quat.z)
    dpdq.z = 2.0f * x * (dpdRt[3] + dpdRt[1]) +   // 2x * (J_R^T_10 + J_R^T_01)
             2.0f * r * (dpdRt[6] - dpdRt[2]) +   // 2r * (J_R^T_20 - J_R^T_02)
             2.0f * z * (dpdRt[5] + dpdRt[7]) -   // 2z * (J_R^T_12 + J_R^T_21)
             4.0f * y * (dpdRt[8] + dpdRt[0]);    // 4y * (J_R^T_22 + J_R^T_00)

    // dp/dz (w.r.t. quat.w)
    dpdq.w = 2.0f * r * (dpdRt[1] - dpdRt[3]) +   // 2r * (J_R^T_01 - J_R^T_10)
             2.0f * x * (dpdRt[6] + dpdRt[2]) +   // 2x * (J_R^T_20 + J_R^T_02)
             2.0f * y * (dpdRt[5] + dpdRt[7]) -   // 2y * (J_R^T_12 + J_R^T_21)
             4.0f * z * (dpdRt[4] + dpdRt[0]);    // 4z * (J_R^T_11 + J_R^T_00)



    return dpdq;
}

// ============================================================
// Gradient of Sigmoid (for opacity)
// ============================================================

/**
 * Gradient of sigmoid function
 * 
 * sigmoid(x) = 1 / (1 + exp(-x))
 * ∂sigmoid/∂x = sigmoid(x) * (1 - sigmoid(x))
 */
__device__ __forceinline__ float grad_sigmoid(float sigmoid_value) {
    return sigmoid_value * (1.0f - sigmoid_value);
}

// ============================================================
// Gradient of View Direction w.r.t. Mean
// ============================================================

/**
 * Compute gradient of normalized view direction w.r.t. Gaussian mean
 * 
 * view_dir = (mean - cam_pos) / ||mean - cam_pos||
 * 
 * Returns: ∂view_dir/∂mean (3x3 Jacobian stored as 9 floats)
 * 
 * This is needed because SH evaluation depends on view_dir,
 * which depends on mean position.
 */
__device__ __forceinline__ void grad_view_dir_wrt_mean(
    const float3& mean,
    const float3& cam_pos,
    float* jacobian  // Output: 9 floats (3x3 matrix, row-major)
) {
    float3 diff = mean - cam_pos;
    float norm = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    
    if (norm < 1e-8f) {
        // Degenerate case: return identity (or zero?)
        for (int i = 0; i < 9; i++) jacobian[i] = 0.0f;
        jacobian[0] = jacobian[4] = jacobian[8] = 1.0f;
        return;
    }
    
    float inv_norm = 1.0f / norm;
    float inv_norm3 = inv_norm * inv_norm * inv_norm;
    
    // Jacobian = (I / norm) - (diff * diff^T) / norm^3
    // J_ij = δ_ij / norm - diff_i * diff_j / norm^3
    
    jacobian[0] = inv_norm - diff.x * diff.x * inv_norm3;  // ∂vx/∂mx
    jacobian[1] = -diff.x * diff.y * inv_norm3;            // ∂vx/∂my
    jacobian[2] = -diff.x * diff.z * inv_norm3;            // ∂vx/∂mz
    
    jacobian[3] = -diff.y * diff.x * inv_norm3;            // ∂vy/∂mx
    jacobian[4] = inv_norm - diff.y * diff.y * inv_norm3;  // ∂vy/∂my
    jacobian[5] = -diff.y * diff.z * inv_norm3;            // ∂vy/∂mz
    
    jacobian[6] = -diff.z * diff.x * inv_norm3;            // ∂vz/∂mx
    jacobian[7] = -diff.z * diff.y * inv_norm3;            // ∂vz/∂my
    jacobian[8] = inv_norm - diff.z * diff.z * inv_norm3;  // ∂vz/∂mz
}

// ============================================================
// Gradient of Spherical Harmonics w.r.t. Direction
// ============================================================

/**
 * Compute gradient of SH evaluation w.r.t. view direction
 * 
 * Given: rho = Σ_k c_k * Y_k(dir)
 * Compute: ∂rho/∂dir
 * 
 * This differentiates each SH basis function w.r.t. direction.
 * 
 * Returns: ∂SH/∂dir (3D vector)
 */
__device__ __forceinline__ float3 grad_sh_wrt_direction(
    int degree,
    const float* sh_coeffs,
    const float3& dir
) {
    float3 grad = make_float3(0.0f, 0.0f, 0.0f);
    
    // Degree 0: Y_0^0 = 0.28209479... (constant)
    // ∂Y_0^0/∂dir = 0 (no contribution)
    
    if (degree >= 1) {
        // Degree 1 (linear terms):
        // Y_1^-1 = -0.488603 * dir.y
        // Y_1^0  =  0.488603 * dir.z
        // Y_1^1  = -0.488603 * dir.x
        
        const float c1 = 0.48860251190291992f;
        
        // ∂/∂x: only Y_1^1 depends on x
        grad.x += sh_coeffs[3] * (-c1);
        
        // ∂/∂y: only Y_1^-1 depends on y
        grad.y += sh_coeffs[1] * (-c1);
        
        // ∂/∂z: only Y_1^0 depends on z
        grad.z += sh_coeffs[2] * c1;
    }
    
    if (degree >= 2) {
        // Degree 2 (quadratic terms):
        // Y_2^-2 = 1.092548 * x * y
        // Y_2^-1 = -1.092548 * y * z
        // Y_2^0  = 0.315392 * (2z^2 - x^2 - y^2)
        // Y_2^1  = -1.092548 * x * z
        // Y_2^2  = 0.546274 * (x^2 - y^2)
        
        const float c2_0 = 1.0925484305920792f;
        const float c2_1 = 0.31539156525252005f;
        const float c2_2 = 0.54627421529603959f;
        
        float x = dir.x, y = dir.y, z = dir.z;
        
        // ∂/∂x
        grad.x += sh_coeffs[4] * (c2_0 * y);              // from Y_2^-2
        grad.x += sh_coeffs[6] * (c2_1 * (-2.0f * x));    // from Y_2^0
        grad.x += sh_coeffs[7] * (-c2_0 * z);             // from Y_2^1
        grad.x += sh_coeffs[8] * (c2_2 * 2.0f * x);       // from Y_2^2
        
        // ∂/∂y
        grad.y += sh_coeffs[4] * (c2_0 * x);              // from Y_2^-2
        grad.y += sh_coeffs[5] * (-c2_0 * z);             // from Y_2^-1
        grad.y += sh_coeffs[6] * (c2_1 * (-2.0f * y));    // from Y_2^0
        grad.y += sh_coeffs[8] * (c2_2 * (-2.0f * y));    // from Y_2^2
        
        // ∂/∂z
        grad.z += sh_coeffs[5] * (-c2_0 * y);             // from Y_2^-1
        grad.z += sh_coeffs[6] * (c2_1 * 4.0f * z);       // from Y_2^0
        grad.z += sh_coeffs[7] * (-c2_0 * x);             // from Y_2^1
    }
    
    // Degree 3 would continue with cubic terms...
    // For most applications, degree 2 is sufficient
    
    return grad;
}

// ============================================================
// Gradient of Spherical Harmonics w.r.t. Coefficients
// ============================================================

/**
 * Compute gradient of SH evaluation w.r.t. SH coefficients
 * 
 * This is simply the basis function values at the given direction.
 * 
 * SH(dir) = sum_k c_k * Y_k(dir)
 * ∂SH/∂c_k = Y_k(dir)
 * 
 * Stores result in grad_coeffs array
 */
__device__ __forceinline__ void grad_sh_wrt_coeffs(
    int degree,
    const float3& dir,
    float* grad_coeffs  // Output: array of size (degree+1)^2
) {
    // Compute all SH basis functions at this direction
    // This is exactly the same as eval_sh, but we store intermediate values
    
    // Degree 0 (constant)
    grad_coeffs[0] = 0.28209479177387814f;  // sqrt(1/(4*pi))
    
    if (degree == 0) return;
    
    // Degree 1 (linear)
    grad_coeffs[1] = -0.48860251190291992f * dir.y;
    grad_coeffs[2] = 0.48860251190291992f * dir.z;
    grad_coeffs[3] = -0.48860251190291992f * dir.x;
    
    if (degree == 1) return;
    
    // Degree 2 (quadratic)
    float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
    float xy = dir.x * dir.y, xz = dir.x * dir.z, yz = dir.y * dir.z;
    
    grad_coeffs[4] = 1.0925484305920792f * xy;
    grad_coeffs[5] = -1.0925484305920792f * yz;
    grad_coeffs[6] = 0.31539156525252005f * (2.0f * zz - xx - yy);
    grad_coeffs[7] = -1.0925484305920792f * xz;
    grad_coeffs[8] = 0.54627421529603959f * (xx - yy);
    
    if (degree == 2) return;
    
    // Degree 3 and higher: TODO if needed
}

// ============================================================
// Quaternion Gradient Utilities
// ============================================================

/**
 * Compute gradient of rotation matrix w.r.t. quaternion
 * 
 * This is very complex. We use a simplified approach:
 * - Compute finite difference approximation, OR
 * - Use pre-derived analytical formula from literature
 * 
 * For now, this is a placeholder that returns approximate gradients.
 */
__device__ __forceinline__ void grad_rotation_wrt_quaternion(
    const float4& quat,
    const float3& grad_output,  // Gradient signal from upstream
    float4& grad_quat           // Output: gradient w.r.t. quaternion
) {
    // TODO: Implement full quaternion gradient
    // This requires computing ∂R_ij/∂q_k for all i,j,k
    // 
    // For now, we approximate by assuming small perturbations
    // This is the most complex part of the backward pass!
    
    // Placeholder: return zero gradient
    grad_quat = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Note: A full implementation would involve:
    // 1. Computing the 9x4 Jacobian matrix ∂R/∂q
    // 2. Contracting with grad_output
    // 3. Handling quaternion normalization constraint
}

#endif // BACKWARD_UTILS_CUH

