#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Math utilities
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float len = length(v);
    return v * (1.0f / (len + 1e-8f));
}

// Quaternion to rotation matrix
__device__ __forceinline__ void quat_to_rotmat(
    const float4& q,  // quaternion (w, x, y, z)
    float* R          // output 3x3 rotation matrix (row-major)
) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    // Normalize quaternion
    float norm = sqrtf(w*w + x*x + y*y + z*z);
    w /= norm; x /= norm; y /= norm; z /= norm;
    
    R[0] = 1.0f - 2.0f*(y*y + z*z);
    R[1] = 2.0f*(x*y - w*z);
    R[2] = 2.0f*(x*z + w*y);
    
    R[3] = 2.0f*(x*y + w*z);
    R[4] = 1.0f - 2.0f*(x*x + z*z);
    R[5] = 2.0f*(y*z - w*x);
    
    R[6] = 2.0f*(x*z - w*y);
    R[7] = 2.0f*(y*z + w*x);
    R[8] = 1.0f - 2.0f*(x*x + y*y);
}

// Matrix-vector multiplication (3x3 * 3x1)
__device__ __forceinline__ float3 matvec3(const float* M, const float3& v) {
    return make_float3(
        M[0]*v.x + M[1]*v.y + M[2]*v.z,
        M[3]*v.x + M[4]*v.y + M[5]*v.z,
        M[6]*v.x + M[7]*v.y + M[8]*v.z
    );
}

// Ray-AABB intersection
__device__ __forceinline__ bool ray_aabb_intersect(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& bbox_min,
    const float3& bbox_max,
    float& t_min,
    float& t_max
) {
    float3 inv_dir = make_float3(
        1.0f / (ray_dir.x + 1e-8f),
        1.0f / (ray_dir.y + 1e-8f),
        1.0f / (ray_dir.z + 1e-8f)
    );
    
    float3 t0 = (bbox_min - ray_origin) * inv_dir;
    float3 t1 = (bbox_max - ray_origin) * inv_dir;
    
    float3 tmin_vec = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax_vec = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    
    t_min = fmaxf(fmaxf(tmin_vec.x, tmin_vec.y), tmin_vec.z);
    t_max = fminf(fminf(tmax_vec.x, tmax_vec.y), tmax_vec.z);
    
    return t_max >= t_min && t_max >= 0.0f;
}

// Gaussian PDF evaluation
__device__ __forceinline__ float eval_gaussian_pdf(
    const float3& pos,
    const float3& mean,
    const float3& scale,
    const float4& quat
) {
    float3 diff = pos - mean;
    
    // Build rotation matrix
    float R[9];
    quat_to_rotmat(quat, R);
    
    // Rotate diff: T = R^T * diff
    float3 T = make_float3(
        R[0]*diff.x + R[3]*diff.y + R[6]*diff.z,
        R[1]*diff.x + R[4]*diff.y + R[7]*diff.z,
        R[2]*diff.x + R[5]*diff.y + R[8]*diff.z
    );
    
    // Mahalanobis distance: (T / scale)^2
    float mahal_sq = (T.x/scale.x)*(T.x/scale.x) + 
                     (T.y/scale.y)*(T.y/scale.y) + 
                     (T.z/scale.z)*(T.z/scale.z);
    
    return expf(-0.5f * mahal_sq);
}

#endif // CUDA_UTILS_CUH


