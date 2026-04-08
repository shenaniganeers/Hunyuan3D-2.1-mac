#include <metal_stdlib>
using namespace metal;

// We encode z-buffer as a single uint32 packing:
//   high 20 bits: quantized depth (0..1048575)
//   low 12 bits:  NOT used for face index (too few bits)
//
// Instead, we use TWO atomic buffers:
//   zbuf_depth:  atomic_uint storing quantized depth (higher = farther)
//   zbuf_faceId: atomic_uint storing face index + 1 (0 = empty)
//
// A thread wins the pixel by atomically setting a lower depth.
// We use a compare-and-swap loop on zbuf_depth, and when we win,
// we also store our face index.
//
// NOTE: There is a subtle race between depth and faceId writes.
// Multiple faces may overlap, but the final state will be consistent
// because we always pair the CAS on depth with an immediate write
// to faceId. The last thread to successfully CAS depth also writes
// the correct faceId. A concurrent reader (kernel 2) only runs AFTER
// kernel 1 completes, so it sees a consistent state.

constant uint MAXDEPTH = 0xFFFFFFFFu;

// --- Helper functions ---

inline float calculateSignedArea2(float2 a, float2 b, float2 c) {
    return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}

inline void calculateBarycentricCoordinate(float2 a, float2 b, float2 c, float2 p,
    thread float* barycentric)
{
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0.0f) {
        barycentric[0] = -1.0f;
        barycentric[1] = -1.0f;
        barycentric[2] = -1.0f;
        return;
    }
    float tri_inv = 1.0f / area;
    float beta = beta_tri * tri_inv;
    float gamma = gamma_tri * tri_inv;
    float alpha = 1.0f - beta - gamma;
    barycentric[0] = alpha;
    barycentric[1] = beta;
    barycentric[2] = gamma;
}

inline bool isBarycentricCoordInBounds(thread float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0f && barycentricCoord[0] <= 1.0f &&
           barycentricCoord[1] >= 0.0f && barycentricCoord[1] <= 1.0f &&
           barycentricCoord[2] >= 0.0f && barycentricCoord[2] <= 1.0f;
}

// --- Triangle rasterization (called per-face) ---

inline void rasterizeTriangle(int idx, float3 vt0, float3 vt1, float3 vt2,
    uint width, uint height,
    device atomic_uint* zbuf_depth,
    device atomic_uint* zbuf_faceId,
    device const float* d, float occlusion_truncation, uint use_depth_prior)
{
    float x_min = min(vt0.x, min(vt1.x, vt2.x));
    float x_max = max(vt0.x, max(vt1.x, vt2.x));
    float y_min = min(vt0.y, min(vt1.y, vt2.y));
    float y_max = max(vt0.y, max(vt1.y, vt2.y));

    int px_min = int(x_min);
    int px_max = int(x_max + 1.0f);
    int py_min = int(y_min);
    int py_max = int(y_max + 1.0f);

    for (int px = px_min; px <= px_max; ++px) {
        if (px < 0 || px >= int(width))
            continue;
        for (int py = py_min; py <= py_max; ++py) {
            if (py < 0 || py >= int(height))
                continue;

            float2 vt = float2(float(px) + 0.5f, float(py) + 0.5f);
            float baryCentricCoordinate[3];
            calculateBarycentricCoordinate(float2(vt0.x, vt0.y), float2(vt1.x, vt1.y),
                                           float2(vt2.x, vt2.y), vt, baryCentricCoordinate);

            if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
                int pixel = py * int(width) + px;

                float depth = baryCentricCoordinate[0] * vt0.z +
                              baryCentricCoordinate[1] * vt1.z +
                              baryCentricCoordinate[2] * vt2.z;

                float depth_thres = 0.0f;
                if (use_depth_prior && d != nullptr) {
                    depth_thres = d[pixel] * 0.49999f + 0.5f + occlusion_truncation;
                }

                if (depth < depth_thres)
                    continue;

                // Quantize depth to uint32 — depth is in ~[0,1] after normalization
                // Use full 32-bit range for maximum precision
                uint z_quantize = uint(clamp(depth, 0.0f, 1.0f) * float(MAXDEPTH - 1));

                // Atomic min on depth: try to set our depth if it's smaller
                uint current = atomic_load_explicit(&zbuf_depth[pixel], memory_order_relaxed);
                while (z_quantize < current) {
                    if (atomic_compare_exchange_weak_explicit(&zbuf_depth[pixel], &current, z_quantize,
                            memory_order_relaxed, memory_order_relaxed)) {
                        // Won the depth test — store our face index
                        atomic_store_explicit(&zbuf_faceId[pixel], uint(idx + 1), memory_order_relaxed);
                        break;
                    }
                    // current is updated by CAS on failure, loop continues
                }
            }
        }
    }
}

// --- Kernel 1: Rasterize image coordinates (one thread per face) ---

kernel void rasterizeImagecoords(
    device const float* V [[buffer(0)]],         // vertices [N,4] float32
    device const int* F [[buffer(1)]],           // faces [M,3] int32
    device const float* D [[buffer(2)]],         // depth prior (may be empty)
    device atomic_uint* zbuf_depth [[buffer(3)]],  // depth buffer [H*W] uint32
    device atomic_uint* zbuf_faceId [[buffer(4)]], // face index buffer [H*W] uint32
    constant uint& width [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& num_faces [[buffer(7)]],
    constant float& occlusion_trunc [[buffer(8)]],
    constant uint& use_depth_prior [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_faces)
        return;

    int f = int(tid);

    device const float* vt0_ptr = V + (F[f * 3]     * 4);
    device const float* vt1_ptr = V + (F[f * 3 + 1] * 4);
    device const float* vt2_ptr = V + (F[f * 3 + 2] * 4);

    float3 vt0 = float3(
        (vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
        (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * float(height - 1) + 0.5f,
        vt0_ptr[2] / vt0_ptr[3] * 0.49999f + 0.5f
    );
    float3 vt1 = float3(
        (vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
        (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * float(height - 1) + 0.5f,
        vt1_ptr[2] / vt1_ptr[3] * 0.49999f + 0.5f
    );
    float3 vt2 = float3(
        (vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
        (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * float(height - 1) + 0.5f,
        vt2_ptr[2] / vt2_ptr[3] * 0.49999f + 0.5f
    );

    device const float* d_ptr = (use_depth_prior != 0) ? D : nullptr;
    rasterizeTriangle(f, vt0, vt1, vt2, width, height, zbuf_depth, zbuf_faceId, d_ptr, occlusion_trunc, use_depth_prior);
}

// --- Kernel 2: Compute barycentric coordinates (one thread per pixel) ---

kernel void barycentricFromImgcoord(
    device const float* V [[buffer(0)]],
    device const int* F [[buffer(1)]],
    device int* findices [[buffer(2)]],
    device const uint* zbuf_faceId [[buffer(3)]],  // face index buffer (non-atomic read)
    device float* barycentric_map [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& num_vertices [[buffer(7)]],
    constant uint& num_faces [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint pix = tid;
    if (pix >= width * height)
        return;

    uint f = zbuf_faceId[pix];
    if (f == 0) {
        // No face covers this pixel
        findices[pix] = 0;
        barycentric_map[pix * 3]     = 0.0f;
        barycentric_map[pix * 3 + 1] = 0.0f;
        barycentric_map[pix * 3 + 2] = 0.0f;
        return;
    }

    findices[pix] = int(f);
    int fi = int(f) - 1;

    float barycentric[3] = {0.0f, 0.0f, 0.0f};

    if (fi >= 0) {
        float2 vt = float2(float(pix % width) + 0.5f, float(pix / width) + 0.5f);

        device const float* vt0_ptr = V + (F[fi * 3]     * 4);
        device const float* vt1_ptr = V + (F[fi * 3 + 1] * 4);
        device const float* vt2_ptr = V + (F[fi * 3 + 2] * 4);

        float2 vt0 = float2(
            (vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
            (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * float(height - 1) + 0.5f
        );
        float2 vt1 = float2(
            (vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
            (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * float(height - 1) + 0.5f
        );
        float2 vt2 = float2(
            (vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * float(width - 1) + 0.5f,
            (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * float(height - 1) + 0.5f
        );

        calculateBarycentricCoordinate(vt0, vt1, vt2, vt, barycentric);

        // Perspective-correct interpolation
        barycentric[0] = barycentric[0] / vt0_ptr[3];
        barycentric[1] = barycentric[1] / vt1_ptr[3];
        barycentric[2] = barycentric[2] / vt2_ptr[3];
        float w = 1.0f / (barycentric[0] + barycentric[1] + barycentric[2]);
        barycentric[0] *= w;
        barycentric[1] *= w;
        barycentric[2] *= w;
    }

    barycentric_map[pix * 3]     = barycentric[0];
    barycentric_map[pix * 3 + 1] = barycentric[1];
    barycentric_map[pix * 3 + 2] = barycentric[2];
}
