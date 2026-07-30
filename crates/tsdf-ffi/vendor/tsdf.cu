// tsdf.cu
//
// Dense sliding-window TSDF. See tsdf.h for the public C ABI; this file
// implements the device-side accumulation and extraction kernels plus the
// host-side state machine that shifts the grid origin on set_focus.
//
// The grid is a dense 3D array of D³ voxels (D = grid_dim). Voxel (vx,vy,vz)
// occupies world-space extent:
//   [(origin_vx + vx) * vs, (origin_vx + vx + 1) * vs)   (and analogous y, z)
// where vs = voxel_size_m. `origin_vx/vy/vz` are the world-voxel-coord offsets
// of the grid's "voxel (0,0,0)" — they shift on set_focus so the active
// window follows the camera.
//
// On slide, each voxel's world position is computed from its current
// (vx,vy,vz) + current origin. Voxels whose new L-inf distance to the focus
// exceeds commit_distance are emitted to the committed-points buffer, then
// cleared in the grid so fresh voxels start at zero.
//
// Performance notes:
// - Atomic float-adds on sum_* require sm_60+. We target sm_89 (L4) and
//   sm_120 (Blackwell), both fine.
// - SoA layout chosen so atomicAdd accesses a single 4-byte word per thread,
//   avoiding AoS-induced memory divergence.
// - Integration is O(n_points) with no search: voxel index is computed
//   directly from world coord.

#include "tsdf.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// ===== device-side layout =====

struct TsdfGrid {
    float voxel_size_m;
    int   grid_dim;          // D: voxels per side
    float commit_distance_m;

    // Origin: world-voxel-coord offset of grid voxel (0,0,0). Shifts on
    // set_focus so the grid window follows the camera.
    int   origin_vx;
    int   origin_vy;
    int   origin_vz;
    bool  origin_initialized;
    // When true, set_focus does NOT slide/shift the origin. Used by the
    // multi-tile manager: each tile occupies a fixed sub-volume of the
    // cube, and shifting is owned by the multi-tile layer (not per-tile).
    // Default false (single-grid sliding-window mode unchanged).
    bool  pin_origin;
    // Per-instance CUDA stream. All kernel launches and memcpy operations
    // run on this stream so multiple TsdfGrid instances can execute in
    // parallel on the same device (used by the multi-tile manager).
    cudaStream_t stream;  // set to true on first set_focus

    // Device SoA buffers, each D^3 entries.
    float* d_sum_x;
    float* d_sum_y;
    float* d_sum_z;
    float* d_sum_r;
    float* d_sum_g;
    float* d_sum_b;
    float* d_sum_sq_x;  // for variance (spread) gate
    float* d_sum_sq_y;
    float* d_sum_sq_z;
    float* d_weight;
    unsigned int* d_count;

    // Device-side committed-points buffer. Points written here by the slide
    // kernel. Drained by tsdf_grid_extract_points(... drain_committed=1).
    //   layout: xyz interleaved f32 | rgb interleaved u8 | separate counters
    float*         d_committed_xyz;
    unsigned char* d_committed_rgb;
    unsigned int*  d_committed_count_ptr;  // single u32 in device mem
    int            committed_cap;

    // Temporaries for point upload (reused across calls; grown on demand).
    float*         d_tmp_positions;
    unsigned char* d_tmp_colors;
    float*         d_tmp_weights;
    int            tmp_capacity;

    // Per-voxel weight saturation cap. Once weight_buf[idx] >= weight_cap,
    // integrate_points_kernel skips further accumulation into that voxel.
    // This decreases TSDF "overlap": once a high-confidence voxel has
    // absorbed enough samples, additional chunks (which may carry small
    // pose drift) don't keep smearing the surface. <=0 disables (uncapped).
    float          weight_cap;
};

// ===== host-side helpers =====

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[tsdf.cu] CUDA error %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return _e;                                                         \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_NULL(expr)                                                  \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[tsdf.cu] CUDA error %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return nullptr;                                                    \
        }                                                                      \
    } while (0)

static inline int grid_voxel_count(int dim) { return dim * dim * dim; }

// Linear index for voxel (vx, vy, vz) in the grid.
__device__ __host__ static inline int lin_idx(int vx, int vy, int vz, int dim) {
    return (vz * dim + vy) * dim + vx;
}

// ===== integration kernel =====

__global__ void integrate_points_kernel(
    const float* __restrict__ positions,
    const unsigned char* __restrict__ colors,
    const float* __restrict__ weights,
    int n_points,
    // grid params
    float voxel_size_m,
    int grid_dim,
    int origin_vx, int origin_vy, int origin_vz,
    // saturation cap; <=0 disables. See TsdfGrid::weight_cap.
    float weight_cap,
    // grid buffers (SoA)
    float* sum_x, float* sum_y, float* sum_z,
    float* sum_r, float* sum_g, float* sum_b,
    float* sum_sq_x, float* sum_sq_y, float* sum_sq_z,
    float* weight_buf, unsigned int* count_buf
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

    float px = positions[tid * 3 + 0];
    float py = positions[tid * 3 + 1];
    float pz = positions[tid * 3 + 2];

    // Reject non-finite points silently.
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) return;

    float inv_vs = 1.0f / voxel_size_m;
    int world_vx = (int)floorf(px * inv_vs);
    int world_vy = (int)floorf(py * inv_vs);
    int world_vz = (int)floorf(pz * inv_vs);

    // Translate to grid-local voxel coords.
    int vx = world_vx - origin_vx;
    int vy = world_vy - origin_vy;
    int vz = world_vz - origin_vz;

    // Reject points outside the current window.
    if (vx < 0 || vx >= grid_dim) return;
    if (vy < 0 || vy >= grid_dim) return;
    if (vz < 0 || vz >= grid_dim) return;

    int idx = lin_idx(vx, vy, vz, grid_dim);

    float w = weights ? weights[tid] : 1.0f;
    if (!(w > 0.0f) || !isfinite(w)) return;

    // Saturation gate: skip integration into voxels that already hold
    // enough confidence. The read is racy w.r.t. concurrent atomicAdds in
    // this same kernel launch, so we may overshoot the cap by up to one
    // batch's worth — that's acceptable. Across batches the cap holds
    // tightly. Cuts ghosting/stringing in dense regions where many chunks
    // re-observe the same surface.
    if (weight_cap > 0.0f && weight_buf[idx] >= weight_cap) return;

    unsigned char r = 128, g = 128, b = 128;
    if (colors) {
        r = colors[tid * 3 + 0];
        g = colors[tid * 3 + 1];
        b = colors[tid * 3 + 2];
    }

    // Atomic accumulation. On modern GPUs atomicAdd(float*, float) is a
    // single-instruction reduction.
    atomicAdd(&sum_x[idx], px * w);
    atomicAdd(&sum_y[idx], py * w);
    atomicAdd(&sum_z[idx], pz * w);
    atomicAdd(&sum_sq_x[idx], px * px);
    atomicAdd(&sum_sq_y[idx], py * py);
    atomicAdd(&sum_sq_z[idx], pz * pz);
    atomicAdd(&sum_r[idx], (float)r * w);
    atomicAdd(&sum_g[idx], (float)g * w);
    atomicAdd(&sum_b[idx], (float)b * w);
    atomicAdd(&weight_buf[idx], w);
    atomicAdd(&count_buf[idx], 1u);
}

// ===== slide kernel =====
//
// For each grid voxel: compute its world-space center, check L-inf distance
// to the new focus. If > commit_distance, emit to committed buffer and clear
// the voxel. Then the new origin is set by the host after this kernel.
//
// NOTE: this kernel operates on the CURRENT origin. After it runs, the host
// zeroes cells corresponding to the shifted-in region and updates the origin.
// For v0 we take the simpler (correct-but-wasteful) approach of FULL slide:
// commit everything and reset origin centered on new focus. Partial-slide
// optimization (shift in place) is a follow-up.

__global__ void slide_full_commit_kernel(
    int grid_dim,
    float voxel_size_m,
    int origin_vx, int origin_vy, int origin_vz,
    const float* sum_x, const float* sum_y, const float* sum_z,
    const float* sum_r, const float* sum_g, const float* sum_b,
    const float* weight_buf, const unsigned int* count_buf,
    float min_weight,
    int min_count,
    float* committed_xyz,
    unsigned char* committed_rgb,
    unsigned int* committed_count_ptr,
    int committed_cap
) {
    int total = grid_dim * grid_dim * grid_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    float w = weight_buf[tid];
    unsigned int c = count_buf[tid];
    if (w < min_weight || (int)c < min_count) return;

    float inv_w = 1.0f / w;
    float px = sum_x[tid] * inv_w;
    float py = sum_y[tid] * inv_w;
    float pz = sum_z[tid] * inv_w;

    // Acquire a slot in the committed buffer.
    unsigned int slot = atomicAdd(committed_count_ptr, 1u);
    if ((int)slot >= committed_cap) {
        // Overflow — undo the increment so subsequent threads see the true
        // cap. Lost voxels are silently dropped in v0 (matches tsdf.h
        // contract: committed buffer overflow drops, not retains).
        // Roll back isn't strictly necessary; the cap check above handles it.
        return;
    }

    float cr_f = sum_r[tid] * inv_w;
    float cg_f = sum_g[tid] * inv_w;
    float cb_f = sum_b[tid] * inv_w;
    unsigned char cr = (unsigned char)fminf(fmaxf(cr_f, 0.0f), 255.0f);
    unsigned char cg = (unsigned char)fminf(fmaxf(cg_f, 0.0f), 255.0f);
    unsigned char cb = (unsigned char)fminf(fmaxf(cb_f, 0.0f), 255.0f);

    committed_xyz[slot * 3 + 0] = px;
    committed_xyz[slot * 3 + 1] = py;
    committed_xyz[slot * 3 + 2] = pz;
    committed_rgb[slot * 3 + 0] = cr;
    committed_rgb[slot * 3 + 1] = cg;
    committed_rgb[slot * 3 + 2] = cb;
}

// ===== extract kernel =====
//
// For currently-active voxels (not drained to committed), apply gates and
// write the surviving voxel's mean position + color to the output buffer.

__global__ void extract_active_kernel(
    int grid_dim,
    float voxel_size_m,
    const float* sum_x, const float* sum_y, const float* sum_z,
    const float* sum_sq_x, const float* sum_sq_y, const float* sum_sq_z,
    const float* sum_r, const float* sum_g, const float* sum_b,
    const float* weight_buf,
    const unsigned int* count_buf,
    float min_weight,
    int min_count,
    float max_spread_frac,
    // output
    float* out_xyz,
    unsigned char* out_rgb,
    unsigned int* out_count_ptr,
    int out_cap
) {
    int total = grid_dim * grid_dim * grid_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    float w = weight_buf[tid];
    unsigned int c = count_buf[tid];
    if (w < min_weight) return;
    if ((int)c < min_count) return;

    float inv_w = 1.0f / w;
    float mx = sum_x[tid] * inv_w;
    float my = sum_y[tid] * inv_w;
    float mz = sum_z[tid] * inv_w;

    // Spread gate — matches CPU-side tsdf_fuse_volumes. We use the weighted
    // mean as a stand-in for the unweighted mean here; for most callers
    // weights are near-uniform so the two coincide. If we ever want to
    // match the CPU variance formula exactly we'd need a separate unweighted
    // sum buffer, which doubles memory for a marginal correctness gain.
    if (max_spread_frac < 1.0f && c >= 2) {
        float cn = (float)c;
        float ux = sum_x[tid] * inv_w;
        float uy = sum_y[tid] * inv_w;
        float uz = sum_z[tid] * inv_w;
        float var_x = fmaxf(sum_sq_x[tid] / cn - ux * ux, 0.0f);
        float var_y = fmaxf(sum_sq_y[tid] / cn - uy * uy, 0.0f);
        float var_z = fmaxf(sum_sq_z[tid] / cn - uz * uz, 0.0f);
        float max_std = sqrtf(fmaxf(fmaxf(var_x, var_y), var_z));
        float max_spread = voxel_size_m * max_spread_frac;
        if (max_std > max_spread) return;
    }

    unsigned int slot = atomicAdd(out_count_ptr, 1u);
    if ((int)slot >= out_cap) {
        // Caller must ensure buffer is large enough; we still cap-check.
        return;
    }

    float cr_f = sum_r[tid] * inv_w;
    float cg_f = sum_g[tid] * inv_w;
    float cb_f = sum_b[tid] * inv_w;
    unsigned char cr = (unsigned char)fminf(fmaxf(cr_f, 0.0f), 255.0f);
    unsigned char cg = (unsigned char)fminf(fmaxf(cg_f, 0.0f), 255.0f);
    unsigned char cb = (unsigned char)fminf(fmaxf(cb_f, 0.0f), 255.0f);

    out_xyz[slot * 3 + 0] = mx;
    out_xyz[slot * 3 + 1] = my;
    out_xyz[slot * 3 + 2] = mz;
    out_rgb[slot * 3 + 0] = cr;
    out_rgb[slot * 3 + 1] = cg;
    out_rgb[slot * 3 + 2] = cb;
}

// ===== extern "C" API implementation =====

static int alloc_grid_buffers(TsdfGrid* g) {
    int vox = grid_voxel_count(g->grid_dim);
    size_t bytes_f = (size_t)vox * sizeof(float);
    size_t bytes_u = (size_t)vox * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&g->d_sum_x, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_y, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_z, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_r, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_g, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_b, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_sq_x, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_sq_y, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_sum_sq_z, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_weight, bytes_f));
    CUDA_CHECK(cudaMalloc(&g->d_count,  bytes_u));

    CUDA_CHECK(cudaMemset(g->d_sum_x, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_y, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_z, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_r, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_g, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_b, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_sq_x, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_sq_y, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_sum_sq_z, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_weight, 0, bytes_f));
    CUDA_CHECK(cudaMemset(g->d_count,  0, bytes_u));
    return 0;
}

extern "C" TsdfGrid* tsdf_grid_create(
    float voxel_size_m,
    int grid_dim,
    float commit_distance_m,
    int committed_cap_points
) {
    if (grid_dim <= 0 || voxel_size_m <= 0.0f || committed_cap_points <= 0) {
        return nullptr;
    }
    TsdfGrid* g = new TsdfGrid();
    g->voxel_size_m = voxel_size_m;
    g->grid_dim = grid_dim;
    g->commit_distance_m = commit_distance_m;
    g->origin_vx = g->origin_vy = g->origin_vz = 0;
    g->origin_initialized = false;
    g->pin_origin = false;
    g->committed_cap = committed_cap_points;
    g->tmp_capacity = 0;
    g->d_tmp_positions = nullptr;
    g->d_tmp_colors = nullptr;
    g->d_tmp_weights = nullptr;
    g->weight_cap = 0.0f;  // disabled by default; set via tsdf_grid_set_weight_cap
    // Each grid owns a non-blocking stream so multi-tile orchestrators can
    // launch concurrent kernel streams across tiles on the same device.
    if (cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete g;
        return nullptr;
    }

    if (alloc_grid_buffers(g) != 0) {
        cudaStreamDestroy(g->stream);
        delete g;
        return nullptr;
    }

    CUDA_CHECK_NULL(cudaMalloc(&g->d_committed_xyz,
                                (size_t)committed_cap_points * 3 * sizeof(float)));
    CUDA_CHECK_NULL(cudaMalloc(&g->d_committed_rgb,
                                (size_t)committed_cap_points * 3 * sizeof(unsigned char)));
    CUDA_CHECK_NULL(cudaMalloc(&g->d_committed_count_ptr, sizeof(unsigned int)));
    CUDA_CHECK_NULL(cudaMemset(g->d_committed_count_ptr, 0, sizeof(unsigned int)));

    return g;
}

extern "C" void tsdf_grid_destroy(TsdfGrid* g) {
    if (!g) return;
    // Drain any pending work on the stream before freeing buffers.
    cudaStreamSynchronize(g->stream);
    cudaFree(g->d_sum_x); cudaFree(g->d_sum_y); cudaFree(g->d_sum_z);
    cudaFree(g->d_sum_r); cudaFree(g->d_sum_g); cudaFree(g->d_sum_b);
    cudaFree(g->d_sum_sq_x); cudaFree(g->d_sum_sq_y); cudaFree(g->d_sum_sq_z);
    cudaFree(g->d_weight); cudaFree(g->d_count);
    cudaFree(g->d_committed_xyz);
    cudaFree(g->d_committed_rgb);
    cudaFree(g->d_committed_count_ptr);
    if (g->d_tmp_positions) cudaFree(g->d_tmp_positions);
    if (g->d_tmp_colors)    cudaFree(g->d_tmp_colors);
    if (g->d_tmp_weights)   cudaFree(g->d_tmp_weights);
    cudaStreamDestroy(g->stream);
    delete g;
}

static int ensure_tmp_capacity(TsdfGrid* g, int n_points) {
    if (n_points <= g->tmp_capacity) return 0;
    // Grow to 1.5× requested to avoid churn.
    int new_cap = n_points + n_points / 2;

    if (g->d_tmp_positions) cudaFree(g->d_tmp_positions);
    if (g->d_tmp_colors)    cudaFree(g->d_tmp_colors);
    if (g->d_tmp_weights)   cudaFree(g->d_tmp_weights);
    g->d_tmp_positions = nullptr;
    g->d_tmp_colors    = nullptr;
    g->d_tmp_weights   = nullptr;

    CUDA_CHECK(cudaMalloc(&g->d_tmp_positions, (size_t)new_cap * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g->d_tmp_colors,    (size_t)new_cap * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&g->d_tmp_weights,   (size_t)new_cap * sizeof(float)));
    g->tmp_capacity = new_cap;
    return 0;
}

extern "C" int tsdf_grid_add_points(
    TsdfGrid* g,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points
) {
    if (!g || !positions || n_points <= 0) return 1;
    if (!g->origin_initialized) {
        // Auto-center on the centroid of the first batch so the grid has a
        // sane starting window. Caller is still encouraged to call
        // tsdf_grid_set_focus explicitly.
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int i = 0; i < n_points; ++i) {
            cx += positions[i*3+0];
            cy += positions[i*3+1];
            cz += positions[i*3+2];
        }
        cx /= n_points; cy /= n_points; cz /= n_points;
        float inv_vs = 1.0f / g->voxel_size_m;
        int cvx = (int)floorf((float)cx * inv_vs);
        int cvy = (int)floorf((float)cy * inv_vs);
        int cvz = (int)floorf((float)cz * inv_vs);
        g->origin_vx = cvx - g->grid_dim / 2;
        g->origin_vy = cvy - g->grid_dim / 2;
        g->origin_vz = cvz - g->grid_dim / 2;
        g->origin_initialized = true;
    }

    if (ensure_tmp_capacity(g, n_points) != 0) return 2;

    CUDA_CHECK(cudaMemcpyAsync(g->d_tmp_positions, positions,
                          (size_t)n_points * 3 * sizeof(float),
                          cudaMemcpyHostToDevice, g->stream));
    if (colors) {
        CUDA_CHECK(cudaMemcpyAsync(g->d_tmp_colors, colors,
                              (size_t)n_points * 3 * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, g->stream));
    }
    if (weights) {
        CUDA_CHECK(cudaMemcpyAsync(g->d_tmp_weights, weights,
                              (size_t)n_points * sizeof(float),
                              cudaMemcpyHostToDevice, g->stream));
    }

    int threads = 256;
    int blocks = (n_points + threads - 1) / threads;
    integrate_points_kernel<<<blocks, threads, 0, g->stream>>>(
        g->d_tmp_positions,
        colors  ? g->d_tmp_colors  : nullptr,
        weights ? g->d_tmp_weights : nullptr,
        n_points,
        g->voxel_size_m, g->grid_dim,
        g->origin_vx, g->origin_vy, g->origin_vz,
        g->weight_cap,
        g->d_sum_x, g->d_sum_y, g->d_sum_z,
        g->d_sum_r, g->d_sum_g, g->d_sum_b,
        g->d_sum_sq_x, g->d_sum_sq_y, g->d_sum_sq_z,
        g->d_weight, g->d_count
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Pin-origin variant: explicit world-corner-voxel-coords. Used by the
// multi-tile manager so each tile occupies a fixed sub-volume; subsequent
// set_focus calls become no-ops (origin is owned by the multi-tile layer).
extern "C" int tsdf_grid_set_origin(TsdfGrid* g, int origin_vx, int origin_vy, int origin_vz) {
    if (!g) return -1;
    g->origin_vx = origin_vx;
    g->origin_vy = origin_vy;
    g->origin_vz = origin_vz;
    g->origin_initialized = true;
    g->pin_origin = true;
    return 0;
}

// Drain every active voxel passing min_weight=1e-6 into the per-tile
// committed buffer, zero all voxel buffers, and set a new origin. Used
// by the multi-tile manager when the cube slides: evicted tiles dump
// their voxels (which a later extract drain returns to host) and then
// re-pin to a fresh world position on the leading edge of the cube.
//
// Returns the number of voxels committed by this drain (>=0), or -1
// on error.
extern "C" int tsdf_grid_drain_and_reset(TsdfGrid* g, int new_origin_vx, int new_origin_vy, int new_origin_vz) {
    if (!g) return -1;

    unsigned int committed_before = 0;
    CUDA_CHECK(cudaMemcpyAsync(&committed_before, g->d_committed_count_ptr,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_CHECK(cudaStreamSynchronize(g->stream));

    int total = grid_voxel_count(g->grid_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    slide_full_commit_kernel<<<blocks, threads, 0, g->stream>>>(
        g->grid_dim, g->voxel_size_m,
        g->origin_vx, g->origin_vy, g->origin_vz,
        g->d_sum_x, g->d_sum_y, g->d_sum_z,
        g->d_sum_r, g->d_sum_g, g->d_sum_b,
        g->d_weight, g->d_count,
        /* min_weight */ 1e-6f,
        /* min_count  */ 1,
        g->d_committed_xyz, g->d_committed_rgb, g->d_committed_count_ptr,
        g->committed_cap
    );
    CUDA_CHECK(cudaGetLastError());

    size_t bytes_f = (size_t)total * sizeof(float);
    size_t bytes_u = (size_t)total * sizeof(unsigned int);
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_x, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_y, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_z, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_r, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_g, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_b, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_x, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_y, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_z, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_weight, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_count,  0, bytes_u, g->stream));

    g->origin_vx = new_origin_vx;
    g->origin_vy = new_origin_vy;
    g->origin_vz = new_origin_vz;
    g->origin_initialized = true;
    g->pin_origin = true;

    unsigned int committed_after = 0;
    CUDA_CHECK(cudaMemcpyAsync(&committed_after, g->d_committed_count_ptr,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_CHECK(cudaStreamSynchronize(g->stream));
    return (int)(committed_after - committed_before);
}

extern "C" int tsdf_grid_set_focus(TsdfGrid* g, float cx, float cy, float cz) {
    if (!g) return -1;
    // Multi-tile callers pin origin once per tile; subsequent set_focus must
    // not slide. Origin movement is owned by the manager.
    if (g->pin_origin) return 0;

    float inv_vs = 1.0f / g->voxel_size_m;
    int cvx = (int)floorf(cx * inv_vs);
    int cvy = (int)floorf(cy * inv_vs);
    int cvz = (int)floorf(cz * inv_vs);

    int desired_origin_x = cvx - g->grid_dim / 2;
    int desired_origin_y = cvy - g->grid_dim / 2;
    int desired_origin_z = cvz - g->grid_dim / 2;

    if (!g->origin_initialized) {
        g->origin_vx = desired_origin_x;
        g->origin_vy = desired_origin_y;
        g->origin_vz = desired_origin_z;
        g->origin_initialized = true;
        return 0;
    }

    int shift_x = desired_origin_x - g->origin_vx;
    int shift_y = desired_origin_y - g->origin_vy;
    int shift_z = desired_origin_z - g->origin_vz;
    if (shift_x == 0 && shift_y == 0 && shift_z == 0) return 0;

    // v0 FULL slide: commit every active voxel, clear grid, set new origin.
    // Future optimization: partial slide (move only voxels crossing the
    // window boundary). For phone-speed motion at 15fps * 0.5m window, full
    // slide triggers rarely (only on > 0.25 m camera jumps) so v0 is fine.
    unsigned int committed_before;
    CUDA_CHECK(cudaMemcpyAsync(&committed_before, g->d_committed_count_ptr,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_CHECK(cudaStreamSynchronize(g->stream));

    int total = grid_voxel_count(g->grid_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    slide_full_commit_kernel<<<blocks, threads, 0, g->stream>>>(
        g->grid_dim, g->voxel_size_m,
        g->origin_vx, g->origin_vy, g->origin_vz,
        g->d_sum_x, g->d_sum_y, g->d_sum_z,
        g->d_sum_r, g->d_sum_g, g->d_sum_b,
        g->d_weight, g->d_count,
        /* min_weight */ 1e-6f,  // emit any voxel that saw any point
        /* min_count  */ 1,
        g->d_committed_xyz, g->d_committed_rgb, g->d_committed_count_ptr,
        g->committed_cap
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(g->stream));

    // Clear grid buffers for the new origin.
    size_t bytes_f = (size_t)total * sizeof(float);
    size_t bytes_u = (size_t)total * sizeof(unsigned int);
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_x, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_y, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_z, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_r, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_g, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_b, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_x, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_y, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_sum_sq_z, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_weight, 0, bytes_f, g->stream));
    CUDA_CHECK(cudaMemsetAsync(g->d_count,  0, bytes_u, g->stream));

    g->origin_vx = desired_origin_x;
    g->origin_vy = desired_origin_y;
    g->origin_vz = desired_origin_z;

    unsigned int committed_after;
    CUDA_CHECK(cudaMemcpyAsync(&committed_after, g->d_committed_count_ptr,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_CHECK(cudaStreamSynchronize(g->stream));
    return (int)(committed_after - committed_before);
}

extern "C" int tsdf_grid_extract_points(
    TsdfGrid* g,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float max_spread_frac,
    int drain_committed
) {
    if (!g || !buffer_xyz || !buffer_rgb || buffer_cap <= 0) return -1;

    // Total capacity we'll need: (active voxels passing gate) + (committed
    // already queued, if drain). We don't know active count a priori, so we
    // extract into a device scratch buffer first, then copy the written
    // prefix to host.
    unsigned int committed_count;
    CUDA_CHECK(cudaMemcpyAsync(&committed_count, g->d_committed_count_ptr,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_CHECK(cudaStreamSynchronize(g->stream));

    // Reuse committed_xyz/rgb as the output buffer when drain=1 (it already
    // holds the committed part). For clarity, we always use a separate
    // device scratch sized at buffer_cap.
    float*         d_out_xyz = nullptr;
    unsigned char* d_out_rgb = nullptr;
    unsigned int*  d_out_cnt = nullptr;
    if (cudaMalloc(&d_out_xyz, (size_t)buffer_cap * 3 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_out_rgb, (size_t)buffer_cap * 3 * sizeof(unsigned char)) != cudaSuccess) { cudaFree(d_out_xyz); return -1; }
    if (cudaMalloc(&d_out_cnt, sizeof(unsigned int)) != cudaSuccess) { cudaFree(d_out_xyz); cudaFree(d_out_rgb); return -1; }
    cudaMemsetAsync(d_out_cnt, 0, sizeof(unsigned int), g->stream);

    // 1. Copy committed points into head of output if draining.
    if (drain_committed && committed_count > 0) {
        unsigned int to_copy = committed_count;
        if ((int)to_copy > buffer_cap) to_copy = (unsigned int)buffer_cap;
        cudaMemcpyAsync(d_out_xyz,
                   g->d_committed_xyz,
                   (size_t)to_copy * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, g->stream);
        cudaMemcpyAsync(d_out_rgb,
                   g->d_committed_rgb,
                   (size_t)to_copy * 3 * sizeof(unsigned char),
                   cudaMemcpyDeviceToDevice, g->stream);
        cudaMemcpyAsync(d_out_cnt, &to_copy, sizeof(unsigned int), cudaMemcpyHostToDevice, g->stream);
    }

    // 2. Launch extract-active kernel to append gate-passing active voxels.
    int total = grid_voxel_count(g->grid_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    extract_active_kernel<<<blocks, threads, 0, g->stream>>>(
        g->grid_dim, g->voxel_size_m,
        g->d_sum_x, g->d_sum_y, g->d_sum_z,
        g->d_sum_sq_x, g->d_sum_sq_y, g->d_sum_sq_z,
        g->d_sum_r, g->d_sum_g, g->d_sum_b,
        g->d_weight, g->d_count,
        min_weight, min_count, max_spread_frac,
        d_out_xyz, d_out_rgb, d_out_cnt, buffer_cap
    );
    cudaStreamSynchronize(g->stream);

    unsigned int final_count = 0;
    cudaMemcpyAsync(&final_count, d_out_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost, g->stream);
    cudaStreamSynchronize(g->stream);
    if ((int)final_count > buffer_cap) final_count = (unsigned int)buffer_cap;

    cudaMemcpyAsync(buffer_xyz, d_out_xyz,
               (size_t)final_count * 3 * sizeof(float), cudaMemcpyDeviceToHost, g->stream);
    cudaMemcpyAsync(buffer_rgb, d_out_rgb,
               (size_t)final_count * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost, g->stream);
    cudaStreamSynchronize(g->stream);

    cudaFree(d_out_xyz);
    cudaFree(d_out_rgb);
    cudaFree(d_out_cnt);

    if (drain_committed && committed_count > 0 && (int)committed_count <= buffer_cap) {
        // Drain: zero the committed-count pointer so next slide starts fresh.
        unsigned int zero = 0;
        cudaMemcpyAsync(g->d_committed_count_ptr, &zero, sizeof(unsigned int),
                   cudaMemcpyHostToDevice, g->stream);
        cudaStreamSynchronize(g->stream);
    }

    return (int)final_count;
}

extern "C" int tsdf_grid_active_voxel_count(TsdfGrid* g) {
    if (!g) return -1;
    // Exact count requires a reduction over count_buf; v0 returns -1 until
    // the dedicated kernel is wired. This is diagnostic-only.
    return -1;
}

extern "C" int tsdf_grid_committed_count(TsdfGrid* g) {
    if (!g) return -1;
    unsigned int c = 0;
    if (cudaMemcpyAsync(&c, g->d_committed_count_ptr, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost, g->stream) != cudaSuccess) {
        return -1;
    }
    if (cudaStreamSynchronize(g->stream) != cudaSuccess) {
        return -1;
    }
    return (int)c;
}

extern "C" int tsdf_grid_set_weight_cap(TsdfGrid* g, float cap) {
    if (!g) return -1;
    g->weight_cap = cap;
    return 0;
}
