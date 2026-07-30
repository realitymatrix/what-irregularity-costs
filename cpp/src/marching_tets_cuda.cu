// Marching tetrahedra on device (arm A3). Mirrors the CPU extractor case for
// case; see src/marching_tets.cpp for why tetrahedra rather than cubes.
//
// One CUDA block per allocated pool block, one thread per voxel. That makes
// kBlockDim^3 = 512 the natural launch shape, which is also why kBlockDim is 8:
// 512 threads is a legal block size on every architecture this targets.

#include <cmath>
#include <vector>

#include "cuda_impl.cuh"

namespace osn_tsdf {
namespace {

__constant__ int c_tets[6][4] = {
    {0, 5, 1, 7}, {0, 1, 3, 7}, {0, 3, 2, 7},
    {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 2, 6, 7},
};
__constant__ int c_corner[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
};

__device__ __forceinline__ uint32_t dh(int32_t x, int32_t y, int32_t z, uint32_t mask) {
    return ((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u) & mask;
}
__device__ __forceinline__ int32_t dfd(int32_t a, int32_t b) {
    const int32_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

__device__ bool d_sample(const DeviceView& v, int32_t vx, int32_t vy, int32_t vz, float min_w,
                         float& out_t, float out_c[3]) {
    const int32_t bx = dfd(vx, kBlockDim), by = dfd(vy, kBlockDim), bz = dfd(vz, kBlockDim);
    const uint32_t size = v.hash_mask + 1;
    const uint32_t slot = dh(bx, by, bz, v.hash_mask);
    const int64_t want = (((int64_t)bx + kCoordBias) << (2 * kCoordBits)) |
                         (((int64_t)by + kCoordBias) << kCoordBits) | ((int64_t)bz + kCoordBias);
    int32_t bi = -1;
    for (uint32_t p = 0; p < size; ++p) {
        const HashEntry& e = v.table[(slot + p) & v.hash_mask];
        if (e.key == kEmptyKey) return false;
        if (e.key == want) { bi = e.block_idx; break; }
    }
    if (bi < 0) return false;

    const int32_t lx = vx - bx * kBlockDim, ly = vy - by * kBlockDim, lz = vz - bz * kBlockDim;
    const size_t idx = (size_t)bi * kBlockVoxels + (size_t)((lz * kBlockDim + ly) * kBlockDim + lx);
    const float w = v.weight[idx];
    if (w < min_w || w <= 0.0f) return false;
    const float inv = 1.0f / w;   // stored as weighted sums, mean formed here
    out_t = v.tsdf[idx] * inv;
    out_c[0] = v.r[idx] * inv; out_c[1] = v.g[idx] * inv; out_c[2] = v.b[idx] * inv;
    return true;
}

/// Position-ordered interpolation, so an edge shared by two tetrahedra yields
/// bitwise identical vertices from both sides and the mesh stays watertight.
__device__ void d_interp(float iso, const float* pa, const float* pb, float va, float vb,
                         const float* ca, const float* cb, float* op, float* oc) {
    const bool sw = (pb[0] < pa[0]) || (pb[0] == pa[0] && pb[1] < pa[1]) ||
                    (pb[0] == pa[0] && pb[1] == pa[1] && pb[2] < pa[2]);
    const float* p0 = sw ? pb : pa; const float* p1 = sw ? pa : pb;
    const float* c0 = sw ? cb : ca; const float* c1 = sw ? ca : cb;
    const float v0 = sw ? vb : va;  const float v1 = sw ? va : vb;
    const float den = v1 - v0;
    float t = fabsf(den) < 1e-12f ? 0.5f : (iso - v0) / den;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    for (int k = 0; k < 3; ++k) { op[k] = p0[k] + t * (p1[k] - p0[k]); oc[k] = c0[k] + t * (c1[k] - c0[k]); }
}

__global__ void mc_kernel(DeviceView v, int32_t n_blocks, float min_w, float iso, float* out_posnor,
                          uint8_t* out_rgb, int32_t cap, int32_t* n_written, int32_t* overflow) {
    const int32_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const int tid = threadIdx.x;
    if (tid >= kBlockVoxels) return;

    const BlockCoord bc = v.block_coord[bi];
    const int lx = tid % kBlockDim;
    const int ly = (tid / kBlockDim) % kBlockDim;
    const int lz = tid / (kBlockDim * kBlockDim);
    const int32_t vx = bc.x * kBlockDim + lx;
    const int32_t vy = bc.y * kBlockDim + ly;
    const int32_t vz = bc.z * kBlockDim + lz;

    float val[8], pos[8][3], col[8][3];
    for (int c = 0; c < 8; ++c) {
        float rgb[3];
        if (!d_sample(v, vx + c_corner[c][0], vy + c_corner[c][1], vz + c_corner[c][2], min_w,
                      val[c], rgb))
            return;  // cube straddles unobserved space
        // Voxel CENTRES: voxel v spans [v*vs,(v+1)*vs) so its sample is at (v+0.5)*vs.
        pos[c][0] = ((float)(vx + c_corner[c][0]) + 0.5f) * v.voxel_size;
        pos[c][1] = ((float)(vy + c_corner[c][1]) + 0.5f) * v.voxel_size;
        pos[c][2] = ((float)(vz + c_corner[c][2]) + 0.5f) * v.voxel_size;
        col[c][0] = rgb[0]; col[c][1] = rgb[1]; col[c][2] = rgb[2];
    }

    float tp[6][3], tc[6][3];
    for (int t = 0; t < 6; ++t) {
        const int i0 = c_tets[t][0], i1 = c_tets[t][1], i2 = c_tets[t][2], i3 = c_tets[t][3];
        const int code = (val[i0] < iso ? 1 : 0) | (val[i1] < iso ? 2 : 0) |
                         (val[i2] < iso ? 4 : 0) | (val[i3] < iso ? 8 : 0);
        int n_tri = 0;
        int a = -1, bq = -1, cq = -1, dq = -1;
        bool quad = false;
        switch (code) {
            case 0x00: case 0x0F: continue;
            case 0x01: case 0x0E: a = i0; bq = i1; cq = i2; dq = i3; break;
            case 0x02: case 0x0D: a = i1; bq = i0; cq = i2; dq = i3; break;
            case 0x04: case 0x0B: a = i2; bq = i0; cq = i1; dq = i3; break;
            case 0x08: case 0x07: a = i3; bq = i0; cq = i1; dq = i2; break;
            case 0x03: a = i0; bq = i1; cq = i2; dq = i3; quad = true; break;
            case 0x0C: a = i2; bq = i3; cq = i0; dq = i1; quad = true; break;
            case 0x05: a = i0; bq = i2; cq = i1; dq = i3; quad = true; break;
            case 0x0A: a = i1; bq = i3; cq = i0; dq = i2; quad = true; break;
            case 0x06: a = i1; bq = i2; cq = i0; dq = i3; quad = true; break;
            case 0x09: a = i0; bq = i3; cq = i1; dq = i2; quad = true; break;
            default: continue;
        }
        if (!quad) {
            d_interp(iso, pos[a], pos[bq], val[a], val[bq], col[a], col[bq], tp[0], tc[0]);
            d_interp(iso, pos[a], pos[cq], val[a], val[cq], col[a], col[cq], tp[1], tc[1]);
            d_interp(iso, pos[a], pos[dq], val[a], val[dq], col[a], col[dq], tp[2], tc[2]);
            n_tri = 1;
        } else {
            float ac[3], ac_c[3], ad[3], ad_c[3], bd[3], bd_c[3], bc[3], bc_c[3];
            d_interp(iso, pos[a], pos[cq], val[a], val[cq], col[a], col[cq], ac, ac_c);
            d_interp(iso, pos[a], pos[dq], val[a], val[dq], col[a], col[dq], ad, ad_c);
            d_interp(iso, pos[bq], pos[dq], val[bq], val[dq], col[bq], col[dq], bd, bd_c);
            d_interp(iso, pos[bq], pos[cq], val[bq], val[cq], col[bq], col[cq], bc, bc_c);
            for (int k = 0; k < 3; ++k) {
                tp[0][k] = ac[k]; tp[1][k] = ad[k]; tp[2][k] = bd[k];
                tp[3][k] = ac[k]; tp[4][k] = bd[k]; tp[5][k] = bc[k];
                tc[0][k] = ac_c[k]; tc[1][k] = ad_c[k]; tc[2][k] = bd_c[k];
                tc[3][k] = ac_c[k]; tc[4][k] = bd_c[k]; tc[5][k] = bc_c[k];
            }
            n_tri = 2;
        }

        const int32_t nv = n_tri * 3;
        const int32_t base = atomicAdd(n_written, nv);
        if (base + nv > cap) { atomicExch(overflow, 1); continue; }
        for (int32_t k = 0; k < nv; ++k) {
            float* dst = out_posnor + (size_t)(base + k) * 6;
            dst[0] = tp[k][0]; dst[1] = tp[k][1]; dst[2] = tp[k][2];
            if (out_rgb) {
                uint8_t* c = out_rgb + (size_t)(base + k) * 3;
                c[0] = (uint8_t)fminf(fmaxf(tc[k][0], 0.0f), 255.0f);
                c[1] = (uint8_t)fminf(fmaxf(tc[k][1], 0.0f), 255.0f);
                c[2] = (uint8_t)fminf(fmaxf(tc[k][2], 0.0f), 255.0f);
            }
        }
        for (int32_t f = 0; f < n_tri; ++f) {
            float* v0 = out_posnor + (size_t)(base + f * 3) * 6;
            float* v1 = v0 + 6; float* v2 = v0 + 12;
            const float e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
            const float e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
            float nx = e1[1]*e2[2]-e1[2]*e2[1], ny = e1[2]*e2[0]-e1[0]*e2[2], nz = e1[0]*e2[1]-e1[1]*e2[0];
            const float len = sqrtf(nx*nx+ny*ny+nz*nz);
            if (len > 1e-20f) { nx/=len; ny/=len; nz/=len; }
            v0[3]=nx; v0[4]=ny; v0[5]=nz; v1[3]=nx; v1[4]=ny; v1[5]=nz; v2[3]=nx; v2[4]=ny; v2[5]=nz;
        }
    }
}

}  // namespace

int32_t CudaVolume::extract_mesh(const MeshBuffers& out, float min_weight, float iso) const {
    if (!valid() || out.capacity_vertices <= 0) return -1;
    const int32_t n_blocks = block_count();   // synchronises
    if (n_blocks <= 0) return 0;

    const size_t cap = (size_t)out.capacity_vertices;
    float* d_posnor = nullptr;
    uint8_t* d_rgb = nullptr;
    int32_t* d_count = nullptr;
    int32_t* d_overflow = nullptr;
    if (cudaMalloc(&d_posnor, cap * 6 * sizeof(float)) != cudaSuccess) return -1;
    if (out.rgb && cudaMalloc(&d_rgb, cap * 3) != cudaSuccess) { cudaFree(d_posnor); return -1; }
    cudaMalloc(&d_count, sizeof(int32_t));
    cudaMalloc(&d_overflow, sizeof(int32_t));
    cudaMemset(d_count, 0, sizeof(int32_t));
    cudaMemset(d_overflow, 0, sizeof(int32_t));

    DeviceView v = impl_->v;
    // One CUDA block per pool block, one thread per voxel.
    mc_kernel<<<n_blocks, kBlockVoxels, 0, impl_->stream>>>(v, n_blocks, min_weight, iso, d_posnor,
                                                            d_rgb, out.capacity_vertices, d_count,
                                                            d_overflow);
    cudaStreamSynchronize(impl_->stream);

    int32_t n = 0, ovf = 0;
    cudaMemcpy(&n, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ovf, d_overflow, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (!ovf && n > 0) {
        cudaMemcpy(out.posnor, d_posnor, (size_t)n * 6 * sizeof(float), cudaMemcpyDeviceToHost);
        if (out.rgb && d_rgb) cudaMemcpy(out.rgb, d_rgb, (size_t)n * 3, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_posnor);
    if (d_rgb) cudaFree(d_rgb);
    cudaFree(d_count);
    cudaFree(d_overflow);
    return ovf ? -1 : n;
}

}  // namespace osn_tsdf
