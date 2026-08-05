#include <cmath>

#include "cuda_impl.cuh"

namespace osn_tsdf {
namespace {

/// Device mirror of the host hash. Must stay bit-identical to `hash_block`:
/// a different mix changes probe sequences, which changes which blocks share
/// cache lines, and that would surface as a "language" difference.
__device__ __forceinline__ uint32_t d_hash(int32_t x, int32_t y, int32_t z, uint32_t mask) {
    const uint32_t h = static_cast<uint32_t>(x) * 73856093u ^
                       static_cast<uint32_t>(y) * 19349663u ^
                       static_cast<uint32_t>(z) * 83492791u;
    return h & mask;
}

__device__ __forceinline__ int32_t d_floor_div(int32_t a, int32_t b) {
    const int32_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

__device__ __forceinline__ int64_t d_pack(int32_t x, int32_t y, int32_t z) {
    return (((int64_t)x + kCoordBias) << (2 * kCoordBits)) |
           (((int64_t)y + kCoordBias) << kCoordBits) | ((int64_t)z + kCoordBias);
}

__device__ int32_t d_find_block(const DeviceView& v, int32_t bx, int32_t by, int32_t bz) {
    const uint32_t size = v.hash_mask + 1;
    const uint32_t slot = d_hash(bx, by, bz, v.hash_mask);
    const int64_t want = d_pack(bx, by, bz);
    for (uint32_t probe = 0; probe < size; ++probe) {
        const HashEntry& e = v.table[(slot + probe) & v.hash_mask];
        const int64_t key = e.key;
        if (key == kEmptyKey) return -1;
        if (key == want) {
            int32_t idx = e.block_idx;
            while (idx < 0) idx = *(volatile int32_t*)&e.block_idx;
            return idx;
        }
    }
    return -1;
}

/// Find or insert, mirroring the CPU protocol: claim the slot with a
/// compare-exchange on the key alone, then publish the block index.
__device__ int32_t d_find_or_insert(const DeviceView& v, int32_t bx, int32_t by, int32_t bz) {
    const uint32_t size = v.hash_mask + 1;
    const uint32_t slot = d_hash(bx, by, bz, v.hash_mask);
    const int64_t want = d_pack(bx, by, bz);
    for (uint32_t probe = 0; probe < size; ++probe) {
        HashEntry& e = v.table[(slot + probe) & v.hash_mask];
        int64_t key = e.key;
        if (key == want) {
            // Spin until the winner publishes the index. Without this a peer
            // that saw our key but not yet its block_idx would read -1 and
            // silently drop the point.
            int32_t idx = e.block_idx;
            while (idx < 0) idx = *(volatile int32_t*)&e.block_idx;
            return idx;
        }
        if (key == kEmptyKey) {
            // One CAS publishes the entire coordinate. Splitting it across
            // words lets a reader match a partially written slot, mismatch on
            // the rest, and probe onward; at GPU thread counts that becomes a
            // full-table scan per lookup.
            const int64_t prev = (int64_t)atomicCAS((unsigned long long*)&e.key,
                                                    (unsigned long long)kEmptyKey,
                                                    (unsigned long long)want);
            if (prev == kEmptyKey) {
                const int32_t idx = atomicAdd(v.block_count, 1);
                if (idx >= v.pool_capacity) {
                    atomicSub(v.block_count, 1);
                    atomicExch((unsigned long long*)&e.key, (unsigned long long)kEmptyKey);
                    atomicAdd(v.drop_count, 1ULL);
                    return -1;
                }
                v.block_coord[idx] = BlockCoord{bx, by, bz};
                __threadfence();  // publish coords before the index
                atomicExch(&e.block_idx, idx);
                return idx;
            }
            if (prev == want) {
                int32_t idx = e.block_idx;
                while (idx < 0) idx = *(volatile int32_t*)&e.block_idx;
                return idx;
            }
        }
    }
    atomicAdd(v.drop_count, 1ULL);
    return -1;
}

/// Shared geometry: walk the truncation band along the view ray.
///
/// Returns false when the point is gated out. `body(vx, vy, vz, sdf)` receives
/// each voxel in the band and the signed distance AT ITS CENTRE.
template <typename F>
__device__ __forceinline__ void walk_band(const DeviceView& v, const float* pos, int i,
                                          const float cam[3], float radius_sq, F&& body) {
    const float px = pos[i * 3 + 0], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    const float dx = px - cam[0], dy = py - cam[1], dz = pz - cam[2];
    const float d2 = dx * dx + dy * dy + dz * dz;
    if (radius_sq > 0.0f && d2 > radius_sq) return;
    const float dist = sqrtf(d2);
    if (!(dist > 1e-6f)) return;
    const float ux = dx / dist, uy = dy / dist, uz = dz / dist;

    const float inv_voxel = 1.0f / v.voxel_size;
    const int32_t steps = max(1, (int32_t)ceilf(v.trunc * inv_voxel));
    for (int32_t s = -steps; s <= steps; ++s) {
        const float t = (float)s * v.voxel_size;
        const int32_t vx = (int32_t)floorf((px + ux * t) * inv_voxel);
        const int32_t vy = (int32_t)floorf((py + uy * t) * inv_voxel);
        const int32_t vz = (int32_t)floorf((pz + uz * t) * inv_voxel);

        // Signed distance at the VOXEL CENTRE, not at the ray sample that
        // selected it. Points are binned with floor(p / voxel), so the centre
        // is up to half a voxel away; using the sample's own offset biases
        // every voxel and shows up as a uniform radius error on curved
        // surfaces. See the CPU arm for the measured value.
        const float cx = ((float)vx + 0.5f) * v.voxel_size;
        const float cy = ((float)vy + 0.5f) * v.voxel_size;
        const float cz = ((float)vz + 0.5f) * v.voxel_size;
        const float ex = cx - cam[0], ey = cy - cam[1], ez = cz - cam[2];
        const float sdf = dist - sqrtf(ex * ex + ey * ey + ez * ez);
        if (sdf < -v.trunc) continue;
        body(vx, vy, vz, sdf);
    }
}

__global__ void alloc_kernel(DeviceView v, const float* pos, int32_t n, float cam0, float cam1,
                             float cam2, float radius_sq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cam[3] = {cam0, cam1, cam2};
    walk_band(v, pos, i, cam, radius_sq, [&](int32_t vx, int32_t vy, int32_t vz, float) {
        d_find_or_insert(v, d_floor_div(vx, kBlockDim), d_floor_div(vy, kBlockDim),
                         d_floor_div(vz, kBlockDim));
    });
}

__global__ void update_kernel(DeviceView v, const float* pos, const uint8_t* col, const float* wts,
                              int32_t n, float cam0, float cam1, float cam2, float radius_sq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cam[3] = {cam0, cam1, cam2};
    const float w_in = wts ? wts[i] : 1.0f;
    const float cr = col ? (float)col[i * 3 + 0] : 128.0f;
    const float cg = col ? (float)col[i * 3 + 1] : 128.0f;
    const float cb = col ? (float)col[i * 3 + 2] : 128.0f;
    const float inv_trunc = 1.0f / v.trunc;

    walk_band(v, pos, i, cam, radius_sq, [&](int32_t vx, int32_t vy, int32_t vz, float sdf) {
        const int32_t bx = d_floor_div(vx, kBlockDim);
        const int32_t by = d_floor_div(vy, kBlockDim);
        const int32_t bz = d_floor_div(vz, kBlockDim);
        const int32_t bi = d_find_block(v, bx, by, bz);
        if (bi < 0) return;

        const int32_t lx = vx - bx * kBlockDim;
        const int32_t ly = vy - by * kBlockDim;
        const int32_t lz = vz - bz * kBlockDim;
        const size_t idx = (size_t)bi * kBlockVoxels + (size_t)((lz * kBlockDim + ly) * kBlockDim + lx);

        if (v.weight_cap > 0.0f && v.weight[idx] >= v.weight_cap) return;

        const float sdf_n = fminf(fmaxf(sdf * inv_trunc, -1.0f), 1.0f);
        // Weighted SUMS, matching the CPU arm. Sums commute, so plain atomic
        // adds suffice; a running mean would need a read-modify-write that no
        // single atomic can make safe across four arrays.
        atomicAdd(&v.weight[idx], w_in);
        atomicAdd(&v.tsdf[idx], sdf_n * w_in);
        atomicAdd(&v.r[idx], cr * w_in);
        atomicAdd(&v.g[idx], cg * w_in);
        atomicAdd(&v.b[idx], cb * w_in);
    });
}

}  // namespace

// ---------------------------------------------------------------------------

CudaVolume::CudaVolume(const VolumeConfig& cfg) {
    impl_ = new CudaVolumeImpl();
    impl_->cfg = cfg;
    if (!impl_->alloc()) {
        // Leave impl_ allocated but invalid so valid() can report, rather than
        // throwing across what is ultimately a C boundary.
        impl_->ok = false;
    }
}

CudaVolume::~CudaVolume() { delete impl_; }

bool CudaVolume::valid() const { return impl_ && impl_->ok; }
const VolumeConfig& CudaVolume::config() const { return impl_->cfg; }

void CudaVolume::synchronize() const { cudaStreamSynchronize(impl_->stream); }

const DeviceView& CudaVolume::device_view() const { return impl_->v; }

void CudaVolume::reset() {
    if (!valid()) return;
    const uint32_t size = impl_->v.hash_mask + 1;
    const size_t n_vox = (size_t)impl_->cfg.pool_capacity_blocks * kBlockVoxels;
    // The table must be refilled with kEmptyKey, not zeroed: zero is a valid
    // packed coordinate, so a zeroed table reads as occupied.
    std::vector<HashEntry> host(size + kScratchSlots, HashEntry{kEmptyKey, -1, 0});
    for (uint32_t i = size; i < size + kScratchSlots; ++i) host[i].key = -2;
    cudaMemcpyAsync(impl_->v.table, host.data(),
                    (size_t)(size + kScratchSlots) * sizeof(HashEntry),
                    cudaMemcpyHostToDevice, impl_->stream);
    cudaMemsetAsync(impl_->v.block_count, 0, sizeof(int32_t), impl_->stream);
    cudaMemsetAsync(impl_->v.drop_count, 0, sizeof(unsigned long long), impl_->stream);
    cudaMemsetAsync(impl_->v.tsdf, 0, n_vox * sizeof(float), impl_->stream);
    cudaMemsetAsync(impl_->v.weight, 0, n_vox * sizeof(float), impl_->stream);
    cudaMemsetAsync(impl_->v.r, 0, n_vox * sizeof(float), impl_->stream);
    cudaMemsetAsync(impl_->v.g, 0, n_vox * sizeof(float), impl_->stream);
    cudaMemsetAsync(impl_->v.b, 0, n_vox * sizeof(float), impl_->stream);
    cudaStreamSynchronize(impl_->stream);
}

void CudaVolume::allocate_blocks(const PointBatch& b) {
    if (!valid() || b.n <= 0) return;
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    alloc_kernel<<<blocks, threads, 0, impl_->stream>>>(impl_->v, b.positions, b.n, b.cam[0],
                                                        b.cam[1], b.cam[2], rsq);
}

void CudaVolume::update_voxels(const PointBatch& b) {
    if (!valid() || b.n <= 0) return;
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    update_kernel<<<blocks, threads, 0, impl_->stream>>>(impl_->v, b.positions, b.colors, b.weights,
                                                         b.n, b.cam[0], b.cam[1], b.cam[2], rsq);
}

void CudaVolume::integrate(const PointBatch& b) {
    allocate_blocks(b);
    // Allocation must complete before the update looks blocks up. Same stream
    // is enough: work on one stream is ordered.
    update_voxels(b);
}

int32_t CudaVolume::block_count() const {
    if (!valid()) return 0;
    int32_t n = 0;
    cudaStreamSynchronize(impl_->stream);
    cudaMemcpy(&n, impl_->v.block_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return n;
}

uint64_t CudaVolume::drop_count() const {
    if (!valid()) return 0;
    unsigned long long n = 0;
    cudaStreamSynchronize(impl_->stream);
    cudaMemcpy(&n, impl_->v.drop_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return n;
}

}  // namespace osn_tsdf
