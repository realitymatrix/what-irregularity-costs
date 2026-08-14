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
/// `COUNT_CAS` is a measurement switch, default off, that tallies every
/// compare-exchange ATTEMPT into `drop_count`. Static SASS says A3 and A4's
/// allocate kernels are at 1.02x instruction parity (520 against 528) while A4
/// runs 5.9x slower, so the difference has to be in how often instructions
/// execute rather than how many exist. Arm A4 carries the identical switch.
/// Safe to overload `drop_count` because the counted kernel is only ever
/// launched on a pool large enough that no block is dropped.
/// `DO_CAS = false` stops at the first empty slot and never inserts, mirroring
/// arm A4's `CAS=false` specialisation exactly. Both then leave the table empty,
/// so every probe resolves at slot 0 and what remains is the geometry walk plus
/// one hash and one load. That is the only way to compare the two arms' baseline
/// per-thread cost, which is otherwise hidden behind the insert machinery.
template <bool COUNT_CAS = false, bool DO_CAS = true>
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
            if (!DO_CAS) return -1;
            // One CAS publishes the entire coordinate. Splitting it across
            // words lets a reader match a partially written slot, mismatch on
            // the rest, and probe onward; at GPU thread counts that becomes a
            // full-table scan per lookup.
            if (COUNT_CAS) atomicAdd(v.drop_count, 1ULL);
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

/// MEASUREMENT ONLY: the same probe, reporting how deep it had to go.
///
/// The number this collects is what separates a SIMT model from a tile model.
/// A thread that resolves on its first probe stops there and costs one step; a
/// tile program cannot stop until every lane in it has resolved, so it costs
/// the deepest lane's count for all of them. Summing the first and taking the
/// per-program maximum of the second prices both models on the same run.
__device__ int32_t d_find_or_insert_depth(const DeviceView& v, int32_t bx, int32_t by, int32_t bz,
                                          uint32_t* steps) {
    const uint32_t size = v.hash_mask + 1;
    const uint32_t slot = d_hash(bx, by, bz, v.hash_mask);
    const int64_t want = d_pack(bx, by, bz);
    for (uint32_t probe = 0; probe < size; ++probe) {
        ++(*steps);
        HashEntry& e = v.table[(slot + probe) & v.hash_mask];
        int64_t key = e.key;
        if (key == want) {
            int32_t idx = e.block_idx;
            while (idx < 0) idx = *(volatile int32_t*)&e.block_idx;
            return idx;
        }
        if (key == kEmptyKey) {
            const int64_t prev = (int64_t)atomicCAS((unsigned long long*)&e.key,
                                                    (unsigned long long)kEmptyKey,
                                                    (unsigned long long)want);
            if (prev == kEmptyKey) {
                const int32_t idx = atomicAdd(v.block_count, 1);
                if (idx >= v.pool_capacity) {
                    atomicSub(v.block_count, 1);
                    atomicExch((unsigned long long*)&e.key, (unsigned long long)kEmptyKey);
                    return -1;
                }
                v.block_coord[idx] = BlockCoord{bx, by, bz};
                __threadfence();
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

/// Find or insert with the block index DERIVED FROM THE SLOT.
///
/// A hash entry holds two logically different things: the block's identity, the
/// packed coordinate, known before insertion; and its location in the voxel
/// pool, handed out by a counter and known only after winning the slot.
/// Publishing both atomically needs the location first, but the location is
/// only yours if the exchange succeeds. That circularity is why the production
/// path publishes the key and then the index, which leaves a window in which a
/// reader sees a key whose index is still -1 and must spin.
///
/// Setting `block_idx = slot` removes the circularity: the location is implied
/// by the identity's position, so there is one value to publish, and a reader
/// that matches a key knows the index immediately. No publication store, no
/// fence, no wait.
///
/// The price is that every table slot needs voxel storage. This mirrors arm
/// A4's variant exactly, including how that is arranged without reallocating:
/// the hash is masked to `pool_capacity` rather than to `hash_mask`, so the
/// effective table is the pool and half the allocated table goes unused. A real
/// implementation would size the pool to the table and pay ~2x voxel memory.
///
/// The counter is still bumped once per successful insert so the host can check
/// the block count. Nothing reads it and no thread waits on it.
__device__ int32_t d_find_or_insert_slotidx(const DeviceView& v, int32_t bx, int32_t by,
                                            int32_t bz) {
    const uint32_t mask = (uint32_t)v.pool_capacity - 1u;
    const int64_t want = d_pack(bx, by, bz);
    const uint32_t start = d_hash(bx, by, bz, mask);
    for (uint32_t probe = 0; probe <= mask; ++probe) {
        const uint32_t slot = (start + probe) & mask;
        HashEntry& e = v.table[slot];
        const int64_t key = e.key;
        // A matching key means the index IS this slot. Nothing to wait for.
        if (key == want) return (int32_t)slot;
        if (key == kEmptyKey) {
            const int64_t prev = (int64_t)atomicCAS((unsigned long long*)&e.key,
                                                    (unsigned long long)kEmptyKey,
                                                    (unsigned long long)want);
            if (prev == kEmptyKey) {
                v.block_coord[slot] = BlockCoord{bx, by, bz};
                atomicAdd(v.block_count, 1);
                return (int32_t)slot;
            }
            // Lost the race, but if the winner wanted the same block the slot
            // is still the answer.
            if (prev == want) return (int32_t)slot;
        }
    }
    atomicAdd(v.drop_count, 1ULL);
    return -1;
}

/// Pass 1 with the block index derived from the hash slot. See
/// `d_find_or_insert_slotidx`.
__global__ void alloc_kernel_slotidx(DeviceView v, const float* pos, int32_t n, float cam0,
                                     float cam1, float cam2, float radius_sq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cam[3] = {cam0, cam1, cam2};
    walk_band(v, pos, i, cam, radius_sq, [&](int32_t vx, int32_t vy, int32_t vz, float) {
        d_find_or_insert_slotidx(v, d_floor_div(vx, kBlockDim), d_floor_div(vy, kBlockDim),
                                 d_floor_div(vz, kBlockDim));
    });
}

template <bool COUNT_CAS = false, bool DO_CAS = true>
__global__ void alloc_kernel(DeviceView v, const float* pos, int32_t n, float cam0, float cam1,
                             float cam2, float radius_sq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cam[3] = {cam0, cam1, cam2};
    walk_band(v, pos, i, cam, radius_sq, [&](int32_t vx, int32_t vy, int32_t vz, float) {
        d_find_or_insert<COUNT_CAS, DO_CAS>(v, d_floor_div(vx, kBlockDim), d_floor_div(vy, kBlockDim),
                         d_floor_div(vz, kBlockDim));
    });
}

/// MEASUREMENT ONLY. Emits two totals across the whole launch:
///   lane_steps  sum over threads of the probe steps that thread needed
///   tile_steps  sum over programs of (threads per program x deepest lane)
/// The first is what a per-thread model pays. The second is the floor for any
/// model that cannot retire a lane early, which is the question a tile language
/// is being asked here.
__global__ void alloc_kernel_probe_depth(DeviceView v, const float* pos, int32_t n, float cam0,
                                         float cam1, float cam2, float radius_sq,
                                         unsigned long long* lane_steps,
                                         unsigned long long* tile_steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t mine = 0;
    if (i < n) {
        const float cam[3] = {cam0, cam1, cam2};
        walk_band(v, pos, i, cam, radius_sq, [&](int32_t vx, int32_t vy, int32_t vz, float) {
            d_find_or_insert_depth(v, d_floor_div(vx, kBlockDim), d_floor_div(vy, kBlockDim),
                                   d_floor_div(vz, kBlockDim), &mine);
        });
    }
    __shared__ unsigned int s_max, s_sum;
    if (threadIdx.x == 0) { s_max = 0u; s_sum = 0u; }
    __syncthreads();
    atomicMax(&s_max, mine);
    atomicAdd(&s_sum, mine);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(lane_steps, (unsigned long long)s_sum);
        atomicAdd(tile_steps, (unsigned long long)s_max * (unsigned long long)blockDim.x);
    }
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
    // Live table only. The scratch region past `size` is write-once (its
    // sentinel key can never satisfy a CAS against kEmptyKey), so rewriting it
    // every reset would copy 16 MiB per volume per repetition for no effect.
    std::vector<HashEntry> host(size, HashEntry{kEmptyKey, -1, 0});
    cudaMemcpyAsync(impl_->v.table, host.data(), (size_t)size * sizeof(HashEntry),
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
    alloc_kernel<false><<<blocks, threads, 0, impl_->stream>>>(impl_->v, b.positions, b.n, b.cam[0],
                                                        b.cam[1], b.cam[2], rsq);
}

/// MEASUREMENT ONLY: same kernel with every compare-exchange attempt tallied
/// into `drop_count`. See the note on `d_find_or_insert`.
void CudaVolume::allocate_blocks_counting_cas(const PointBatch& b) {
    if (!valid() || b.n <= 0) return;
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    alloc_kernel<true><<<blocks, threads, 0, impl_->stream>>>(impl_->v, b.positions, b.n, b.cam[0],
                                                              b.cam[1], b.cam[2], rsq);
}

/// MEASUREMENT ONLY: price the same run under a per-thread and a tile model.
void CudaVolume::allocate_blocks_probe_depth(const PointBatch& b, uint64_t* lane_steps,
                                             uint64_t* tile_steps) {
    *lane_steps = 0; *tile_steps = 0;
    if (!valid() || b.n <= 0) return;
    unsigned long long *d_lane = nullptr, *d_tile = nullptr;
    cudaMalloc(&d_lane, sizeof(unsigned long long));
    cudaMalloc(&d_tile, sizeof(unsigned long long));
    cudaMemset(d_lane, 0, sizeof(unsigned long long));
    cudaMemset(d_tile, 0, sizeof(unsigned long long));
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    alloc_kernel_probe_depth<<<blocks, threads, 0, impl_->stream>>>(
        impl_->v, b.positions, b.n, b.cam[0], b.cam[1], b.cam[2], rsq, d_lane, d_tile);
    cudaStreamSynchronize(impl_->stream);
    unsigned long long hl = 0, ht = 0;
    cudaMemcpy(&hl, d_lane, sizeof(hl), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ht, d_tile, sizeof(ht), cudaMemcpyDeviceToHost);
    *lane_steps = hl; *tile_steps = ht;
    cudaFree(d_lane); cudaFree(d_tile);
}

/// MEASUREMENT ONLY: probe and read, never insert. Mirrors arm A4's `-cas`
/// variant so the two arms' baseline per-thread cost can be compared directly.
void CudaVolume::allocate_blocks_no_cas(const PointBatch& b) {
    if (!valid() || b.n <= 0) return;
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    alloc_kernel<false, false><<<blocks, threads, 0, impl_->stream>>>(
        impl_->v, b.positions, b.n, b.cam[0], b.cam[1], b.cam[2], rsq);
}

/// MEASUREMENT ONLY: allocate with the block index derived from the hash slot.
void CudaVolume::allocate_blocks_slot_index(const PointBatch& b) {
    if (!valid() || b.n <= 0) return;
    const int threads = 256;
    const int blocks = (b.n + threads - 1) / threads;
    const float rsq = b.radius_m > 0.0f ? b.radius_m * b.radius_m : 0.0f;
    alloc_kernel_slotidx<<<blocks, threads, 0, impl_->stream>>>(
        impl_->v, b.positions, b.n, b.cam[0], b.cam[1], b.cam[2], rsq);
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
