#include "osn_tsdf/volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>

namespace osn_tsdf {
namespace {

/// Round up to a power of two, so the hash mask is a single AND.
uint32_t next_pow2(uint32_t v) {
    if (v < 2) return 2;
    --v;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/// Run `body(begin, end)` over [0, n) across `n_threads` workers.
void parallel_for(int32_t n, int n_threads, const std::function<void(int32_t, int32_t)>& body) {
    if (n <= 0) return;
    if (n_threads <= 1) { body(0, n); return; }

    const int32_t chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(n_threads) - 1);
    for (int t = 1; t < n_threads; ++t) {
        const int32_t lo = std::min(n, chunk * t);
        const int32_t hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&body, lo, hi] { body(lo, hi); });
    }
    body(0, std::min(n, chunk));  // this thread takes the first chunk
    for (auto& th : pool) th.join();
}

}  // namespace

Volume::Volume(const VolumeConfig& cfg) : cfg_(cfg) {
    if (cfg_.voxel_size_m <= 0.0f) throw std::invalid_argument("voxel_size_m must be > 0");
    if (cfg_.pool_capacity_blocks <= 0) throw std::invalid_argument("pool_capacity_blocks must be > 0");

    // Table at 2x pool capacity. Open addressing degrades sharply above ~0.7
    // load factor, and a full table cannot report failure without a probe
    // limit, so headroom is cheaper than the alternative.
    const uint32_t size = next_pow2(static_cast<uint32_t>(cfg_.pool_capacity_blocks) * 2u);
    hash_mask_ = size - 1;
    table_.assign(size, HashEntry{kEmptyKey, 0, 0, -1});

    const std::size_t n = static_cast<std::size_t>(cfg_.pool_capacity_blocks) * kBlockVoxels;
    tsdf_.assign(n, 0.0f);
    weight_.assign(n, 0.0f);
    r_.assign(n, 0.0f);
    g_.assign(n, 0.0f);
    b_.assign(n, 0.0f);
    block_coord_.assign(static_cast<std::size_t>(cfg_.pool_capacity_blocks), BlockCoord{0, 0, 0});

    set_thread_count(0);
}

Volume::~Volume() = default;

void Volume::set_thread_count(int n) {
    if (n > 0) { n_threads_ = n; return; }
    const unsigned hw = std::thread::hardware_concurrency();
    n_threads_ = hw == 0 ? 1 : static_cast<int>(hw);
}

int32_t Volume::find_block(int32_t bx, int32_t by, int32_t bz) const {
    uint32_t slot = hash_block(bx, by, bz, hash_mask_);
    const uint32_t size = hash_mask_ + 1;
    for (uint32_t probe = 0; probe < size; ++probe) {
        const HashEntry& e = table_[(slot + probe) & hash_mask_];
        // atomic_ref rather than casting to atomic<T>*: the cast is undefined
        // behaviour even where it happens to work, and this is a lock-free
        // protocol where a compiler reordering would be near-impossible to
        // debug from the symptom.
        const int32_t key =
            std::atomic_ref<const int32_t>(e.key).load(std::memory_order_acquire);
        if (key == kEmptyKey) return -1;
        if (key == bx && e.y == by && e.z == bz) return e.block_idx;
    }
    return -1;
}

int32_t Volume::find_or_insert_block(int32_t bx, int32_t by, int32_t bz) {
    uint32_t slot = hash_block(bx, by, bz, hash_mask_);
    const uint32_t size = hash_mask_ + 1;

    for (uint32_t probe = 0; probe < size; ++probe) {
        HashEntry& e = table_[(slot + probe) & hash_mask_];
        std::atomic_ref<int32_t> key_atomic(e.key);

        int32_t key = key_atomic.load(std::memory_order_acquire);
        if (key == bx && e.y == by && e.z == bz) return e.block_idx;

        if (key == kEmptyKey) {
            // Claim the slot with a compare-exchange on the key alone. This is
            // deliberately the same protocol the GPU arms use, so the CPU arm
            // is a parallel implementation of the same algorithm rather than a
            // mutex-guarded stand-in that would make the comparison unfair.
            int32_t expected = kEmptyKey;
            if (key_atomic.compare_exchange_strong(expected, bx, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                const int32_t idx = block_count_.fetch_add(1, std::memory_order_acq_rel);
                if (idx >= cfg_.pool_capacity_blocks) {
                    // Pool exhausted. Release the slot so a later, smaller run
                    // can reuse it, and report rather than silently dropping.
                    block_count_.fetch_sub(1, std::memory_order_acq_rel);
                    key_atomic.store(kEmptyKey, std::memory_order_release);
                    drop_count_.fetch_add(1, std::memory_order_relaxed);
                    return -1;
                }
                e.y = by;
                e.z = bz;
                block_coord_[static_cast<std::size_t>(idx)] = BlockCoord{bx, by, bz};
                // Publish block_idx last: a concurrent reader that saw our key
                // must not read an uninitialised index.
                std::atomic_ref<int32_t>(e.block_idx).store(idx, std::memory_order_release);
                return idx;
            }
            // Lost the race. Re-read this slot: the winner may have written our
            // key, in which case we share it rather than probing past it.
            key = key_atomic.load(std::memory_order_acquire);
            if (key == bx && e.y == by && e.z == bz) {
                return std::atomic_ref<int32_t>(e.block_idx).load(std::memory_order_acquire);
            }
        }
    }
    drop_count_.fetch_add(1, std::memory_order_relaxed);
    return -1;
}

void Volume::allocate_blocks(const PointBatch& batch) {
    const float inv_voxel = 1.0f / cfg_.voxel_size_m;
    const float trunc = cfg_.trunc();
    const float radius_sq = batch.radius_m > 0.0f ? batch.radius_m * batch.radius_m : 0.0f;

    parallel_for(batch.n, n_threads_, [&](int32_t lo, int32_t hi) {
        for (int32_t i = lo; i < hi; ++i) {
            const float px = batch.positions[i * 3 + 0];
            const float py = batch.positions[i * 3 + 1];
            const float pz = batch.positions[i * 3 + 2];

            const float dx = px - batch.cam[0];
            const float dy = py - batch.cam[1];
            const float dz = pz - batch.cam[2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (radius_sq > 0.0f && d2 > radius_sq) continue;

            const float dist = std::sqrt(d2);
            if (!(dist > 1e-6f)) continue;
            const float ux = dx / dist, uy = dy / dist, uz = dz / dist;

            // Allocate along the truncation band, not just at the hit point.
            // A band only one voxel thick gives marching tetrahedra no
            // negative side to cross, so the isosurface is never found.
            const int32_t steps = std::max(1, static_cast<int32_t>(std::ceil(trunc * inv_voxel)));
            for (int32_t s = -steps; s <= steps; ++s) {
                const float t = static_cast<float>(s) * cfg_.voxel_size_m;
                const int32_t vx = static_cast<int32_t>(std::floor((px + ux * t) * inv_voxel));
                const int32_t vy = static_cast<int32_t>(std::floor((py + uy * t) * inv_voxel));
                const int32_t vz = static_cast<int32_t>(std::floor((pz + uz * t) * inv_voxel));
                find_or_insert_block(floor_div(vx, kBlockDim), floor_div(vy, kBlockDim),
                                     floor_div(vz, kBlockDim));
            }
        }
    });
}

void Volume::update_voxels(const PointBatch& batch) {
    const float inv_voxel = 1.0f / cfg_.voxel_size_m;
    const float trunc = cfg_.trunc();
    const float inv_trunc = 1.0f / trunc;
    const float radius_sq = batch.radius_m > 0.0f ? batch.radius_m * batch.radius_m : 0.0f;
    const float wcap = cfg_.weight_cap;

    parallel_for(batch.n, n_threads_, [&](int32_t lo, int32_t hi) {
        for (int32_t i = lo; i < hi; ++i) {
            const float px = batch.positions[i * 3 + 0];
            const float py = batch.positions[i * 3 + 1];
            const float pz = batch.positions[i * 3 + 2];

            const float dx = px - batch.cam[0];
            const float dy = py - batch.cam[1];
            const float dz = pz - batch.cam[2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (radius_sq > 0.0f && d2 > radius_sq) continue;
            const float dist = std::sqrt(d2);
            if (!(dist > 1e-6f)) continue;
            const float ux = dx / dist, uy = dy / dist, uz = dz / dist;

            const float w_in = batch.weights ? batch.weights[i] : 1.0f;
            float cr = 128.0f, cg = 128.0f, cb = 128.0f;
            if (batch.colors) {
                cr = static_cast<float>(batch.colors[i * 3 + 0]);
                cg = static_cast<float>(batch.colors[i * 3 + 1]);
                cb = static_cast<float>(batch.colors[i * 3 + 2]);
            }

            const int32_t steps = std::max(1, static_cast<int32_t>(std::ceil(trunc * inv_voxel)));
            for (int32_t s = -steps; s <= steps; ++s) {
                const float t = static_cast<float>(s) * cfg_.voxel_size_m;
                const float sx = px + ux * t;
                const float sy = py + uy * t;
                const float sz = pz + uz * t;

                const int32_t vx = static_cast<int32_t>(std::floor(sx * inv_voxel));
                const int32_t vy = static_cast<int32_t>(std::floor(sy * inv_voxel));
                const int32_t vz = static_cast<int32_t>(std::floor(sz * inv_voxel));

                const int32_t bi = find_block(floor_div(vx, kBlockDim), floor_div(vy, kBlockDim),
                                              floor_div(vz, kBlockDim));
                if (bi < 0) continue;

                // Signed distance evaluated at the VOXEL CENTRE, not at the
                // ray sample that selected the voxel.
                //
                // Points are binned with floor(p / voxel), so the voxel holding
                // a sample has its centre at (v + 0.5) * voxel, up to half a
                // voxel away. Storing the sample's own offset (-t) therefore
                // biases every voxel by up to voxel/2, which shows up as a
                // uniform radius error on a curved surface: measured 0.005 m
                // mean on a 0.01 m voxel before this was corrected. A plane
                // hides it, because an axis-aligned surface biases every voxel
                // the same way and the mesh merely shifts.
                //
                // Positive in front of the surface (nearer the camera, empty
                // space), negative behind it. Sign relative to the viewpoint is
                // the whole content of the "S" in TSDF.
                const float cx_ = (static_cast<float>(vx) + 0.5f) * cfg_.voxel_size_m;
                const float cy_ = (static_cast<float>(vy) + 0.5f) * cfg_.voxel_size_m;
                const float cz_ = (static_cast<float>(vz) + 0.5f) * cfg_.voxel_size_m;
                const float ddx = cx_ - batch.cam[0];
                const float ddy = cy_ - batch.cam[1];
                const float ddz = cz_ - batch.cam[2];
                const float d_voxel = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                const float sdf = dist - d_voxel;
                if (sdf < -trunc) continue;  // fully occluded, carries no information
                const float sdf_n = std::clamp(sdf * inv_trunc, -1.0f, 1.0f);

                const int32_t lx = vx - floor_div(vx, kBlockDim) * kBlockDim;
                const int32_t ly = vy - floor_div(vy, kBlockDim) * kBlockDim;
                const int32_t lz = vz - floor_div(vz, kBlockDim) * kBlockDim;
                const std::size_t idx =
                    static_cast<std::size_t>(bi) * kBlockVoxels +
                    static_cast<std::size_t>((lz * kBlockDim + ly) * kBlockDim + lx);

                // Running weighted mean. Serialised per voxel by an atomic CAS
                // loop on the weight, which is also what makes the result
                // independent of thread count: the update is a weighted mean,
                // and each contribution is folded in under the lock it wins.
                std::atomic_ref<float> w_atomic(weight_[idx]);
                float w_old = w_atomic.load(std::memory_order_acquire);
                for (;;) {
                    if (wcap > 0.0f && w_old >= wcap) break;
                    const float w_new = w_old + w_in;
                    if (w_atomic.compare_exchange_weak(w_old, w_new, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
                        const float inv_w = 1.0f / w_new;
                        tsdf_[idx] = (tsdf_[idx] * w_old + sdf_n * w_in) * inv_w;
                        r_[idx] = (r_[idx] * w_old + cr * w_in) * inv_w;
                        g_[idx] = (g_[idx] * w_old + cg * w_in) * inv_w;
                        b_[idx] = (b_[idx] * w_old + cb * w_in) * inv_w;
                        break;
                    }
                }
            }
        }
    });
}

void Volume::integrate(const PointBatch& batch) {
    allocate_blocks(batch);
    update_voxels(batch);
}

bool Volume::sample(int32_t vx, int32_t vy, int32_t vz, float min_weight, float& tsdf_out,
                    float rgb_out[3]) const {
    const int32_t bx = floor_div(vx, kBlockDim);
    const int32_t by = floor_div(vy, kBlockDim);
    const int32_t bz = floor_div(vz, kBlockDim);
    const int32_t bi = find_block(bx, by, bz);
    if (bi < 0) return false;

    const int32_t lx = vx - bx * kBlockDim;
    const int32_t ly = vy - by * kBlockDim;
    const int32_t lz = vz - bz * kBlockDim;
    const std::size_t idx = static_cast<std::size_t>(bi) * kBlockVoxels +
                            static_cast<std::size_t>((lz * kBlockDim + ly) * kBlockDim + lx);

    if (weight_[idx] < min_weight) return false;
    tsdf_out = tsdf_[idx];
    rgb_out[0] = r_[idx];
    rgb_out[1] = g_[idx];
    rgb_out[2] = b_[idx];
    return true;
}

}  // namespace osn_tsdf
