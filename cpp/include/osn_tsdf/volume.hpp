// Hash-blocked TSDF volume, CPU implementation (arm A2).
//
// The volume owns allocation and the hash table; a backend owns the per-voxel
// arithmetic. That split is the point of the whole project: A3/A4/A5 replace
// the backend and reuse this bookkeeping, so a measured difference between
// arms is a difference in the kernels rather than in how blocks were found.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "osn_tsdf/types.hpp"

namespace osn_tsdf {

/// Surface extraction uses marching *tetrahedra*, not marching cubes.
///
/// Each cube is split into 6 tetrahedra; a tetrahedron has 4 corners and so 16
/// sign cases, each producing 0, 1 or 2 triangles. That is a few lines of
/// derivable logic instead of a 256-entry lookup table, and it is watertight by
/// construction because adjacent tetrahedra share a face and therefore share
/// the interpolated vertices on it.
///
/// Cost: roughly 2x the triangles of marching cubes for the same surface.
/// Accepted deliberately. Every arm uses the same extractor, so the comparison
/// is unaffected, and a table transcription error would be a correctness bug
/// that no amount of benchmarking discipline would catch.
class Volume {
public:
    explicit Volume(const VolumeConfig& cfg);
    ~Volume();

    // Neither copyable nor movable. The atomic members are not movable, and
    // moving a volume mid-integration would be a race rather than a transfer.
    // Callers that need to relocate one should hold it behind a pointer, which
    // is what the C API does.
    Volume(const Volume&) = delete;
    Volume& operator=(const Volume&) = delete;
    Volume(Volume&&) = delete;
    Volume& operator=(Volume&&) = delete;

    /// Integrate one batch of world-space points.
    ///
    /// Two passes, matching what the GPU arms must do: allocate the blocks the
    /// batch touches, then update voxels. Splitting them is what allows arm A5a
    /// to run a Triton update over blocks allocated by CUDA.
    void integrate(const PointBatch& batch);

    /// Allocate blocks for a batch without updating any voxel.
    void allocate_blocks(const PointBatch& batch);

    /// Update voxels, assuming the blocks already exist.
    void update_voxels(const PointBatch& batch);

    /// Extract a triangle soup. Returns the vertex count, or -1 if `out` was
    /// too small, in which case nothing is written.
    int32_t extract_mesh(const MeshBuffers& out, float min_weight, float iso) const;

    int32_t block_count() const { return block_count_.load(std::memory_order_relaxed); }
    uint64_t drop_count() const { return drop_count_.load(std::memory_order_relaxed); }

    const VolumeConfig& config() const { return cfg_; }

    /// Number of worker threads. 0 selects hardware concurrency.
    void set_thread_count(int n);
    int thread_count() const { return n_threads_; }

private:
    /// Find a block, or -1. Lock-free, safe to call concurrently with itself.
    int32_t find_block(int32_t bx, int32_t by, int32_t bz) const;

    /// Find or insert. Safe to call concurrently. Returns -1 when the pool is
    /// exhausted, having incremented the drop counter.
    int32_t find_or_insert_block(int32_t bx, int32_t by, int32_t bz);

    /// Trilinear-free sample of the stored field at integer voxel coordinates.
    /// Returns false when the voxel is absent or below `min_weight`, which is
    /// what makes the extracted surface stop at unobserved space rather than
    /// closing over it.
    bool sample(int32_t vx, int32_t vy, int32_t vz, float min_weight, float& tsdf_out,
                float rgb_out[3]) const;

    VolumeConfig cfg_;
    uint32_t hash_mask_ = 0;

    std::vector<HashEntry> table_;
    std::atomic<int32_t> block_count_{0};
    std::atomic<uint64_t> drop_count_{0};

    /// Structure of arrays, one span of `kBlockVoxels` per allocated block.
    ///
    /// SoA rather than AoS: the update touches tsdf and weight on every point
    /// but colour only when present, and the extractor reads tsdf alone across
    /// whole blocks. Interleaving would pull colour into cache on every
    /// tsdf-only sweep.
    ///
    /// These hold WEIGHTED SUMS, not means. The mean is formed on read.
    ///
    /// That is a correctness requirement, not a style choice. Storing a running
    /// mean requires reading the old mean, combining, and writing back, and
    /// there is no way to make that atomic across four arrays with one
    /// compare-exchange. An earlier version did exactly that under a CAS on the
    /// weight and had a lost-update race: two threads winning consecutive CASes
    /// could interleave their read-modify-write of the mean. It survived the
    /// analytic test because the lost updates average out, which is how races
    /// like it normally survive testing. Sums commute, so plain atomic adds are
    /// sufficient and the result is order-independent up to float associativity.
    std::vector<float> tsdf_;
    std::vector<float> weight_;
    std::vector<float> r_, g_, b_;

    /// Block coordinate per pool slot, for the extractor to walk allocations
    /// without rescanning the hash table.
    std::vector<BlockCoord> block_coord_;

    int n_threads_ = 0;
};

}  // namespace osn_tsdf
