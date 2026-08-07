// Core types for the hash-blocked TSDF volume.
//
// Shared by every arm this project implements, so the layout decisions here
// are the ones the comparison holds constant. Anything an arm is free to vary
// (threading, kernel shape, launch config) is deliberately absent.
#pragma once

#include <cstdint>
#include <cstddef>

namespace osn_tsdf {

/// Scratch slots appended to the hash table, for the Triton arm's inert CASes.
///
/// Sized for the largest region any experiment asks for, not for the region
/// actually used: the kernel takes a runtime mask, so the *used* region is
/// chosen per launch and only this allocation has to be big enough. Costs
/// 16 MiB per volume, small beside the voxel pool.
///
/// The size is load-bearing, not a tuning knob. `tl.atomic_cas` takes no mask,
/// so every resolved lane still issues a CAS somewhere; with the original
/// 256-slot region every program in the grid collided on the same 256
/// addresses. That was measured to be the entire reason arm A5's allocate
/// failed to scale with SM count while the other arms tracked the machine, and
/// sizing it properly made the kernel 3.5x faster. See
/// docs/SCRATCH-HYPOTHESIS.md.
///
/// Anything sizing a volume must account for this, or it will underestimate by
/// 16 MiB per volume.
#ifndef OSN_TSDF_SCRATCH_SLOTS
#define OSN_TSDF_SCRATCH_SLOTS (1u << 20)
#endif
inline constexpr uint32_t kScratchSlots = OSN_TSDF_SCRATCH_SLOTS;

/// Voxels per side of a block. Compile-time so indexing folds to shifts.
///
/// 8 gives 512 voxels per block: small enough that a block is a cheap
/// allocation unit and the hash table stays short, large enough that one hash
/// lookup is amortised over a useful amount of work. The GPU arms want this as
/// their CUDA block size too, and 512 threads is a legal launch shape.
inline constexpr int kBlockDim = 8;
inline constexpr int kBlockVoxels = kBlockDim * kBlockDim * kBlockDim;

/// Sentinel for an empty hash slot. Chosen so a zeroed table is *not* empty,
/// which turns "forgot to initialise the table" into an immediate miss rather
/// than a silently valid-looking entry at block coordinate (0,0,0).
///
/// -1 cannot collide with a real key: `pack_coord` biases each axis to be
/// non-negative, so every packed key is in [0, 2^63).
inline constexpr int64_t kEmptyKey = -1;

/// Half-range per axis, in blocks. 2^20 blocks at 8 voxels of 1 cm is +/- 80 km.
inline constexpr int64_t kCoordBias = 1 << 20;
inline constexpr int64_t kCoordBits = 21;

/// Pack a block coordinate into one 64-bit key.
///
/// The whole coordinate must live in a single word so that one compare-exchange
/// publishes all of it. An earlier layout CAS'd `x` and then wrote `y` and `z`
/// separately: a concurrent reader could see a matching `x` with stale `y`/`z`,
/// conclude the slot was a different block, and probe onward. Under low CPU
/// contention that is invisible; with hundreds of thousands of GPU threads it
/// degenerated into full-table scans and the integrate never finished.
inline int64_t pack_coord(int32_t x, int32_t y, int32_t z) {
    const int64_t bx = static_cast<int64_t>(x) + kCoordBias;
    const int64_t by = static_cast<int64_t>(y) + kCoordBias;
    const int64_t bz = static_cast<int64_t>(z) + kCoordBias;
    return (bx << (2 * kCoordBits)) | (by << kCoordBits) | bz;
}

/// Integer block coordinate in world-block space.
struct BlockCoord {
    int32_t x, y, z;

    friend bool operator==(const BlockCoord& a, const BlockCoord& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

/// One open-addressed hash slot.
///
/// `key` is the packed block coordinate and doubles as the occupancy flag:
/// `kEmptyKey` means free. One 64-bit word holds the entire coordinate, so
/// insertion is a single compare-exchange that publishes all of it atomically.
/// That is the operation the GPU arms have to express, and the CPU arm mirrors
/// it rather than substituting a mutex, so the comparison stays fair.
struct HashEntry {
    int64_t key;        // pack_coord(...), or kEmptyKey
    int32_t block_idx;  // index into the block pool
    int32_t pad;        // keep the struct 16-byte aligned
};

/// Construction parameters.
struct VolumeConfig {
    float voxel_size_m = 0.01f;
    int32_t pool_capacity_blocks = 1 << 16;
    /// TSDF truncation half-width in metres. <= 0 selects 4 * voxel_size.
    float trunc_m = -1.0f;
    /// Cap on accumulated weight. Once a voxel reaches it, later observations
    /// stop moving it. Without a cap, an early surface estimate is dragged by
    /// every subsequent frame including drifted ones. <= 0 disables.
    float weight_cap = 32.0f;

    float trunc() const {
        return trunc_m > 0.0f ? trunc_m : 4.0f * voxel_size_m;
    }

    /// Bytes the block pool will occupy. Worth checking before allocating: an
    /// oversized pool fails as a bare allocation error that does not name the
    /// pool as the cause.
    std::size_t pool_bytes() const {
        // tsdf + weight + r + g + b = 5 floats per voxel.
        return static_cast<std::size_t>(pool_capacity_blocks) * kBlockVoxels * 5 *
               sizeof(float);
    }
};

/// A batch of world-space points to integrate.
///
/// World space, so pose composition happens once in the caller and cannot
/// differ between arms.
struct PointBatch {
    const float* positions = nullptr;  // 3 * n, interleaved xyz
    const uint8_t* colors = nullptr;   // 3 * n, or null for neutral grey
    const float* weights = nullptr;    // n, or null for uniform 1.0
    int32_t n = 0;
    int32_t chunk_id = 0;
    /// Camera origin. Required, not optional: the *sign* of a truncated signed
    /// distance is only defined relative to a viewpoint. A TSDF integrated
    /// without it degenerates to centroid binning, which produces no zero
    /// crossing and therefore no surface.
    float cam[3] = {0.0f, 0.0f, 0.0f};
    /// Far-field cutoff in metres. <= 0 disables.
    float radius_m = 0.0f;
};

/// Output triangle soup: 6 floats per vertex (pos xyz, normal xyz).
struct MeshBuffers {
    float* posnor = nullptr;
    uint8_t* rgb = nullptr;
    int32_t capacity_vertices = 0;
};

/// Spatial hash over block coordinates.
///
/// The multiply-xor mix is the usual one for voxel hashing (Teschner et al.).
/// It matters that this is cheap and identical across arms: a different mix
/// changes probe sequences, which changes which blocks share cache lines, and
/// that would show up as a performance difference attributed to the language.
inline uint32_t hash_block(int32_t x, int32_t y, int32_t z, uint32_t mask) {
    const uint32_t h = static_cast<uint32_t>(x) * 73856093u ^
                       static_cast<uint32_t>(y) * 19349663u ^
                       static_cast<uint32_t>(z) * 83492791u;
    return h & mask;
}

/// floor(a / b) for positive b, correct for negative a.
///
/// Plain integer division truncates toward zero, which mirrors block
/// coordinates across the origin and produces a seam exactly at x=0. Cheap to
/// get wrong and invisible unless a scan crosses the origin.
inline int32_t floor_div(int32_t a, int32_t b) {
    const int32_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

}  // namespace osn_tsdf
