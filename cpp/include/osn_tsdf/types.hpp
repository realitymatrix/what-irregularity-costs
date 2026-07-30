// Core types for the hash-blocked TSDF volume.
//
// Shared by every arm this project implements, so the layout decisions here
// are the ones the comparison holds constant. Anything an arm is free to vary
// (threading, kernel shape, launch config) is deliberately absent.
#pragma once

#include <cstdint>
#include <cstddef>

namespace osn_tsdf {

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
inline constexpr int32_t kEmptyKey = INT32_MIN;

/// Integer block coordinate in world-block space.
struct BlockCoord {
    int32_t x, y, z;

    friend bool operator==(const BlockCoord& a, const BlockCoord& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

/// One open-addressed hash slot.
///
/// `key` doubles as the occupancy flag: `kEmptyKey` means free. Packing the
/// flag into the key is what lets insertion be a single compare-exchange on
/// one 32-bit word, which is the operation the GPU arms have to express and
/// the CPU arm therefore mirrors rather than replacing with a mutex.
struct HashEntry {
    int32_t key;        // BlockCoord::x, or kEmptyKey
    int32_t y, z;       // remaining coordinates
    int32_t block_idx;  // index into the block pool
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
