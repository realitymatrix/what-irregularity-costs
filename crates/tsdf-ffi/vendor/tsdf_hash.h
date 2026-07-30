// tsdf_hash.h
//
// GPU sparse voxel-block-hashed TSDF. Companion to the dense
// sliding-window TSDF in tsdf.{h,cu} — same per-voxel accumulator
// semantics (sum_xyz, sum_rgb, weight, count, optional sum_sq), but
// blocks of voxels are allocated on demand wherever points are observed
// instead of pre-allocating a fixed N×N×N cube.
//
// Why: the dense cube is bounded by VRAM × voxel_size. To cover a 30 m
// outdoor scan at 100 mm voxel needs a 30 m cube = ~5 GB VRAM. Going
// finer (10 mm) blows the budget. The hash design only allocates blocks
// containing surface, so memory scales with surface area, not the
// volume of the world.
//
// === Block layout ===
//
// World is divided into uniform blocks of `BLOCK_DIM³` voxels each.
// BLOCK_DIM is fixed at compile time (8 — gives 4 KB per block at
// 8 fields × 4 B × 8³). Block coords are floor_div(world_voxel_coord,
// BLOCK_DIM). Within a block, voxel index is
// `vz·BLOCK_DIM² + vy·BLOCK_DIM + vx`.
//
// Per-block data is stored SoA in a `block_pool` of `pool_capacity_blocks`
// entries. Each block index in the pool gets its own contiguous slice
// of every per-voxel buffer. This keeps atomics cheap: same access
// pattern as the dense grid, just routed through a hash lookup first.
//
// === Hash table ===
//
// Open-addressing with linear probing. `hash_size` ≥ 2 × pool_capacity.
// Each entry holds (block_coord_x, block_coord_y, block_coord_z,
// block_idx) packed into 16 B. Empty entries store BLOCK_IDX_EMPTY
// (sentinel). Insertion uses atomicCAS to claim an empty slot; lookup
// linear-probes until the matching coord is found or empty is hit.
//
// === Allocation ===
//
// On integrate-time miss, a thread atomicAdd-claims a free block index
// from a counter, then atomicCAS-inserts into the hash table. The
// claimed index ≥ pool_capacity_blocks signals OOM (reverted by host
// for diagnostics; new points dropped). v0 has NO eviction; once full,
// further new blocks are dropped silently. Future v1 can layer LRU.
//
// === Concurrency ===
//
// All public C ABI calls are sequential per-handle from the host
// (matching the dense tsdf_grid contract). Per-instance CUDA stream so
// multiple TsdfHash handles can run concurrent kernels on the same
// device — the same parallelism story as TsdfGrid.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TsdfHash TsdfHash;

// Block dimension is fixed at compile time. 8³ = 512 voxels per block,
// SoA-stored across 8 (or 11 with sum_sq) fields per voxel.
//   8: 4 KB (8 fields) or 5.5 KB (11 fields) per block
//  16: 32 KB (8 fields) or 44 KB (11 fields) per block
// 8³ keeps allocations small + parallel-friendly without exploding
// the hash entry count for typical scenes.
#define TSDF_HASH_BLOCK_DIM 8

// Create a sparse-hash TSDF.
//   voxel_size_m            : edge length of one voxel in meters
//   pool_capacity_blocks    : maximum number of blocks that can be
//                             allocated. Each block is BLOCK_DIM³ × 8
//                             field × 4 B = 4 KB. Choose to cap VRAM
//                             at the budget you can afford (e.g.
//                             1 048 576 blocks ≈ 4 GB for the voxel
//                             buffers, plus ~16 MB for the hash
//                             table at 2× pool capacity).
//   committed_cap_points    : maximum number of Omnixels the
//                             committed-output buffer can hold
//                             between drain calls. Like the dense
//                             grid: voxels evicted by host-side
//                             "drain N least-recently-touched blocks"
//                             land here.
//
// Returns NULL on allocation failure.
TsdfHash* tsdf_hash_create(
    float voxel_size_m,
    int pool_capacity_blocks,
    int committed_cap_points
);

// Release all device memory. Safe to call with NULL.
void tsdf_hash_destroy(TsdfHash* h);

// Add n_points world-space points to the hash. Each point's voxel coord
// is computed; the containing block is looked up (or atomically
// allocated) and the voxel within it is updated by atomicAdd.
//
// Identical I/O contract to tsdf_grid_add_points.
//   positions : 3*n_points floats interleaved
//   colors    : 3*n_points bytes (or NULL → neutral grey 128)
//   weights   : n_points floats (or NULL → uniform 1.0)
//
// Returns 0 on success, nonzero on launch / copy error. Block-pool
// overflow does NOT return an error — overflowing points are silently
// dropped, mirroring the dense grid's out-of-window contract.
int tsdf_hash_add_points(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points
);

// Same as tsdf_hash_add_points but adds a per-batch camera-radius gate.
// Points farther than `radius_m` from (cam_x, cam_y, cam_z) are skipped
// before block lookup, so they don't burn pool slots on far-field
// background pixels. radius_m <= 0 disables the gate (legacy behavior).
//
// All points in one call must come from the same camera origin (caller
// is expected to break per-frame batches when radius > 0).
int tsdf_hash_add_points_radius(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points,
    float cam_x, float cam_y, float cam_z,
    float radius_m
);

// Full-featured integrate. Stamps every block touched by points in this
// batch with `chunk_id` so LRU eviction can identify stale blocks.
// Callers should advance chunk_id monotonically across batches (typical:
// the per-chunk loop counter); blocks not touched in K chunks become
// candidates for tsdf_hash_evict_older_than(current - K).
int tsdf_hash_add_points_chunk(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points,
    int chunk_id,
    float cam_x, float cam_y, float cam_z,
    float radius_m
);

// Device-pointer variant of tsdf_hash_add_points_chunk. Skips the
// internal H2D upload and reads positions/colors/weights directly from
// caller-supplied device pointers. d_colors and d_weights may be 0
// (NULL) for the same neutral-grey / uniform-1.0 fallbacks the
// host-pointer path supports.
//
// Designed for the chunk-loop case where backproject already produces
// world-space points on device — feeding them into TSDF integrate
// without a CPU roundtrip eliminates ~50-80 ms/chunk of marshalling +
// PCIe transfer.
int tsdf_hash_add_points_chunk_device(
    TsdfHash* h,
    uint64_t d_positions,
    uint64_t d_colors,
    uint64_t d_weights,
    int n_points,
    int chunk_id,
    float cam_x, float cam_y, float cam_z,
    float radius_m
);

// Extract all currently-allocated blocks' voxels into a host buffer,
// applying the same gates as the dense grid:
//   min_weight, min_count, max_spread_frac, drain_committed
// `drain_committed` empties the per-instance committed buffer (used
// when the host has explicitly evicted blocks via tsdf_hash_evict_*).
// v0 has no host-driven eviction so committed is always empty; the
// flag is plumbed for API parity.
//
// Returns the number of voxels written, or -1 if buffer_cap was
// insufficient (caller should retry with a larger buffer).
int tsdf_hash_extract_points(
    TsdfHash* h,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float max_spread_frac,
    int drain_committed
);

// Same as tsdf_hash_extract_points but with a GPU-side downsample gate
// for delta-quality control. `keep_thresh_256` selects a deterministic
// subset:
//   0          : keep all (no decimation, identical to base call)
//   1..255     : keep iff hash(world_voxel_coord) % 256 < keep_thresh
//                ⇒ approximate density factor = keep_thresh / 256
//   256+       : same as 0 (no decimation)
//
// Decision is computed BEFORE the output-buffer atomicAdd so dropped
// voxels don't waste output slots. Hash is stable per-coord across
// chunks — viewer's accumulator never sees a voxel flicker on/off.
int tsdf_hash_extract_points_target(
    TsdfHash* h,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float max_spread_frac,
    int drain_committed,
    int keep_thresh_256
);

// ── Mesh mode (projective TSDF + marching cubes) ──
//
// Enable/disable mesh mode. When on, tsdf_hash_add_points_chunk splats a
// truncated signed distance along each point's view ray (instead of
// centroid binning), and tsdf_hash_extract_mesh runs marching cubes.
// trunc_m = TSDF truncation half-width in metres; <=0 → 4×voxel. Reset
// the grid before toggling mid-stream. Returns 0 on success.
int tsdf_hash_set_mesh_mode(TsdfHash* h, int on, float trunc_m);

// Extract a triangle mesh from the TSDF. dirty_only != 0 re-meshes only
// blocks touched in current_chunk (incremental). Host buffers must hold
// vert_cap vertices: buffer_posnor[vert_cap*6] (pos3,nor3),
// buffer_rgb[vert_cap*3], buffer_block[vert_cap] (owner block_idx).
// Vertices are a triangle soup. Returns vertex count or -1 on overflow.
int tsdf_hash_extract_mesh(
    TsdfHash* h,
    float* buffer_posnor,
    unsigned char* buffer_rgb,
    int* buffer_block,
    int vert_cap,
    float min_weight,
    float iso,
    int dirty_only,
    int current_chunk
);

// Diagnostic: how many blocks are currently allocated (active + held
// for committed).
int tsdf_hash_block_count(TsdfHash* h);
int tsdf_hash_committed_count(TsdfHash* h);

// Cap the per-voxel accumulated weight, same as the dense grid.
// Returns 0 on success, nonzero on null handle.
int tsdf_hash_set_weight_cap(TsdfHash* h, float cap);

// Built-in TSDF downsampling. Sets a hash-decimation threshold applied
// at integrate time (per world-voxel-coord) so dropped voxels never
// enter the pool — the TSDF itself acts as the downsampler. Effect:
//
//   thresh_256 = 0   : disabled (default, full quality)
//   thresh_256 = 64  : ~25% of voxels integrated, ~25% pool fill,
//                      ~25% integrate-kernel atomics, ~25% delta size
//   thresh_256 = 128 : ~50% of all the above
//   thresh_256 = 256 : same as 0 (no decimation)
//
// Decision is stable across chunks (same world-voxel-coord always
// kept/dropped), so changing the threshold mid-mission is safe — only
// the previously-uncovered voxel set starts integrating fresh on the
// next chunk. Returns 0 on success, nonzero on null handle.
int tsdf_hash_set_decimation_thresh(TsdfHash* h, int thresh_256);

// Variable-resolution (LOD) integration. Sets the radius of the
// finest LOD band: points within `radius_m` of the camera land at the
// finest level (voxel_size_m). Each successive band of doubled radius
// drops a level — voxel doubles. radius_m = 0 disables LOD (legacy
// single-resolution behavior, every point at finest level).
//
// Stable across chunks; safe to set/clear mid-mission. Returns 0 on
// success, nonzero on null handle.
int tsdf_hash_set_lod_finest_radius(TsdfHash* h, float radius_m);

// Currently-allocated block count = high_water - free_stack_depth.
// Use to check if a hash-table reset is safe (active_count == 0
// ⇒ wiping the table won't lose any live block). Returns -1 on error.
int tsdf_hash_active_count(TsdfHash* h);

// Telemetry: monotonic count of points dropped because the block pool
// was exhausted. Read after each integrate to see if the pool is
// silently saturating; use the delta vs the previous read as the
// "this chunk dropped N points" signal.
unsigned long long tsdf_hash_drop_count(TsdfHash* h);

// Reset the hash table to all-empty and counters to zero. SAFE only
// when no live blocks exist in the pool. Use after a full-evict sweep
// that drained the pool — without it, tombstones accumulate across
// repeated evictions and saturate the table.
int tsdf_hash_reset_table(TsdfHash* h);

// LRU eviction. Frees blocks whose last_touched_chunk stamp is < the
// supplied threshold and spills their converged voxels (those with
// weight > 0) into the caller-provided buffers. Spilled voxels carry
// the running mean position and color — same format as the output of
// tsdf_hash_extract_points — so the host can append them to a
// persistent mesh accumulator without further processing.
//
//   evict_before_chunk : threshold; blocks with stamp < this are freed
//   out_xyz, out_rgb   : spill output, must hold out_cap voxel slots
//   out_cap            : capacity of the spill buffer in voxels
//   n_evicted_blocks   : (optional, may be NULL) receives count of
//                        blocks freed in this pass
//
// Returns:
//   >= 0   : number of voxels written to (out_xyz, out_rgb). If equal
//            to out_cap, some converged voxels were dropped — caller
//            should grow the buffer and retry the next sweep.
//   < 0    : launch / copy error (handle remains valid).
//
// MUST NOT be called concurrently with tsdf_hash_add_points* on the
// same handle. The host serializes the two via stream sync.
int tsdf_hash_evict_older_than(
    TsdfHash* h,
    int evict_before_chunk,
    float* out_xyz,
    unsigned char* out_rgb,
    int out_cap,
    int* n_evicted_blocks
);

// Same as tsdf_hash_evict_older_than but applies the same hash-
// decimation gate documented on tsdf_hash_extract_points_target to
// the spilled converged voxels. `keep_thresh_256 = 0` is no-op.
int tsdf_hash_evict_older_than_target(
    TsdfHash* h,
    int evict_before_chunk,
    float* out_xyz,
    unsigned char* out_rgb,
    int out_cap,
    int* n_evicted_blocks,
    int keep_thresh_256
);

#ifdef __cplusplus
}  // extern "C"
#endif
