// tsdf_hash.cu
//
// GPU open-addressing voxel-block hashmap implementing the API in
// tsdf_hash.h. Companion to the dense sliding-window TSDF in tsdf.cu;
// see that file for the per-voxel SoA accumulation contract — this
// file uses the same accumulators on a sparse block pool.

#include "tsdf_hash.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Sentinels used in the hash table.
#define HASH_EMPTY_KEY (int32_t)(-2147483647 - 1)  // INT32_MIN
#define HASH_TOMBSTONE (int32_t)(-2147483647)       // INT32_MIN+1 (reserved for future eviction)

// Block constants.
#define BD       TSDF_HASH_BLOCK_DIM
#define BD3      (TSDF_HASH_BLOCK_DIM * TSDF_HASH_BLOCK_DIM * TSDF_HASH_BLOCK_DIM)

// ── LOD (variable-resolution) constants ──
// Number of discrete level-of-detail levels. Compile-time fixed for
// fast level-arithmetic in kernels (shifts not divisions). Level 0 is
// the COARSEST (largest voxels), level N_LEVELS-1 is the FINEST
// (matches the user's `voxel_size_m`). Voxel size at level L equals
// `voxel_finest * 2^(N_LEVELS-1-L)`. Block extent in metres = voxel × BD.
//
// 5 levels gives a 16× span — finest = 10 mm → coarsest = 160 mm —
// which covers most practical scene-scale variation (close-up scans
// up to long aerial walks). Levels are kept opt-in via
// OPENSTRATE_TSDF_GPU_HASH_VARIABLE_RES; when disabled, all blocks
// pass level=N_LEVELS-1 and behavior is byte-identical to the single-
// resolution legacy path.
#define TSDF_HASH_N_LEVELS 5

// Hash entry: 4 × i32 = 16 B. Coords identify the block in world tile-
// coords; block_idx is the slot in the block pool.
struct __align__(16) HashEntry {
    int32_t bx;
    int32_t by;
    int32_t bz;
    int32_t block_idx;
};

struct TsdfHash {
    float voxel_size_m;
    int   pool_capacity;
    int   hash_size;        // 2 × pool_capacity, power of 2 for fast modulo

    // Per-instance stream so multiple TsdfHash handles run kernels
    // concurrently on the same GPU.
    cudaStream_t stream;

    // ── Hash table (open-addressing, linear probing) ──
    // Each entry is HashEntry. Empty slots have bx=HASH_EMPTY_KEY.
    HashEntry* d_hash_table;
    int*       d_hash_size_dev; // single counter: number of live blocks

    // ── Block pool: per-voxel SoA buffers, one slice of length
    // pool_capacity × BD³ per field. ──
    float*         d_sum_x;
    float*         d_sum_y;
    float*         d_sum_z;
    float*         d_sum_r;
    float*         d_sum_g;
    float*         d_sum_b;
    float*         d_weight;
    unsigned int*  d_count;

    // Per-block "is allocated" flag. Used by the extract kernel to
    // skip empty slots; updated atomically when a thread claims a
    // free block. Index space matches block_idx.
    int*           d_block_alloc;
    // Per-block world coords (parallel to d_block_alloc) — extract
    // needs them to compute world-space output positions.
    int3*          d_block_coord;

    // ── Free-list: high-water mark for first-time allocations + a
    // recycle stack populated by LRU eviction. Allocators try the
    // free-stack first (atomicSub on d_free_count, read d_free_stack);
    // on miss they fall through to the high-water counter. ──
    int* d_block_alloc_counter;   // high-water mark (monotonic)
    int* d_free_stack;            // [pool_capacity] of evicted block_idx values
    int* d_free_count;            // top-of-stack pointer (initial 0)

    // ── Eviction state: per-block last-touched-chunk stamp; integrate
    // kernel writes here, evict kernel reads. last_touched < threshold
    // means "hasn't been touched in the recent window — recycle me." ──
    int* d_last_touched_chunk;    // [pool_capacity], initial -1

    // ── LOD state ──
    // Per-block level (0 = coarsest, N_LEVELS-1 = finest). When
    // variable-resolution is disabled (default) every block is at the
    // finest level and this array contains constant N_LEVELS-1; the
    // integrate / extract / evict kernels still read it so the same
    // code path serves both modes.
    int* d_block_level;           // [pool_capacity], initial N_LEVELS-1

    // ── Telemetry: monotonic counter of points dropped because the
    // block pool was exhausted. Surfaced by tsdf_hash_drop_count() so
    // the host can see when the pool is silently saturating. ──
    unsigned long long* d_n_dropped_pool_full;

    // ── Committed buffer for explicit eviction (host-driven, not
    // populated in v0; reserved for future LRU eviction). ──
    float*         d_committed_xyz;
    unsigned char* d_committed_rgb;
    unsigned int*  d_committed_count_ptr;
    int            committed_cap;

    // ── Temp buffers for input upload, grow on demand. ──
    float*         d_tmp_positions;
    unsigned char* d_tmp_colors;
    float*         d_tmp_weights;
    int            tmp_capacity;

    float          weight_cap;

    // ── Built-in downsampling ──
    // Hash decimation threshold applied INSIDE the integrate kernel at
    // each point's destination voxel coord. 0 = disabled (no
    // decimation). 1..255 = keep iff decimation_hash(world_voxel_coord)
    // < thresh. Effect cascades through the whole pipeline:
    //   - dropped points return early before lookup_or_insert      ⇒ fewer atomicAdds
    //   - voxels that never receive a hit don't get blocks alloc'd ⇒ smaller pool fill
    //   - extract/evict naturally see only kept voxels             ⇒ smaller delta payload
    // So the TSDF acts as the downsampler — one knob reduces both
    // memory pressure AND integrate-kernel compute proportionally.
    int            decimation_thresh_256;

    // ── LOD configuration ──
    // Radius (metres) of the finest LOD band: points within this
    // distance from the camera land at the finest level (voxel_size_m
    // exactly). Each successive band of doubled radius drops a level
    // (voxel doubles). 0 disables LOD — every point lands at the
    // finest level, behavior is byte-identical to the legacy single-
    // resolution code path. Set via tsdf_hash_set_lod_finest_radius.
    float          lod_finest_radius_m;

    // ── Mesh mode: projective TSDF + marching cubes ──
    // When mesh_mode != 0, tsdf_hash_add_points_chunk dispatches
    // hash_integrate_tsdf_kernel instead of the legacy centroid binner:
    // each (dense) point splats a TRUNCATED signed distance along its
    // view ray (cam → point) into d_tsdf as a running weighted mean in
    // [-1,1]. d_weight + d_count + d_sum_{r,g,b} are reused (weight +
    // colour); d_sum_{x,y,z} are unused. hash_marching_cubes_kernel then
    // reads d_tsdf to emit a triangle mesh. Default off → every legacy
    // code path is byte-identical.
    float* d_tsdf;          // [pool_capacity*BD3], running mean signed dist
    // Per-block "touched this chunk" stamp, written on EVERY voxel update
    // by the TSDF integrate kernel (unlike d_last_touched_chunk, which is
    // FIFO/alloc-time only). Marching cubes uses it to re-mesh only the
    // blocks that changed this chunk. [pool_capacity], initial -1.
    int*   d_dirty_chunk;
    int    mesh_mode;       // 0 = legacy centroid binning (default)
    float  trunc_m;         // TSDF truncation half-width (m); <=0 → 4*voxel
    // Pre-allocated marching-cubes scratch: hoisted out of
    // tsdf_hash_extract_mesh because cudaMalloc was being called 4× per
    // call (posnor, rgb, block_idx, count) and on a busy GPU each cost
    // ~10-50 ms — dominant latency contributor at ~600 ms total per
    // chunk. Sized at lazy first use (or grown if vert_cap rises). Freed
    // in tsdf_hash_destroy. All four nullptr/0 until first extract.
    float*         d_mesh_posnor;
    unsigned char* d_mesh_rgb;
    int*           d_mesh_block;
    unsigned int*  d_mesh_count;
    int            mesh_scratch_vert_cap;
    // Dirty-block compaction: a cheap pre-scan kernel writes the indices
    // of blocks tagged with current_chunk into d_dirty_list, and stores
    // the count in d_dirty_count. The marching-cubes kernel then launches
    // with grid=N_dirty (typically ~300) instead of pool_capacity
    // (~65 k), eliminating ~65 ms of empty-block launch overhead per
    // extract call. Sized to pool_capacity at first mesh-mode init.
    int*           d_dirty_list;
    unsigned int*  d_dirty_count;
};

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[tsdf_hash.cu] CUDA error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return _e;                                                         \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_NULL(expr)                                                  \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[tsdf_hash.cu] CUDA error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return nullptr;                                                    \
        }                                                                      \
    } while (0)

// ── Hashing ──
//
// Spatial hash: combine block coords with three large primes, xor and
// take modulo against the table size. Cheap, well-distributed for
// world-aligned grids. The `level` term ensures blocks at the same
// world tile but different LOD levels hash to different slots.
__device__ __host__ static inline uint32_t hash_block_coord(int bx, int by, int bz, int level, int hash_size) {
    uint32_t h = (uint32_t)bx * 73856093u
               ^ (uint32_t)by * 19349663u
               ^ (uint32_t)bz * 83492791u
               ^ (uint32_t)level * 31337u;
    return h & (uint32_t)(hash_size - 1);  // hash_size is power of 2
}

// Floor-divide a world-voxel coord by BLOCK_DIM to get block coord.
// Need to handle negatives correctly (Rust-style truncation gives
// wrong sign for negative-X regions).
__device__ static inline int floor_div_int(int a, int b) {
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

// Look up or atomically insert a block. Returns the block_idx in the
// pool, or -1 if the table is full or the pool is exhausted.
//
// Linear probing with atomicCAS on the bx field. Tombstones are not
// used in v0 (no eviction); emptiness is signalled by bx == HASH_EMPTY_KEY.
//
// Concurrency notes (fixed 2026-05-01):
//   - All reads of e->bx, by, bz, block_idx use `volatile` casts so
//     NVCC cannot hoist them out of the outer probe loop. The pre-fix
//     plain reads + non-volatile spin (`while (e->block_idx < 0)`)
//     could be loop-invariant-cached into a register at -O3, producing
//     an indefinite spin under SIMT / Independent Thread Scheduling
//     when the racing writer was in the same warp. That manifested as
//     scene-dependent chunk-0 stalls (mission 96c052ec / ruins, plus
//     historical 95d5b142 / 0537d59e). Empirically tied to voxel-size
//     × camera-motion: bigger voxels concentrate more points per
//     block → higher same-warp lost-race probability → triggered
//     more often.
//   - Lost-race recovery now re-probes the SAME slot on the next
//     loop iteration (via probe-- then continue), instead of spinning
//     in place. This lets the warp scheduler dispatch the writing
//     thread between iterations.
//   - Total re-read budget is bounded across all slots (MAX_REREAD)
//     so a stuck writer can never produce an unbounded loop; on
//     budget exhaustion we drop the point (caller's add_points
//     contract permits silent drops on contention, same as a
//     pool-full condition).
// ── Spatial hash for built-in TSDF downsampling ──
//
// Maps an integer voxel coord (vx, vy, vz) to a uniformly-distributed
// 8-bit value. Used to gate voxels at integrate / extract / evict
// time: keep iff `decimation_hash(vx,vy,vz) < keep_thresh_256`. Same
// coord always hashes the same value, so the keep/drop decision is
// stable across chunks — pool, accumulator, and viewer all see a
// consistent subset.
//
// xorshift32-style mix; bias is fine for this purpose.
__device__ static inline unsigned int decimation_hash(int vx, int vy, int vz) {
    unsigned int h = (unsigned int)vx * 0x8da6b343u
                   ^ (unsigned int)vy * 0xd8163841u
                   ^ (unsigned int)vz * 0xcb1ab31fu;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h & 0xffu;
}

__device__ static int lookup_or_insert_block(
    HashEntry* hash_table,
    int hash_size,
    int* block_alloc_counter,
    int pool_capacity,
    int* free_stack,
    int* free_count,
    int* block_alloc,
    int3* block_coord,
    int* block_level,
    int* last_touched_chunk,
    int  current_chunk,
    int bx, int by, int bz,
    int level
) {
    uint32_t start = hash_block_coord(bx, by, bz, level, hash_size);
    // Bounded re-probing budget; see header comment above. 64 is
    // generous — racing writers publish block_idx in tens of cycles.
    const int MAX_REREAD = 64;
    int reread_budget = MAX_REREAD;
    // Bounded probing: at most hash_size probes; in practice the
    // fanout is short because hash_size ≥ 2× pool capacity. Tombstones
    // (HASH_TOMBSTONE) left by LRU eviction are skipped here — they're
    // only reused after we exhaust empty slots in the probe chain.
    for (int probe = 0; probe < hash_size; ++probe) {
        uint32_t slot = (start + (uint32_t)probe) & (uint32_t)(hash_size - 1);
        HashEntry* e = &hash_table[slot];

        // Volatile load of bx so the compiler cannot hoist it across
        // loop iterations.
        int cur_bx = *((volatile int*)&e->bx);
        if (cur_bx == bx) {
            // bx matches — but by/bz might still be initial-zero if
            // the winning writer hasn't published yet. Read
            // block_idx FIRST: that's the canonical "slot is live"
            // flag (writer's __threadfence before storing block_idx
            // makes by/bz globally visible BEFORE block_idx becomes
            // non-negative). If block_idx >= 0, by/bz are safely
            // readable; otherwise re-probe THIS slot.
            int idx = *((volatile int*)&e->block_idx);
            if (idx < 0) {
                if (--reread_budget <= 0) return -1;
                --probe;
                continue;
            }
            int cur_by = *((volatile int*)&e->by);
            int cur_bz = *((volatile int*)&e->bz);
            // Level lives in a parallel array; verify it matches too.
            // Two blocks at the same world tile but different LOD
            // levels coexist as separate entries — same (bx,by,bz),
            // different levels — and we must NOT collapse them.
            int cur_level = *((volatile int*)&block_level[idx]);
            if (cur_by == by && cur_bz == bz && cur_level == level) return idx;
            // Real hash collision (or different-level same-tile): bx
            // matches but the rest of the key differs. Keep probing.
            continue;
        }
        if (cur_bx == HASH_TOMBSTONE) {
            // Evicted slot. Skip — but DON'T terminate the probe chain;
            // our key may still be in a later slot from when this slot
            // was occupied by a different (collided) key.
            continue;
        }
        // Empty slot — try to claim it.
        if (cur_bx == HASH_EMPTY_KEY) {
            // Race: claim the entry by CAS-installing our bx.
            int prev = atomicCAS(&e->bx, HASH_EMPTY_KEY, bx);
            if (prev == HASH_EMPTY_KEY) {
                // We claimed the slot. Allocate a block index — try
                // the free-stack first (recycled from eviction), then
                // fall through to the high-water counter.
                int new_idx = -1;
                int prev_free = atomicSub(free_count, 1);
                if (prev_free > 0) {
                    // Claimed slot prev_free - 1 of the stack.
                    new_idx = free_stack[prev_free - 1];
                } else {
                    // Under-counted — restore so free_count never
                    // stays negative for long.
                    atomicAdd(free_count, 1);
                }
                if (new_idx < 0) {
                    new_idx = atomicAdd(block_alloc_counter, 1);
                    if (new_idx >= pool_capacity) {
                        // Pool full — undo the CAS so other lookups
                        // following us don't see a half-built entry.
                        e->bx = HASH_EMPTY_KEY;
                        return -1;
                    }
                }
                // Publish order matters. Write by/bz AND the level
                // (which lives in the parallel block_level[] array,
                // checked by readers after block_idx becomes visible)
                // first, then __threadfence to ensure they're globally
                // visible, THEN write block_idx. Readers gate on
                // block_idx (the canonical "slot is live" flag) via
                // volatile load — once they see block_idx >= 0,
                // they're guaranteed to see the matching by/bz/level.
                e->by = by;
                e->bz = bz;
                block_level[new_idx] = level;
                __threadfence();
                e->block_idx = new_idx;
                __threadfence();
                // Mark the block slot as allocated and stamp coord.
                block_alloc[new_idx] = 1;
                block_coord[new_idx] = make_int3(bx, by, bz);
                // FIFO eviction policy: stamp the block's "alloc time"
                // here, NOT on every subsequent lookup-hit. This way
                // sequential walking scans evict their oldest blocks
                // even when consecutive chunk bboxes overlap (which
                // would otherwise keep stamps "fresh" indefinitely
                // under LRU semantics, causing the pool-full death
                // spiral).
                last_touched_chunk[new_idx] = current_chunk;
                return new_idx;
            }
            // Lost the race — re-test this slot on the next iter so
            // we observe the winner's published bx/by/bz/block_idx
            // through the volatile loads above. The previous in-
            // place spin (`while ((idx = e->block_idx) < 0) {}`) was
            // hoistable by NVCC and deadlocked under SIMT/ITS when
            // the winning thread was in the same warp.
            if (--reread_budget <= 0) return -1;
            --probe;
            continue;
        }
        // Occupied by a different block — keep probing.
    }
    // Table full.
    return -1;
}

// ── integrate kernel ──
__global__ void hash_integrate_kernel(
    const float* __restrict__ positions,
    const unsigned char* __restrict__ colors,
    const float* __restrict__ weights,
    int n_points,
    float voxel_size_m,
    float weight_cap,
    HashEntry* hash_table,
    int hash_size,
    int* block_alloc_counter,
    int pool_capacity,
    int* free_stack,
    int* free_count,
    int* block_alloc,
    int3* block_coord,
    int* block_level,
    int* last_touched_chunk,
    int  current_chunk,
    float* sum_x, float* sum_y, float* sum_z,
    float* sum_r, float* sum_g, float* sum_b,
    float* weight_buf, unsigned int* count_buf,
    unsigned long long* n_dropped_pool_full,
    // Camera-radius cap. cam_origin_world is per-frame; we expose it at
    // chunk granularity by only filtering when radius > 0. radius <= 0
    // means "no cap" (legacy behavior).
    float cam_x, float cam_y, float cam_z, float radius_sq,
    // GPU-side downsample gate. 0 = keep all points. 1..255 = drop
    // point if decimation_hash(world_voxel_coord) >= thresh. Filter
    // applied BEFORE lookup_or_insert so dropped points skip all
    // expensive work (probe/CAS/atomic accumulator updates).
    int decimation_thresh_256,
    // LOD finest-band radius in metres. When variable_res is on,
    // points within `lod_finest_radius` of the camera land at the
    // finest level (TSDF_HASH_N_LEVELS-1); each successive band of
    // radius doubles the voxel size. <= 0 disables LOD: every point
    // lands at finest level (legacy single-resolution behavior).
    float lod_finest_radius
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;
    float px = positions[tid * 3 + 0];
    float py = positions[tid * 3 + 1];
    float pz = positions[tid * 3 + 2];
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) return;

    float w = weights ? weights[tid] : 1.0f;
    if (!(w > 0.0f) || !isfinite(w)) return;

    if (radius_sq > 0.0f) {
        float dx = px - cam_x, dy = py - cam_y, dz = pz - cam_z;
        if (dx * dx + dy * dy + dz * dz > radius_sq) return;
    }

    // ── LOD level selection ──
    // Each LOD band doubles the voxel size: level N_LEVELS-1 (finest)
    // covers radius [0, R), level N_LEVELS-2 covers [R, 2R), level
    // N_LEVELS-3 covers [2R, 4R), etc. lod_finest_radius<=0 disables
    // LOD — every point lands at the finest level, byte-identical to
    // the legacy single-voxel-size code path.
    int level = TSDF_HASH_N_LEVELS - 1;
    float voxel_at_level = voxel_size_m;
    if (lod_finest_radius > 0.0f) {
        float dx = px - cam_x, dy = py - cam_y, dz = pz - cam_z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        // band index counted away from finest: 0 = within R, 1 = within 2R, ...
        // level = (N_LEVELS - 1) - band, clamped to [0, N_LEVELS-1].
        if (dist > lod_finest_radius) {
            float ratio = dist / lod_finest_radius;
            int band = (int)floorf(log2f(ratio));
            if (band < 0) band = 0;
            int lvl = (TSDF_HASH_N_LEVELS - 1) - band;
            if (lvl < 0) lvl = 0;
            level = lvl;
            voxel_at_level = voxel_size_m * (float)(1 << ((TSDF_HASH_N_LEVELS - 1) - level));
        }
    }

    float inv_vs = 1.0f / voxel_at_level;
    int world_vx = (int)floorf(px * inv_vs);
    int world_vy = (int)floorf(py * inv_vs);
    int world_vz = (int)floorf(pz * inv_vs);

    // Downsample gate at the integrate side — applied per world-voxel-
    // coord so the keep/drop decision is stable across chunks AND
    // consistent with the same hash used at extract/evict time. Drop
    // here = "this voxel never enters the TSDF": no block alloc, no
    // accumulator updates, no eviction churn. The TSDF itself is the
    // downsampler.
    if (decimation_thresh_256 > 0 && decimation_thresh_256 < 256) {
        if ((int)decimation_hash(world_vx, world_vy, world_vz) >= decimation_thresh_256) {
            return;
        }
    }

    int bx = floor_div_int(world_vx, BD);
    int by = floor_div_int(world_vy, BD);
    int bz = floor_div_int(world_vz, BD);

    int block_idx = lookup_or_insert_block(
        hash_table, hash_size, block_alloc_counter, pool_capacity,
        free_stack, free_count,
        block_alloc, block_coord,
        block_level,
        last_touched_chunk, current_chunk,
        bx, by, bz, level
    );
    if (block_idx < 0) {
        atomicAdd(n_dropped_pool_full, 1ull);
        return;  // pool full or hash table full — drop point
    }
    // No per-touch stamp — see lookup_or_insert_block (FIFO policy).

    int local_vx = world_vx - bx * BD;
    int local_vy = world_vy - by * BD;
    int local_vz = world_vz - bz * BD;
    int voxel_offset = (local_vz * BD + local_vy) * BD + local_vx;
    int idx = block_idx * BD3 + voxel_offset;

    if (weight_cap > 0.0f && weight_buf[idx] >= weight_cap) return;

    unsigned char r = 128, g = 128, b = 128;
    if (colors) {
        r = colors[tid * 3 + 0];
        g = colors[tid * 3 + 1];
        b = colors[tid * 3 + 2];
    }

    atomicAdd(&sum_x[idx], px * w);
    atomicAdd(&sum_y[idx], py * w);
    atomicAdd(&sum_z[idx], pz * w);
    atomicAdd(&sum_r[idx], (float)r * w);
    atomicAdd(&sum_g[idx], (float)g * w);
    atomicAdd(&sum_b[idx], (float)b * w);
    atomicAdd(&weight_buf[idx], w);
    atomicAdd(&count_buf[idx], 1u);
}

// ── Read-only block lookup (no insert) ──
//
// Mirrors lookup_or_insert_block's probe but never mutates the table.
// Used by the marching-cubes kernel to fetch corner voxels that fall in
// neighbouring blocks. MC runs after integrate + a stream sync, so the
// table is quiescent — plain (non-volatile) reads are safe. Returns the
// pool block_idx, or -1 if the block isn't present.
__device__ static int lookup_block_ro(
    const HashEntry* hash_table, int hash_size,
    const int* block_level,
    int bx, int by, int bz, int level
) {
    uint32_t start = hash_block_coord(bx, by, bz, level, hash_size);
    for (int probe = 0; probe < hash_size; ++probe) {
        uint32_t slot = (start + (uint32_t)probe) & (uint32_t)(hash_size - 1);
        const HashEntry* e = &hash_table[slot];
        int cur_bx = e->bx;
        if (cur_bx == HASH_EMPTY_KEY) return -1;     // end of probe chain
        if (cur_bx == HASH_TOMBSTONE) continue;
        if (cur_bx == bx) {
            int idx = e->block_idx;
            if (idx >= 0 && e->by == by && e->bz == bz && block_level[idx] == level)
                return idx;
        }
    }
    return -1;
}

// ── Projective-TSDF integrate kernel (mesh mode) ──
//
// One thread per input point. The point P is a surface sample observed
// from camera C; we splat a truncated signed distance along the view ray
// d = normalize(C - P). For samples V = P + s·d with s in [-trunc, +trunc]
// (stepped at voxel resolution), the signed distance of V to the surface
// along the ray is s — POSITIVE toward the camera (free space), NEGATIVE
// behind the surface (inside). We accumulate a running weighted mean of
// the normalised value s/trunc in tsdf_buf, plus colour, reusing the same
// weight/colour accumulators as the legacy binner. Each touched block is
// stamped dirty for incremental marching cubes.
__global__ void hash_integrate_tsdf_kernel(
    const float* __restrict__ positions,
    const unsigned char* __restrict__ colors,
    const float* __restrict__ weights,
    int n_points,
    float voxel_size_m,
    float weight_cap,
    float trunc_m,
    HashEntry* hash_table, int hash_size,
    int* block_alloc_counter, int pool_capacity,
    int* free_stack, int* free_count,
    int* block_alloc, int3* block_coord, int* block_level,
    int* last_touched_chunk, int* dirty_chunk, int current_chunk,
    float* tsdf_buf,
    float* sum_r, float* sum_g, float* sum_b,
    float* weight_buf, unsigned int* count_buf,
    unsigned long long* n_dropped_pool_full,
    float cam_x, float cam_y, float cam_z, float radius_sq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;
    float px = positions[tid*3+0], py = positions[tid*3+1], pz = positions[tid*3+2];
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) return;
    float w = weights ? weights[tid] : 1.0f;
    if (!(w > 0.0f) || !isfinite(w)) return;

    if (radius_sq > 0.0f) {
        float dx = px-cam_x, dy = py-cam_y, dz = pz-cam_z;
        if (dx*dx + dy*dy + dz*dz > radius_sq) return;
    }
    // View ray (surface → camera), normalised.
    float rx = cam_x - px, ry = cam_y - py, rz = cam_z - pz;
    float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
    if (!(rlen > 1e-6f)) return;
    float inv_rlen = 1.0f / rlen;
    rx *= inv_rlen; ry *= inv_rlen; rz *= inv_rlen;

    unsigned char cr = 128, cg = 128, cb = 128;
    if (colors) { cr = colors[tid*3+0]; cg = colors[tid*3+1]; cb = colors[tid*3+2]; }

    const int level = TSDF_HASH_N_LEVELS - 1;   // mesh mode: finest, no LOD
    float inv_vs = 1.0f / voxel_size_m;
    int nT = (int)floorf(trunc_m * inv_vs);
    if (nT < 1) nT = 1;
    int last_vx = 0x7fffffff, last_vy = 0x7fffffff, last_vz = 0x7fffffff;
    for (int i = -nT; i <= nT; ++i) {
        float s  = (float)i * voxel_size_m;       // signed distance along ray
        float wx = px + s*rx, wy = py + s*ry, wz = pz + s*rz;
        int world_vx = (int)floorf(wx * inv_vs);
        int world_vy = (int)floorf(wy * inv_vs);
        int world_vz = (int)floorf(wz * inv_vs);
        // Skip if this sample maps to the same voxel as the previous step.
        if (world_vx == last_vx && world_vy == last_vy && world_vz == last_vz) continue;
        last_vx = world_vx; last_vy = world_vy; last_vz = world_vz;

        int bx = floor_div_int(world_vx, BD);
        int by = floor_div_int(world_vy, BD);
        int bz = floor_div_int(world_vz, BD);
        int block_idx = lookup_or_insert_block(
            hash_table, hash_size, block_alloc_counter, pool_capacity,
            free_stack, free_count, block_alloc, block_coord, block_level,
            last_touched_chunk, current_chunk, bx, by, bz, level);
        if (block_idx < 0) { atomicAdd(n_dropped_pool_full, 1ull); continue; }

        int lvx = world_vx - bx*BD, lvy = world_vy - by*BD, lvz = world_vz - bz*BD;
        int idx = block_idx*BD3 + (lvz*BD + lvy)*BD + lvx;
        if (weight_cap > 0.0f && weight_buf[idx] >= weight_cap) continue;

        float sdf_n = s / trunc_m;                // normalised to [-1,1]
        if (sdf_n >  1.0f) sdf_n =  1.0f;
        if (sdf_n < -1.0f) sdf_n = -1.0f;
        atomicAdd(&tsdf_buf[idx], sdf_n * w);
        atomicAdd(&sum_r[idx], (float)cr * w);
        atomicAdd(&sum_g[idx], (float)cg * w);
        atomicAdd(&sum_b[idx], (float)cb * w);
        atomicAdd(&weight_buf[idx], w);
        atomicAdd(&count_buf[idx], 1u);
        dirty_chunk[block_idx] = current_chunk;   // benign same-value race
    }
}

// ── evict kernel ──
//
// One CUDA block per pool block. Threads in the block (BD3 = 512) each
// own one voxel. If the host requests eviction of blocks not touched
// since `evict_before_chunk`, this kernel:
//   1. Spills converged voxels (weight > 0) to the caller-provided
//      output buffer (host- or device-resident; we treat it as device
//      memory). Spilled voxels carry the running mean position and
//      mean color; they're the same shape as `tsdf_hash_extract_points`
//      output, so the host can append them to its persistent mesh
//      accumulator without further processing.
//   2. Zeroes the per-voxel SoA fields for the evicted block.
//   3. Marks the hash entry as TOMBSTONE and the block_alloc slot as
//      free, then pushes the block_idx onto the free-stack so future
//      lookup_or_insert calls can recycle it.
//
// Concurrency note: this kernel must NOT run concurrently with
// hash_integrate_kernel — the host serializes them via stream sync.
// Push/pop on free_stack is otherwise unsafe.
__global__ void hash_evict_kernel(
    int pool_capacity,
    int evict_before_chunk,
    int* block_alloc,
    int* last_touched_chunk,
    int3* block_coord,
    int* block_level,
    HashEntry* hash_table,
    int hash_size,
    float voxel_size_m,
    float* sum_x, float* sum_y, float* sum_z,
    float* sum_r, float* sum_g, float* sum_b,
    float* weight_buf, unsigned int* count_buf,
    int* free_stack, int* free_count,
    int* n_evicted_blocks,
    // Spill buffer for converged voxels of evicted blocks.
    float* out_xyz, unsigned char* out_rgb, unsigned int* out_cnt,
    int out_cap,
    // GPU-side downsample gate. 0 = keep all (no decimation). 1..255 =
    // keep iff decimation_hash(world_voxel_coord) < keep_thresh_256.
    int keep_thresh_256
) {
    int block_idx = blockIdx.x;
    if (block_idx >= pool_capacity) return;

    // Phase 1: decide if this block is evictable.
    __shared__ bool sh_evict;
    if (threadIdx.x == 0) {
        sh_evict = (block_alloc[block_idx] != 0) &&
                   (last_touched_chunk[block_idx] >= 0) &&
                   (last_touched_chunk[block_idx] < evict_before_chunk);
    }
    __syncthreads();
    if (!sh_evict) return;

    int base = block_idx * BD3;

    // Phase 2: per-voxel spill + zero. Each thread handles one voxel.
    if (threadIdx.x < BD3) {
        int idx = base + (int)threadIdx.x;
        float w = weight_buf[idx];
        if (w > 0.0f && isfinite(w)) {
            // Downsample gate: keep iff hash(world_voxel_coord) < thresh.
            // We need the world coord; reconstruct it from block_coord
            // + the per-voxel offset within the block.
            bool keep = true;
            if (keep_thresh_256 > 0 && keep_thresh_256 < 256) {
                int3 bc = block_coord[block_idx];
                int local_v = (int)threadIdx.x;
                int lx = local_v % BD;
                int ly = (local_v / BD) % BD;
                int lz = local_v / (BD * BD);
                int wvx = bc.x * BD + lx;
                int wvy = bc.y * BD + ly;
                int wvz = bc.z * BD + lz;
                if ((int)decimation_hash(wvx, wvy, wvz) >= keep_thresh_256) {
                    keep = false;
                }
            }
            if (keep) {
            unsigned int spill_pos = atomicAdd(out_cnt, 1u);
            if ((int)spill_pos < out_cap) {
                float inv_w = 1.0f / w;
                float mx = sum_x[idx] * inv_w;
                float my = sum_y[idx] * inv_w;
                float mz = sum_z[idx] * inv_w;
                out_xyz[spill_pos * 3 + 0] = mx;
                out_xyz[spill_pos * 3 + 1] = my;
                out_xyz[spill_pos * 3 + 2] = mz;
                float mr = sum_r[idx] * inv_w;
                float mg = sum_g[idx] * inv_w;
                float mb = sum_b[idx] * inv_w;
                if (mr < 0.0f) mr = 0.0f; if (mr > 255.0f) mr = 255.0f;
                if (mg < 0.0f) mg = 0.0f; if (mg > 255.0f) mg = 255.0f;
                if (mb < 0.0f) mb = 0.0f; if (mb > 255.0f) mb = 255.0f;
                out_rgb[spill_pos * 3 + 0] = (unsigned char)mr;
                out_rgb[spill_pos * 3 + 1] = (unsigned char)mg;
                out_rgb[spill_pos * 3 + 2] = (unsigned char)mb;
            }
            // If we overflowed out_cap, the voxel is dropped; the
            // host saw the count tick past out_cap and can resize.
            }  // close if(keep)
        }
        // Zero the per-voxel buffers regardless — the slot will be
        // recycled and must start clean.
        sum_x[idx] = 0.0f; sum_y[idx] = 0.0f; sum_z[idx] = 0.0f;
        sum_r[idx] = 0.0f; sum_g[idx] = 0.0f; sum_b[idx] = 0.0f;
        weight_buf[idx] = 0.0f;
        count_buf[idx] = 0u;
    }

    // Phase 3: bookkeeping (one thread per block).
    if (threadIdx.x == 0) {
        int3 c = block_coord[block_idx];
        int lvl = block_level[block_idx];
        // Locate the hash entry. We could re-hash the coord, but the
        // entry's block_idx field is the unique identifier — scan the
        // probe chain starting at the natural hash position.
        uint32_t start = hash_block_coord(c.x, c.y, c.z, lvl, hash_size);
        int probes_max = hash_size < 256 ? hash_size : 256;
        for (int i = 0; i < probes_max; ++i) {
            uint32_t slot = (start + (uint32_t)i) & (uint32_t)(hash_size - 1);
            HashEntry* e = &hash_table[slot];
            int eb = *((volatile int*)&e->bx);
            if (eb == HASH_EMPTY_KEY) break;            // chain ended without match
            if (eb != c.x) continue;                     // collision, keep going
            int by_v = *((volatile int*)&e->by);
            int bz_v = *((volatile int*)&e->bz);
            int idx_v = *((volatile int*)&e->block_idx);
            if (by_v == c.y && bz_v == c.z && idx_v == block_idx) {
                // Mark TOMBSTONE: subsequent probes skip this slot but
                // continue past it. Order: zero block_idx first so
                // any in-flight reader sees "not live", then store
                // tombstone in bx.
                e->block_idx = -1;
                __threadfence();
                e->bx = HASH_TOMBSTONE;
                __threadfence();
                break;
            }
        }
        // Free the pool slot and push to the recycle stack.
        block_alloc[block_idx] = 0;
        last_touched_chunk[block_idx] = -1;
        int pos = atomicAdd(free_count, 1);
        // pos < pool_capacity holds because we only push to free_stack
        // for blocks that were previously allocated (block_alloc==1)
        // and there are at most pool_capacity such blocks.
        free_stack[pos] = block_idx;

        atomicAdd(n_evicted_blocks, 1);
    }
}

// ── extract kernel ──
//
// One thread per voxel in the entire pool. Threads whose block isn't
// allocated bail. Surviving threads apply the gates and write the
// (mean position, mean color) tuple to the output buffer.
__global__ void hash_extract_kernel(
    int pool_capacity,
    float voxel_size_m,
    const int* __restrict__ block_alloc,
    const int3* __restrict__ block_coord,
    const float* __restrict__ sum_x, const float* __restrict__ sum_y, const float* __restrict__ sum_z,
    const float* __restrict__ sum_r, const float* __restrict__ sum_g, const float* __restrict__ sum_b,
    const float* __restrict__ weight_buf,
    const unsigned int* __restrict__ count_buf,
    float min_weight,
    int min_count,
    // output
    float* out_xyz,
    unsigned char* out_rgb,
    unsigned int* out_count_ptr,
    int out_cap,
    // GPU-side downsample gate. 0 = keep all (no decimation). 1..255 =
    // keep iff decimation_hash(world_voxel_coord) < keep_thresh_256.
    // Filter applied BEFORE the atomicAdd-claim of an output slot so
    // we don't waste output buffer slots on dropped voxels.
    int keep_thresh_256
) {
    long long total = (long long)pool_capacity * (long long)BD3;
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (tid >= total) return;

    int idx = (int)tid;
    int block_idx = idx / BD3;
    if (!block_alloc[block_idx]) return;

    float w = weight_buf[idx];
    unsigned int c = count_buf[idx];
    if (w < min_weight) return;
    if ((int)c < min_count) return;

    if (keep_thresh_256 > 0 && keep_thresh_256 < 256) {
        int3 bc = block_coord[block_idx];
        int local_v = idx - block_idx * BD3;
        int lx = local_v % BD;
        int ly = (local_v / BD) % BD;
        int lz = local_v / (BD * BD);
        int wvx = bc.x * BD + lx;
        int wvy = bc.y * BD + ly;
        int wvz = bc.z * BD + lz;
        if ((int)decimation_hash(wvx, wvy, wvz) >= keep_thresh_256) return;
    }

    float inv_w = 1.0f / w;
    float mx = sum_x[idx] * inv_w;
    float my = sum_y[idx] * inv_w;
    float mz = sum_z[idx] * inv_w;
    float cr_f = sum_r[idx] * inv_w;
    float cg_f = sum_g[idx] * inv_w;
    float cb_f = sum_b[idx] * inv_w;

    unsigned int slot = atomicAdd(out_count_ptr, 1u);
    if ((int)slot >= out_cap) return;

    out_xyz[slot * 3 + 0] = mx;
    out_xyz[slot * 3 + 1] = my;
    out_xyz[slot * 3 + 2] = mz;
    out_rgb[slot * 3 + 0] = (unsigned char)fminf(fmaxf(cr_f, 0.0f), 255.0f);
    out_rgb[slot * 3 + 1] = (unsigned char)fminf(fmaxf(cg_f, 0.0f), 255.0f);
    out_rgb[slot * 3 + 2] = (unsigned char)fminf(fmaxf(cb_f, 0.0f), 255.0f);
}

// ── Marching cubes tables (verbatim from Open3D MarchingCubesConst.h,
// MIT) — the SAME tables the offline ScalableTSDFVolume mesh used, so the
// live mesh matches it. Corner order: shift[8] =
// (0,0,0)(1,0,0)(1,1,0)(0,1,0)(0,0,1)(1,0,1)(1,1,1)(0,1,1). ──
__device__ static const int mc_edge_table[256] = {
        0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905,
        0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a,
        0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93,
        0xf99, 0xe90, 0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9,
        0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6,
        0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f,
        0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5,
        0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a,
        0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
        0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
        0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6,
        0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f,
        0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
        0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a,
        0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663,
        0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39,
        0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
        0x13a, 0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f,
        0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605,
        0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};

__device__ static const int mc_tri_table[256][16] = {
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
        {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
        {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
        {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
        {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
        {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
        {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
        {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
        {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
        {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
        {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
        {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
        {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
        {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
        {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
        {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
        {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
        {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
        {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
        {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
        {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
        {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
        {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
        {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
        {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
        {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
        {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
        {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
        {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
        {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

// Open3D edge → corner endpoints (matches mc_tri_table edge numbering).
__device__ static const int mc_edge_v0[12] = {0,1,3,0,4,5,7,4,0,1,2,3};
__device__ static const int mc_edge_v1[12] = {1,2,2,3,5,6,6,7,4,5,6,7};
// Corner offsets (Open3D shift[8]).
__device__ static const int mc_cx[8] = {0,1,1,0,0,1,1,0};
__device__ static const int mc_cy[8] = {0,0,1,1,0,0,1,1};
__device__ static const int mc_cz[8] = {0,0,0,0,1,1,1,1};

// ── Marching cubes kernel ──
//
// One CUDA block per pool block; BD³ threads, each owning one cell whose
// corner-0 is the local voxel (lx,ly,lz). Corners that fall in a
// neighbouring block are resolved via lookup_block_ro. A cell is skipped
// if any of its 8 corners is unobserved (weight < min_weight). Vertices
// are interpolated at the iso-crossing; colours are corner-lerped; the
// normal is the per-triangle geometric normal. Each emitted vertex is
// tagged with its owner block_idx so the client can replace per block.
// Compact the dirty block list: scan dirty_chunk[] / block_alloc[], and
// for each block tagged with current_chunk (and actually allocated),
// append its block_idx to out_dirty_list via atomicAdd. The marching-
// cubes kernel then launches with grid=*out_dirty_count instead of the
// full pool_capacity, cutting ~65 ms of empty-block launch overhead at
// pool_capacity=65k.
__global__ void hash_compact_dirty_kernel(
    int pool_capacity, int current_chunk,
    const int* __restrict__ block_alloc,
    const int* __restrict__ dirty_chunk,
    int* __restrict__ out_dirty_list,
    unsigned int* __restrict__ out_dirty_count
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= pool_capacity) return;
    if (!block_alloc[block_idx]) return;
    if (dirty_chunk[block_idx] != current_chunk) return;
    unsigned int slot = atomicAdd(out_dirty_count, 1u);
    if ((int)slot < pool_capacity) out_dirty_list[slot] = block_idx;
}

__global__ void hash_marching_cubes_kernel(
    int pool_capacity, float voxel_size_m, float iso,
    const int* __restrict__ block_alloc,
    const int3* __restrict__ block_coord,
    const int* __restrict__ block_level,
    const int* __restrict__ dirty_chunk,
    int dirty_only, int current_chunk,
    const HashEntry* __restrict__ hash_table, int hash_size,
    const float* __restrict__ tsdf_buf,
    const float* __restrict__ sum_r, const float* __restrict__ sum_g, const float* __restrict__ sum_b,
    const float* __restrict__ weight_buf,
    float min_weight,
    float* __restrict__ out_posnor,        // 6 floats/vert: pos3, nor3
    unsigned char* __restrict__ out_rgb,   // 3 u8/vert
    int* __restrict__ out_block,           // 1 int/vert: owner block_idx
    unsigned int* __restrict__ out_vcount_ptr,
    int out_vcap,
    // Optional compacted dirty list: when non-null, blockIdx.x indexes
    // into `dirty_list` for the real block index and we skip the
    // dirty/alloc checks (the compaction pre-pass already filtered).
    // When null, fall back to legacy behaviour (blockIdx.x = block_idx,
    // dirty_chunk[] check inline).
    const int* __restrict__ dirty_list
) {
    int block_idx;
    if (dirty_list) {
        // grid size = N_dirty; blockIdx.x is an index INTO dirty_list.
        block_idx = dirty_list[blockIdx.x];
    } else {
        block_idx = blockIdx.x;
        if (block_idx >= pool_capacity) return;
        if (!block_alloc[block_idx]) return;
        if (dirty_only && dirty_chunk[block_idx] != current_chunk) return;
    }

    int level = block_level[block_idx];
    int3 bc = block_coord[block_idx];

    int cell = threadIdx.x;             // 0 .. BD3-1
    int lx = cell % BD;
    int ly = (cell / BD) % BD;
    int lz = cell / (BD * BD);

    float cval[8], cpx[8], cpy[8], cpz[8], ccr[8], ccg[8], ccb[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int wvx = bc.x*BD + lx + mc_cx[c];
        int wvy = bc.y*BD + ly + mc_cy[c];
        int wvz = bc.z*BD + lz + mc_cz[c];
        int nbx = floor_div_int(wvx, BD);
        int nby = floor_div_int(wvy, BD);
        int nbz = floor_div_int(wvz, BD);
        int bidx;
        if (nbx == bc.x && nby == bc.y && nbz == bc.z) bidx = block_idx;
        else bidx = lookup_block_ro(hash_table, hash_size, block_level, nbx, nby, nbz, level);
        if (bidx < 0) return;           // neighbour block absent → skip cell
        int llx = wvx - nbx*BD, lly = wvy - nby*BD, llz = wvz - nbz*BD;
        int vidx = bidx*BD3 + (llz*BD + lly)*BD + llx;
        float wv = weight_buf[vidx];
        if (wv < min_weight) return;    // unobserved corner → skip cell
        float invw = 1.0f / wv;
        cval[c] = tsdf_buf[vidx] * invw;
        cpx[c] = ((float)wvx + 0.5f) * voxel_size_m;
        cpy[c] = ((float)wvy + 0.5f) * voxel_size_m;
        cpz[c] = ((float)wvz + 0.5f) * voxel_size_m;
        ccr[c] = sum_r[vidx] * invw;
        ccg[c] = sum_g[vidx] * invw;
        ccb[c] = sum_b[vidx] * invw;
    }

    int ci = 0;
    #pragma unroll
    for (int c = 0; c < 8; ++c) if (cval[c] < iso) ci |= (1 << c);
    int edges = mc_edge_table[ci];
    if (edges == 0) return;

    float vpx[12], vpy[12], vpz[12], vcr[12], vcg[12], vcb[12];
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        if (!(edges & (1 << e))) continue;
        int a = mc_edge_v0[e], b = mc_edge_v1[e];
        float da = cval[a], db = cval[b];
        float t = (fabsf(db - da) > 1e-12f) ? (iso - da) / (db - da) : 0.5f;
        vpx[e] = cpx[a] + t*(cpx[b]-cpx[a]);
        vpy[e] = cpy[a] + t*(cpy[b]-cpy[a]);
        vpz[e] = cpz[a] + t*(cpz[b]-cpz[a]);
        vcr[e] = ccr[a] + t*(ccr[b]-ccr[a]);
        vcg[e] = ccg[a] + t*(ccg[b]-ccg[a]);
        vcb[e] = ccb[a] + t*(ccb[b]-ccb[a]);
    }

    const int* tri = mc_tri_table[ci];
    for (int t = 0; tri[t] != -1; t += 3) {
        int e0 = tri[t], e1 = tri[t+1], e2 = tri[t+2];
        float ux = vpx[e1]-vpx[e0], uy = vpy[e1]-vpy[e0], uz = vpz[e1]-vpz[e0];
        float wx = vpx[e2]-vpx[e0], wy = vpy[e2]-vpy[e0], wz = vpz[e2]-vpz[e0];
        float nx = uy*wz - uz*wy, ny = uz*wx - ux*wz, nz = ux*wy - uy*wx;
        float nl = sqrtf(nx*nx + ny*ny + nz*nz);
        if (nl > 1e-12f) { nx/=nl; ny/=nl; nz/=nl; }
        unsigned int base = atomicAdd(out_vcount_ptr, 3u);
        if ((int)base + 3 > out_vcap) return;
        int ee[3] = {e0, e1, e2};
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            unsigned int vi = base + k;
            int e = ee[k];
            out_posnor[vi*6+0] = vpx[e]; out_posnor[vi*6+1] = vpy[e]; out_posnor[vi*6+2] = vpz[e];
            out_posnor[vi*6+3] = nx;     out_posnor[vi*6+4] = ny;     out_posnor[vi*6+5] = nz;
            out_rgb[vi*3+0] = (unsigned char)fminf(fmaxf(vcr[e], 0.0f), 255.0f);
            out_rgb[vi*3+1] = (unsigned char)fminf(fmaxf(vcg[e], 0.0f), 255.0f);
            out_rgb[vi*3+2] = (unsigned char)fminf(fmaxf(vcb[e], 0.0f), 255.0f);
            out_block[vi] = block_idx;
        }
    }
}

// ── Initialise hash table to all-empty ──
__global__ void init_hash_table_kernel(HashEntry* table, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    table[tid].bx = HASH_EMPTY_KEY;
    table[tid].by = 0;
    table[tid].bz = 0;
    table[tid].block_idx = -1;
}

// ── Helpers ──

static int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

static int alloc_pool_buffers(TsdfHash* h) {
    size_t voxel_count = (size_t)h->pool_capacity * (size_t)BD3;
    size_t bytes_f = voxel_count * sizeof(float);
    size_t bytes_u = voxel_count * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&h->d_sum_x, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_sum_y, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_sum_z, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_sum_r, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_sum_g, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_sum_b, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_weight, bytes_f));
    CUDA_CHECK(cudaMalloc(&h->d_count,  bytes_u));
    CUDA_CHECK(cudaMemset(h->d_sum_x, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_sum_y, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_sum_z, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_sum_r, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_sum_g, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_sum_b, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_weight, 0, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_count,  0, bytes_u));

    // Mesh-mode signed-distance field (per voxel). Always allocated so
    // mesh mode can be toggled at runtime without realloc; the memory
    // (one float per voxel) is modest next to the existing 7 SoA fields.
    CUDA_CHECK(cudaMalloc(&h->d_tsdf, bytes_f));
    CUDA_CHECK(cudaMemset(h->d_tsdf, 0, bytes_f));

    CUDA_CHECK(cudaMalloc(&h->d_block_alloc, (size_t)h->pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_block_alloc, 0, (size_t)h->pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_block_coord, (size_t)h->pool_capacity * sizeof(int3)));

    CUDA_CHECK(cudaMalloc(&h->d_block_alloc_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_block_alloc_counter, 0, sizeof(int)));

    // Free-list: stack of recycled block_idx values populated by
    // eviction. Capacity == pool_capacity (every block could in theory
    // be evicted at once).
    CUDA_CHECK(cudaMalloc(&h->d_free_stack, (size_t)h->pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_free_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_free_count, 0, sizeof(int)));

    // Per-block last-touched-chunk timestamp. -1 means "never touched"
    // — also serves as the initial value for slots that have been
    // freed by eviction.
    CUDA_CHECK(cudaMalloc(&h->d_last_touched_chunk, (size_t)h->pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_last_touched_chunk, 0xFF, (size_t)h->pool_capacity * sizeof(int)));

    // Per-block "dirty this chunk" stamp for incremental marching cubes.
    // 0xFF memset → -1 (never dirty) initial.
    CUDA_CHECK(cudaMalloc(&h->d_dirty_chunk, (size_t)h->pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_dirty_chunk, 0xFF, (size_t)h->pool_capacity * sizeof(int)));

    // Per-block LOD level. Default: every block is at finest level —
    // matches single-resolution behavior when variable-res is disabled.
    // Block-allocation paths (lookup_or_insert) overwrite this with the
    // level chosen for each new alloc; eviction zeroes it back to
    // finest as a defensive cleanup but a freed block_idx that's
    // recycled gets re-stamped on the next lookup_or_insert anyway.
    CUDA_CHECK(cudaMalloc(&h->d_block_level, (size_t)h->pool_capacity * sizeof(int)));
    {
        // memset can't write a constant non-zero int directly, but we
        // can rely on the integrate kernel always overwriting before
        // read. Initialise to 0 here for a clean default.
        CUDA_CHECK(cudaMemset(h->d_block_level, 0, (size_t)h->pool_capacity * sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&h->d_n_dropped_pool_full, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(h->d_n_dropped_pool_full, 0, sizeof(unsigned long long)));

    return 0;
}

static int alloc_hash_table(TsdfHash* h) {
    h->hash_size = next_pow2(h->pool_capacity * 2);
    CUDA_CHECK(cudaMalloc(&h->d_hash_table, (size_t)h->hash_size * sizeof(HashEntry)));
    int threads = 256;
    int blocks = (h->hash_size + threads - 1) / threads;
    init_hash_table_kernel<<<blocks, threads, 0, h->stream>>>(h->d_hash_table, h->hash_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    cudaStreamSynchronize(h->stream);

    CUDA_CHECK(cudaMalloc(&h->d_hash_size_dev, sizeof(int)));
    CUDA_CHECK(cudaMemset(h->d_hash_size_dev, 0, sizeof(int)));
    return 0;
}

// ── Public API ──

extern "C" TsdfHash* tsdf_hash_create(
    float voxel_size_m,
    int pool_capacity_blocks,
    int committed_cap_points
) {
    if (voxel_size_m <= 0.0f || pool_capacity_blocks <= 0 || committed_cap_points <= 0) {
        return nullptr;
    }
    TsdfHash* h = new TsdfHash();
    h->voxel_size_m = voxel_size_m;
    h->pool_capacity = pool_capacity_blocks;
    h->committed_cap = committed_cap_points;
    h->tmp_capacity = 0;
    h->d_tmp_positions = nullptr;
    h->d_tmp_colors = nullptr;
    h->d_tmp_weights = nullptr;
    h->weight_cap = 0.0f;
    h->decimation_thresh_256 = 0;
    h->lod_finest_radius_m = 0.0f;
    h->d_tsdf = nullptr;
    h->d_dirty_chunk = nullptr;
    h->mesh_mode = 0;
    h->trunc_m = 0.0f;
    h->d_mesh_posnor = nullptr;
    h->d_mesh_rgb = nullptr;
    h->d_mesh_block = nullptr;
    h->d_mesh_count = nullptr;
    h->mesh_scratch_vert_cap = 0;
    h->d_dirty_list = nullptr;
    h->d_dirty_count = nullptr;

    // High-priority non-blocking stream. The hardware scheduler
    // preempts work on lower-priority streams (notably PyTorch's
    // default-priority default stream in the sidecar process) when
    // this stream has kernels ready to run. Combined with
    // cudaStreamNonBlocking — kernels on this stream don't implicitly
    // sync against work on the default stream of THIS process — TSDF
    // work cuts in front of anything else competing for the SMs.
    //
    // The priority range queried with cudaDeviceGetStreamPriorityRange
    // is [low, high] where smaller integer = higher priority on every
    // NVIDIA architecture from Pascal onward. Fall back to default-
    // flags create if the query or create call fails (older drivers).
    int prio_low = 0, prio_high = 0;
    cudaError_t prio_err = cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
    cudaError_t create_err = cudaErrorUnknown;
    if (prio_err == cudaSuccess) {
        create_err = cudaStreamCreateWithPriority(
            &h->stream, cudaStreamNonBlocking, prio_high);
    }
    if (create_err != cudaSuccess) {
        // Fallback: non-blocking but default priority. Same as before
        // this commit; preserves behaviour on drivers that reject
        // the priority API.
        if (cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking)
            != cudaSuccess)
        {
            delete h;
            return nullptr;
        }
    }

    if (alloc_pool_buffers(h) != 0) {
        cudaStreamDestroy(h->stream);
        delete h;
        return nullptr;
    }
    if (alloc_hash_table(h) != 0) {
        cudaStreamDestroy(h->stream);
        delete h;
        return nullptr;
    }

    CUDA_CHECK_NULL(cudaMalloc(&h->d_committed_xyz,
                                (size_t)committed_cap_points * 3 * sizeof(float)));
    CUDA_CHECK_NULL(cudaMalloc(&h->d_committed_rgb,
                                (size_t)committed_cap_points * 3 * sizeof(unsigned char)));
    CUDA_CHECK_NULL(cudaMalloc(&h->d_committed_count_ptr, sizeof(unsigned int)));
    CUDA_CHECK_NULL(cudaMemset(h->d_committed_count_ptr, 0, sizeof(unsigned int)));

    return h;
}

extern "C" void tsdf_hash_destroy(TsdfHash* h) {
    if (!h) return;
    cudaStreamSynchronize(h->stream);
    cudaFree(h->d_sum_x); cudaFree(h->d_sum_y); cudaFree(h->d_sum_z);
    cudaFree(h->d_sum_r); cudaFree(h->d_sum_g); cudaFree(h->d_sum_b);
    cudaFree(h->d_weight); cudaFree(h->d_count);
    cudaFree(h->d_tsdf);
    cudaFree(h->d_dirty_chunk);
    if (h->d_mesh_posnor) cudaFree(h->d_mesh_posnor);
    if (h->d_mesh_rgb)    cudaFree(h->d_mesh_rgb);
    if (h->d_mesh_block)  cudaFree(h->d_mesh_block);
    if (h->d_mesh_count)  cudaFree(h->d_mesh_count);
    if (h->d_dirty_list)  cudaFree(h->d_dirty_list);
    if (h->d_dirty_count) cudaFree(h->d_dirty_count);
    cudaFree(h->d_block_alloc);
    cudaFree(h->d_block_coord);
    cudaFree(h->d_block_alloc_counter);
    cudaFree(h->d_free_stack);
    cudaFree(h->d_free_count);
    cudaFree(h->d_last_touched_chunk);
    cudaFree(h->d_block_level);
    cudaFree(h->d_n_dropped_pool_full);
    cudaFree(h->d_hash_table);
    cudaFree(h->d_hash_size_dev);
    cudaFree(h->d_committed_xyz);
    cudaFree(h->d_committed_rgb);
    cudaFree(h->d_committed_count_ptr);
    if (h->d_tmp_positions) cudaFree(h->d_tmp_positions);
    if (h->d_tmp_colors)    cudaFree(h->d_tmp_colors);
    if (h->d_tmp_weights)   cudaFree(h->d_tmp_weights);
    cudaStreamDestroy(h->stream);
    delete h;
}

static int ensure_tmp_capacity(TsdfHash* h, int n_points) {
    if (n_points <= h->tmp_capacity) return 0;
    int new_cap = n_points + n_points / 2;
    if (h->d_tmp_positions) cudaFree(h->d_tmp_positions);
    if (h->d_tmp_colors)    cudaFree(h->d_tmp_colors);
    if (h->d_tmp_weights)   cudaFree(h->d_tmp_weights);
    h->d_tmp_positions = nullptr;
    h->d_tmp_colors = nullptr;
    h->d_tmp_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&h->d_tmp_positions, (size_t)new_cap * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h->d_tmp_colors,    (size_t)new_cap * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&h->d_tmp_weights,   (size_t)new_cap * sizeof(float)));
    h->tmp_capacity = new_cap;
    return 0;
}

extern "C" int tsdf_hash_add_points(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points
) {
    // Legacy entry: chunk_id = 0 means "all blocks share the same
    // timestamp", which collapses LRU to no-op (every block is
    // equally recent). New callers should use the chunked API.
    return tsdf_hash_add_points_chunk(h, positions, colors, weights, n_points,
                                       /*chunk_id*/ 0,
                                       0.0f, 0.0f, 0.0f, 0.0f);
}

extern "C" int tsdf_hash_add_points_radius(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points,
    float cam_x, float cam_y, float cam_z, float radius_m
) {
    return tsdf_hash_add_points_chunk(h, positions, colors, weights, n_points,
                                      /*chunk_id*/ 0,
                                      cam_x, cam_y, cam_z, radius_m);
}

extern "C" int tsdf_hash_add_points_chunk(
    TsdfHash* h,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points,
    int chunk_id,
    float cam_x, float cam_y, float cam_z, float radius_m
) {
    if (!h || !positions || n_points <= 0) return 1;
    if (ensure_tmp_capacity(h, n_points) != 0) return 2;

    CUDA_CHECK(cudaMemcpyAsync(h->d_tmp_positions, positions,
                          (size_t)n_points * 3 * sizeof(float),
                          cudaMemcpyHostToDevice, h->stream));
    if (colors) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_tmp_colors, colors,
                              (size_t)n_points * 3 * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, h->stream));
    }
    if (weights) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_tmp_weights, weights,
                              (size_t)n_points * sizeof(float),
                              cudaMemcpyHostToDevice, h->stream));
    }

    float radius_sq = (radius_m > 0.0f) ? (radius_m * radius_m) : 0.0f;
    int threads = 256;
    int blocks = (n_points + threads - 1) / threads;
    if (h->mesh_mode) {
        // Projective-TSDF integration. trunc defaults to 4 voxels.
        float trunc_m = (h->trunc_m > 0.0f) ? h->trunc_m : (4.0f * h->voxel_size_m);
        hash_integrate_tsdf_kernel<<<blocks, threads, 0, h->stream>>>(
            h->d_tmp_positions,
            colors  ? h->d_tmp_colors  : nullptr,
            weights ? h->d_tmp_weights : nullptr,
            n_points,
            h->voxel_size_m,
            h->weight_cap,
            trunc_m,
            h->d_hash_table, h->hash_size,
            h->d_block_alloc_counter, h->pool_capacity,
            h->d_free_stack, h->d_free_count,
            h->d_block_alloc, h->d_block_coord, h->d_block_level,
            h->d_last_touched_chunk, h->d_dirty_chunk, chunk_id,
            h->d_tsdf,
            h->d_sum_r, h->d_sum_g, h->d_sum_b,
            h->d_weight, h->d_count,
            h->d_n_dropped_pool_full,
            cam_x, cam_y, cam_z, radius_sq
        );
        CUDA_CHECK(cudaGetLastError());
        return 0;
    }
    hash_integrate_kernel<<<blocks, threads, 0, h->stream>>>(
        h->d_tmp_positions,
        colors  ? h->d_tmp_colors  : nullptr,
        weights ? h->d_tmp_weights : nullptr,
        n_points,
        h->voxel_size_m,
        h->weight_cap,
        h->d_hash_table, h->hash_size,
        h->d_block_alloc_counter, h->pool_capacity,
        h->d_free_stack, h->d_free_count,
        h->d_block_alloc, h->d_block_coord,
        h->d_block_level,
        h->d_last_touched_chunk, chunk_id,
        h->d_sum_x, h->d_sum_y, h->d_sum_z,
        h->d_sum_r, h->d_sum_g, h->d_sum_b,
        h->d_weight, h->d_count,
        h->d_n_dropped_pool_full,
        cam_x, cam_y, cam_z, radius_sq,
        h->decimation_thresh_256,
        h->lod_finest_radius_m
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ── Device-pointer hand-off variant ──
//
// Skips the H2D upload step in the host-path entry above. Caller
// supplies device pointers that are already populated (typical case:
// backproject's `out_world_dev` feeding straight into TSDF integrate).
// Eliminates a per-chunk H2D of ~2-3 MB plus the host-side Vec
// build / flat_map serialisation that backed the host-pointer path.
//
// All other behavior — decimation gate, LOD selection, FIFO stamp,
// dropped-point counter — is identical to the host-path entry.
//
// d_colors and d_weights may be 0 (NULL) for the same neutral-grey /
// uniform-1.0 fallbacks the host path supports.
extern "C" int tsdf_hash_add_points_chunk_device(
    TsdfHash* h,
    uint64_t d_positions,
    uint64_t d_colors,
    uint64_t d_weights,
    int n_points,
    int chunk_id,
    float cam_x, float cam_y, float cam_z, float radius_m
) {
    if (!h || d_positions == 0 || n_points <= 0) return 1;

    float radius_sq = (radius_m > 0.0f) ? (radius_m * radius_m) : 0.0f;
    int threads = 256;
    int blocks = (n_points + threads - 1) / threads;
    hash_integrate_kernel<<<blocks, threads, 0, h->stream>>>(
        reinterpret_cast<const float*>(d_positions),
        d_colors  ? reinterpret_cast<const unsigned char*>(d_colors)  : nullptr,
        d_weights ? reinterpret_cast<const float*>(d_weights) : nullptr,
        n_points,
        h->voxel_size_m,
        h->weight_cap,
        h->d_hash_table, h->hash_size,
        h->d_block_alloc_counter, h->pool_capacity,
        h->d_free_stack, h->d_free_count,
        h->d_block_alloc, h->d_block_coord,
        h->d_block_level,
        h->d_last_touched_chunk, chunk_id,
        h->d_sum_x, h->d_sum_y, h->d_sum_z,
        h->d_sum_r, h->d_sum_g, h->d_sum_b,
        h->d_weight, h->d_count,
        h->d_n_dropped_pool_full,
        cam_x, cam_y, cam_z, radius_sq,
        h->decimation_thresh_256,
        h->lod_finest_radius_m
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int tsdf_hash_extract_points(
    TsdfHash* h,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float max_spread_frac,
    int drain_committed
) {
    return tsdf_hash_extract_points_target(
        h, buffer_xyz, buffer_rgb, buffer_cap,
        min_weight, min_count, max_spread_frac, drain_committed,
        /*keep_thresh_256*/ 0);
}

extern "C" int tsdf_hash_extract_points_target(
    TsdfHash* h,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float /*max_spread_frac*/,
    int /*drain_committed*/,
    int keep_thresh_256
) {
    if (!h || !buffer_xyz || !buffer_rgb || buffer_cap <= 0) return -1;

    // Allocate a device output scratch.
    float*         d_out_xyz = nullptr;
    unsigned char* d_out_rgb = nullptr;
    unsigned int*  d_out_cnt = nullptr;
    if (cudaMalloc(&d_out_xyz, (size_t)buffer_cap * 3 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_out_rgb, (size_t)buffer_cap * 3 * sizeof(unsigned char)) != cudaSuccess) {
        cudaFree(d_out_xyz); return -1;
    }
    if (cudaMalloc(&d_out_cnt, sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(d_out_xyz); cudaFree(d_out_rgb); return -1;
    }
    cudaMemsetAsync(d_out_cnt, 0, sizeof(unsigned int), h->stream);

    long long total = (long long)h->pool_capacity * (long long)BD3;
    long long blocks = (total + 255) / 256;
    if (blocks > 2147483000LL) blocks = 2147483000LL;  // safety cap on grid x

    hash_extract_kernel<<<(int)blocks, 256, 0, h->stream>>>(
        h->pool_capacity, h->voxel_size_m,
        h->d_block_alloc, h->d_block_coord,
        h->d_sum_x, h->d_sum_y, h->d_sum_z,
        h->d_sum_r, h->d_sum_g, h->d_sum_b,
        h->d_weight, h->d_count,
        min_weight, min_count,
        d_out_xyz, d_out_rgb, d_out_cnt, buffer_cap,
        keep_thresh_256
    );
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[tsdf_hash.cu] extract kernel launch error: %s\n", cudaGetErrorString(kerr));
        cudaFree(d_out_xyz); cudaFree(d_out_rgb); cudaFree(d_out_cnt);
        return -1;
    }
    cudaStreamSynchronize(h->stream);

    unsigned int final_count = 0;
    cudaMemcpyAsync(&final_count, d_out_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost, h->stream);
    cudaStreamSynchronize(h->stream);
    int overflow = 0;
    if ((int)final_count > buffer_cap) {
        overflow = 1;
        final_count = (unsigned int)buffer_cap;
    }
    cudaMemcpyAsync(buffer_xyz, d_out_xyz,
               (size_t)final_count * 3 * sizeof(float), cudaMemcpyDeviceToHost, h->stream);
    cudaMemcpyAsync(buffer_rgb, d_out_rgb,
               (size_t)final_count * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost, h->stream);
    cudaStreamSynchronize(h->stream);

    cudaFree(d_out_xyz);
    cudaFree(d_out_rgb);
    cudaFree(d_out_cnt);

    return overflow ? -1 : (int)final_count;
}

// ── Extract a triangle mesh from the TSDF (mesh mode) ──
//
// Runs marching cubes over the pool. With dirty_only != 0, only blocks
// touched in `current_chunk` are re-meshed (incremental). Host output
// buffers must hold `vert_cap` vertices:
//   buffer_posnor : vert_cap * 6 floats  (pos3, nor3)
//   buffer_rgb    : vert_cap * 3 u8
//   buffer_block  : vert_cap   int        (owner block_idx, for replace)
// Vertices are emitted as a triangle soup (3 consecutive = 1 triangle).
// Returns the vertex count, or -1 on overflow / error.
extern "C" int tsdf_hash_extract_mesh(
    TsdfHash* h,
    float* buffer_posnor,
    unsigned char* buffer_rgb,
    int* buffer_block,
    int vert_cap,
    float min_weight,
    float iso,
    int dirty_only,
    int current_chunk
) {
    if (!h || !buffer_posnor || !buffer_rgb || !buffer_block || vert_cap <= 0) return -1;

    // Pre-allocate (or grow) the per-extract scratch buffers ONCE in the
    // TsdfHash and reuse across calls. The previous code cudaMalloc'd four
    // buffers per call which on a busy GPU took ~50-200 ms — the dominant
    // contributor to the ~600 ms per-chunk marching-cubes wall time.
    // Grow strategy: if the caller passed a larger vert_cap than what's
    // already allocated, free + re-alloc. In practice vert_cap is fixed
    // for a session (env-driven), so this fires exactly once.
    if (h->mesh_scratch_vert_cap < vert_cap) {
        if (h->d_mesh_posnor) { cudaFree(h->d_mesh_posnor); h->d_mesh_posnor = nullptr; }
        if (h->d_mesh_rgb)    { cudaFree(h->d_mesh_rgb);    h->d_mesh_rgb    = nullptr; }
        if (h->d_mesh_block)  { cudaFree(h->d_mesh_block);  h->d_mesh_block  = nullptr; }
        if (h->d_mesh_count)  { cudaFree(h->d_mesh_count);  h->d_mesh_count  = nullptr; }
        if (cudaMalloc(&h->d_mesh_posnor, (size_t)vert_cap * 6 * sizeof(float)) != cudaSuccess)
            return -1;
        if (cudaMalloc(&h->d_mesh_rgb, (size_t)vert_cap * 3 * sizeof(unsigned char)) != cudaSuccess) {
            cudaFree(h->d_mesh_posnor); h->d_mesh_posnor = nullptr; return -1;
        }
        if (cudaMalloc(&h->d_mesh_block, (size_t)vert_cap * sizeof(int)) != cudaSuccess) {
            cudaFree(h->d_mesh_posnor); cudaFree(h->d_mesh_rgb);
            h->d_mesh_posnor = nullptr; h->d_mesh_rgb = nullptr; return -1;
        }
        if (cudaMalloc(&h->d_mesh_count, sizeof(unsigned int)) != cudaSuccess) {
            cudaFree(h->d_mesh_posnor); cudaFree(h->d_mesh_rgb); cudaFree(h->d_mesh_block);
            h->d_mesh_posnor = nullptr; h->d_mesh_rgb = nullptr; h->d_mesh_block = nullptr;
            return -1;
        }
        h->mesh_scratch_vert_cap = vert_cap;
    }
    cudaMemsetAsync(h->d_mesh_count, 0, sizeof(unsigned int), h->stream);

    // Dirty-block compaction (only when dirty_only is requested): a
    // cheap atomic-scan kernel writes block indices tagged with
    // current_chunk into d_dirty_list and stores the count in
    // d_dirty_count. The MC kernel then launches with grid=N_dirty
    // instead of pool_capacity (~65k → typically ~300), cutting ~65 ms
    // of empty-block launch overhead per call. Scratch is lazy-allocated.
    const int* d_dirty_list_param = nullptr;
    int dirty_launch_n = h->pool_capacity;
    if (dirty_only) {
        if (!h->d_dirty_list) {
            if (cudaMalloc(&h->d_dirty_list, (size_t)h->pool_capacity * sizeof(int)) != cudaSuccess)
                return -1;
        }
        if (!h->d_dirty_count) {
            if (cudaMalloc(&h->d_dirty_count, sizeof(unsigned int)) != cudaSuccess)
                return -1;
        }
        cudaMemsetAsync(h->d_dirty_count, 0, sizeof(unsigned int), h->stream);
        {
            int threads = 256;
            int blocks_ = (h->pool_capacity + threads - 1) / threads;
            hash_compact_dirty_kernel<<<blocks_, threads, 0, h->stream>>>(
                h->pool_capacity, current_chunk,
                h->d_block_alloc, h->d_dirty_chunk,
                h->d_dirty_list, h->d_dirty_count
            );
            cudaError_t cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "[tsdf_hash.cu] compact-dirty launch error: %s\n",
                        cudaGetErrorString(cerr));
                return -1;
            }
        }
        unsigned int n_dirty_u = 0;
        cudaMemcpyAsync(&n_dirty_u, h->d_dirty_count, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, h->stream);
        cudaStreamSynchronize(h->stream);
        if (n_dirty_u == 0) return 0;   // nothing dirty → empty mesh
        if ((int)n_dirty_u > h->pool_capacity) n_dirty_u = (unsigned int)h->pool_capacity;
        dirty_launch_n = (int)n_dirty_u;
        d_dirty_list_param = h->d_dirty_list;
    }

    hash_marching_cubes_kernel<<<dirty_launch_n, BD3, 0, h->stream>>>(
        h->pool_capacity, h->voxel_size_m, iso,
        h->d_block_alloc, h->d_block_coord, h->d_block_level,
        h->d_dirty_chunk, dirty_only, current_chunk,
        h->d_hash_table, h->hash_size,
        h->d_tsdf, h->d_sum_r, h->d_sum_g, h->d_sum_b, h->d_weight,
        min_weight,
        h->d_mesh_posnor, h->d_mesh_rgb, h->d_mesh_block, h->d_mesh_count, vert_cap,
        d_dirty_list_param
    );
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[tsdf_hash.cu] marching cubes launch error: %s\n", cudaGetErrorString(kerr));
        return -1;
    }
    // Single stream sync — kernel + memcpy share one sync point. The
    // previous code synced twice (after kernel, after count memcpy).
    cudaStreamSynchronize(h->stream);

    unsigned int final_count = 0;
    cudaMemcpyAsync(&final_count, h->d_mesh_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, h->stream);
    cudaStreamSynchronize(h->stream);
    int overflow = 0;
    if ((int)final_count > vert_cap) { overflow = 1; final_count = (unsigned int)vert_cap; }
    cudaMemcpyAsync(buffer_posnor, h->d_mesh_posnor,
                    (size_t)final_count * 6 * sizeof(float), cudaMemcpyDeviceToHost, h->stream);
    cudaMemcpyAsync(buffer_rgb, h->d_mesh_rgb,
                    (size_t)final_count * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost, h->stream);
    cudaMemcpyAsync(buffer_block, h->d_mesh_block,
                    (size_t)final_count * sizeof(int), cudaMemcpyDeviceToHost, h->stream);
    cudaStreamSynchronize(h->stream);

    // Scratch buffers (h->d_mesh_*) intentionally NOT freed here —
    // they're reused across calls (see pre-allocation block at the top
    // of this function). Freed in tsdf_hash_destroy.
    return overflow ? -1 : (int)final_count;
}

// Enable/disable mesh mode (projective TSDF + marching cubes). trunc_m
// is the truncation half-width in metres; <=0 → 4×voxel default. Toggle
// before integrating; switching mid-stream mixes accumulators so the
// caller should reset the grid first.
extern "C" int tsdf_hash_set_mesh_mode(TsdfHash* h, int on, float trunc_m) {
    if (!h) return -1;
    h->mesh_mode = on ? 1 : 0;
    h->trunc_m = (trunc_m > 0.0f) ? trunc_m : 0.0f;
    return 0;
}

extern "C" int tsdf_hash_block_count(TsdfHash* h) {
    if (!h) return -1;
    int c = 0;
    if (cudaMemcpyAsync(&c, h->d_block_alloc_counter, sizeof(int),
                        cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(h->stream) != cudaSuccess) return -1;
    if (c > h->pool_capacity) c = h->pool_capacity;
    return c;
}

extern "C" int tsdf_hash_committed_count(TsdfHash* h) {
    if (!h) return -1;
    unsigned int c = 0;
    if (cudaMemcpyAsync(&c, h->d_committed_count_ptr, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(h->stream) != cudaSuccess) return -1;
    return (int)c;
}

extern "C" int tsdf_hash_set_weight_cap(TsdfHash* h, float cap) {
    if (!h) return -1;
    h->weight_cap = cap;
    return 0;
}

extern "C" int tsdf_hash_set_decimation_thresh(TsdfHash* h, int thresh_256) {
    if (!h) return -1;
    if (thresh_256 < 0) thresh_256 = 0;
    if (thresh_256 > 256) thresh_256 = 256;
    h->decimation_thresh_256 = thresh_256;
    return 0;
}

// Enable variable-resolution (LOD) integration. Sets the radius of the
// finest LOD band — points within this distance from the camera get
// integrated at the finest level (voxel_size_m); each successive band
// of doubled radius drops a level (voxel doubles up to N_LEVELS bands).
//
//   radius_m = 0   : LOD off, every point at finest level (legacy)
//   radius_m > 0   : LOD on
//
// Stable across chunks; safe to set/clear mid-mission. Returns 0 on
// success, nonzero on null handle.
extern "C" int tsdf_hash_set_lod_finest_radius(TsdfHash* h, float radius_m) {
    if (!h) return -1;
    if (radius_m < 0.0f) radius_m = 0.0f;
    h->lod_finest_radius_m = radius_m;
    return 0;
}

// Returns the number of blocks that are currently live in the pool —
// i.e. high-water mark minus the recycle-stack depth. Equals the
// number of slots whose `block_alloc[i] == 1`. Returns -1 on null
// handle or D2H copy failure.
//
// Used by the host to decide whether a hash-table reset is safe
// (active_count == 0 ⇒ no slot in `block_alloc` references any hash
// entry; all entries in the table are tombstones, safe to wipe).
extern "C" int tsdf_hash_active_count(TsdfHash* h) {
    if (!h) return -1;
    int high_water = 0;
    int free_depth = 0;
    if (cudaMemcpyAsync(&high_water, h->d_block_alloc_counter, sizeof(int),
                        cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) return -1;
    if (cudaMemcpyAsync(&free_depth, h->d_free_count, sizeof(int),
                        cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(h->stream) != cudaSuccess) return -1;
    if (high_water > h->pool_capacity) high_water = h->pool_capacity;
    int active = high_water - free_depth;
    if (active < 0) active = 0;
    return active;
}

extern "C" unsigned long long tsdf_hash_drop_count(TsdfHash* h) {
    if (!h) return 0;
    unsigned long long c = 0;
    if (cudaMemcpyAsync(&c, h->d_n_dropped_pool_full, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) return 0;
    if (cudaStreamSynchronize(h->stream) != cudaSuccess) return 0;
    return c;
}

// Reset the hash table back to all-empty + counters to zero. SAFE only
// when no live blocks exist in the pool (caller's responsibility — run
// after a full-evict sweep that frees every block). Without this,
// tombstones from repeated eviction accumulate and saturate the hash
// table after ~3-4 chunks of pool churn, causing every subsequent
// CAS-claim probe to terminate with no EMPTY slot found.
extern "C" int tsdf_hash_reset_table(TsdfHash* h) {
    if (!h) return -1;
    int threads = 256;
    int blocks = (h->hash_size + threads - 1) / threads;
    init_hash_table_kernel<<<blocks, threads, 0, h->stream>>>(h->d_hash_table, h->hash_size);
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[tsdf_hash.cu] reset_table launch error: %s\n",
                cudaGetErrorString(kerr));
        return -1;
    }
    // High-water counter and free-stack depth back to zero. Free-stack
    // contents become irrelevant because the hash table no longer
    // references any of those indices.
    cudaMemsetAsync(h->d_block_alloc_counter, 0, sizeof(int), h->stream);
    cudaMemsetAsync(h->d_free_count, 0, sizeof(int), h->stream);
    cudaStreamSynchronize(h->stream);
    return 0;
}

// LRU eviction. Frees blocks not touched since `evict_before_chunk` and
// spills their converged voxels into the caller's output buffer.
//
// Returns:
//   >= 0   : number of voxels written to (out_xyz, out_rgb). If equal
//            to out_cap, the spill buffer overflowed and the caller
//            should retry with a larger cap (some voxels were dropped).
//   < 0    : launch / copy error.
//
// `n_evicted_blocks_out` (optional) receives the number of blocks
// freed. Useful for diagnostics: a chunk that evicts 0 means the
// keep-window is wider than the active scan trail.
extern "C" int tsdf_hash_evict_older_than(
    TsdfHash* h,
    int evict_before_chunk,
    float* out_xyz, unsigned char* out_rgb, int out_cap,
    int* n_evicted_blocks_out
) {
    return tsdf_hash_evict_older_than_target(
        h, evict_before_chunk, out_xyz, out_rgb, out_cap,
        n_evicted_blocks_out, /*keep_thresh_256*/ 0);
}

extern "C" int tsdf_hash_evict_older_than_target(
    TsdfHash* h,
    int evict_before_chunk,
    float* out_xyz, unsigned char* out_rgb, int out_cap,
    int* n_evicted_blocks_out,
    int keep_thresh_256
) {
    if (!h || !out_xyz || !out_rgb || out_cap <= 0) return -1;

    // Device-side counters for this eviction pass.
    unsigned int* d_out_cnt = nullptr;
    int*          d_n_blocks = nullptr;
    float*         d_out_xyz = nullptr;
    unsigned char* d_out_rgb = nullptr;

    if (cudaMalloc(&d_out_xyz, (size_t)out_cap * 3 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_out_rgb, (size_t)out_cap * 3 * sizeof(unsigned char)) != cudaSuccess) {
        cudaFree(d_out_xyz); return -1;
    }
    if (cudaMalloc(&d_out_cnt, sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(d_out_xyz); cudaFree(d_out_rgb); return -1;
    }
    if (cudaMalloc(&d_n_blocks, sizeof(int)) != cudaSuccess) {
        cudaFree(d_out_xyz); cudaFree(d_out_rgb); cudaFree(d_out_cnt); return -1;
    }
    cudaMemsetAsync(d_out_cnt, 0, sizeof(unsigned int), h->stream);
    cudaMemsetAsync(d_n_blocks, 0, sizeof(int), h->stream);

    // Grid dim = pool_capacity (one CUDA block per pool block). Block
    // dim = BD3 = 512 (one thread per voxel in the block).
    hash_evict_kernel<<<h->pool_capacity, BD3, 0, h->stream>>>(
        h->pool_capacity,
        evict_before_chunk,
        h->d_block_alloc,
        h->d_last_touched_chunk,
        h->d_block_coord,
        h->d_block_level,
        h->d_hash_table, h->hash_size,
        h->voxel_size_m,
        h->d_sum_x, h->d_sum_y, h->d_sum_z,
        h->d_sum_r, h->d_sum_g, h->d_sum_b,
        h->d_weight, h->d_count,
        h->d_free_stack, h->d_free_count,
        d_n_blocks,
        d_out_xyz, d_out_rgb, d_out_cnt, out_cap,
        keep_thresh_256
    );
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[tsdf_hash.cu] evict kernel launch error: %s\n",
                cudaGetErrorString(kerr));
        cudaFree(d_out_xyz); cudaFree(d_out_rgb);
        cudaFree(d_out_cnt); cudaFree(d_n_blocks);
        return -1;
    }
    cudaStreamSynchronize(h->stream);

    unsigned int final_count = 0;
    int n_blocks_h = 0;
    cudaMemcpyAsync(&final_count, d_out_cnt, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, h->stream);
    cudaMemcpyAsync(&n_blocks_h, d_n_blocks, sizeof(int),
                    cudaMemcpyDeviceToHost, h->stream);
    cudaStreamSynchronize(h->stream);

    int written = (int)final_count;
    if (written > out_cap) written = out_cap;  // overflow → caller retries
    if (written > 0) {
        cudaMemcpyAsync(out_xyz, d_out_xyz,
                   (size_t)written * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost, h->stream);
        cudaMemcpyAsync(out_rgb, d_out_rgb,
                   (size_t)written * 3 * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost, h->stream);
        cudaStreamSynchronize(h->stream);
    }

    cudaFree(d_out_xyz); cudaFree(d_out_rgb);
    cudaFree(d_out_cnt); cudaFree(d_n_blocks);

    if (n_evicted_blocks_out) *n_evicted_blocks_out = n_blocks_h;

    // If we hit the cap, the host should grow the buffer and retry —
    // signal by returning out_cap exactly (caller compares to its cap).
    return (int)final_count > out_cap ? out_cap : (int)final_count;
}
