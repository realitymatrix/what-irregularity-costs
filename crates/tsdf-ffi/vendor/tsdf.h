// tsdf.h
//
// GPU-resident sliding-window TSDF for progressive reconstruction. The grid is
// a dense 3D voxel array of fixed extent (typically 512 voxels per side at
// 1 mm resolution = 0.5 m cube). As the camera moves, the grid origin shifts
// and voxels leaving the window are streamed into a device-side committed
// buffer for later extraction. Integration of new points is an atomic-add
// kernel; extraction is a gate + compact kernel.
//
// Memory layout per voxel (struct of arrays — better atomic throughput than
// AoS under random access):
//   sum_x, sum_y, sum_z : f32   (weighted position sums)
//   sum_r, sum_g, sum_b : f32   (weighted color sums)
//   weight              : f32   (sum of per-point weights)
//   count               : u32   (distinct observation count)
//
// Footprint at 512³:
//   8 fields × (f32 or u32 = 4 bytes) × 512³ = 1 GB
// Plus a committed buffer (CPU-visible) for voxels that slid out of the
// window, capped at `committed_cap_points` Omnixels.
//
// Concurrency model: all grid state lives on the device. The host calls
// add_points / set_focus / extract_points sequentially per mission; there's
// no multi-stream support in v0.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TsdfGrid TsdfGrid;

// Create a TSDF grid with a dense voxel window.
//   voxel_size_m       : edge length of one voxel in meters (e.g. 0.001)
//   grid_dim           : voxels per side (e.g. 512 → 512³ = ~134 M voxels)
//   commit_distance_m  : voxels beyond this L-inf distance from the current
//                        focus point (set via tsdf_grid_set_focus) are
//                        considered "slid out of the window" on the next
//                        set_focus call and get copied into the committed
//                        buffer before the window shifts.
//   committed_cap_points : maximum number of Omnixels the committed buffer
//                        can hold at once. tsdf_grid_extract_points with
//                        drain=1 will empty this buffer.
//
// Returns NULL on allocation failure (e.g., VRAM insufficient for the grid).
TsdfGrid* tsdf_grid_create(
    float voxel_size_m,
    int grid_dim,
    float commit_distance_m,
    int committed_cap_points
);

// Release all device memory held by the grid. Safe to call with NULL.
void tsdf_grid_destroy(TsdfGrid* g);

// Add n_points points to the grid. Positions are in world coordinates (same
// frame the focus point is expressed in). Points outside the current window
// are silently dropped. Points inside the window atomically accumulate into
// their voxel's (sum_x, sum_y, sum_z, sum_r, sum_g, sum_b, weight, count).
//
//   positions : interleaved [x0,y0,z0, x1,y1,z1, ...] — 3*n_points floats
//   colors    : interleaved [r0,g0,b0, ...] — 3*n_points bytes. Pass NULL to
//               use a neutral grey (128,128,128) for all points.
//   weights   : per-point weight — n_points floats. Pass NULL to use 1.0
//               for all points.
//
// Returns 0 on success, nonzero if the copy/launch failed.
int tsdf_grid_add_points(
    TsdfGrid* g,
    const float* positions,
    const unsigned char* colors,
    const float* weights,
    int n_points
);

// Set the camera/scene focus point. Voxels more than `commit_distance_m`
// (L-inf) away from the focus are moved to the committed buffer and cleared
// from the active grid. The grid origin shifts so the new focus remains near
// the center of the active window.
//
// Returns the number of voxels committed (may be 0 if the focus moved less
// than one voxel), or a negative value on error (including committed buffer
// overflow — in that case the overflowing voxels are DROPPED, not retained).
int tsdf_grid_set_focus(TsdfGrid* g, float cx, float cy, float cz);

// Extract voxels into a host-side Omnixel buffer.
//   buffer_xyz      : out, 3*cap floats — position per voxel (weighted mean)
//   buffer_rgb      : out, 3*cap bytes — color per voxel (weighted mean)
//   buffer_cap      : max Omnixels to write
//   min_weight      : drop voxels with weight < this (legacy "≥N observations")
//   min_count       : drop voxels with count < this (distinct-obs gate)
//   max_spread_frac : drop voxels with per-axis stddev > voxel_size × this
//                     (bimodal-within-voxel detector). 1.0 = disable.
//   drain_committed : 1 = emit committed + active voxels, clear committed
//                     buffer. 0 = emit only currently-active voxels.
//
// Returns number of Omnixels written (>= 0), or -1 if buffer_cap insufficient.
// If -1, call with a larger buffer.
int tsdf_grid_extract_points(
    TsdfGrid* g,
    float* buffer_xyz,
    unsigned char* buffer_rgb,
    int buffer_cap,
    float min_weight,
    int min_count,
    float max_spread_frac,
    int drain_committed
);

// Diagnostic: total number of voxels in the active grid with weight > 0.
// Useful for logging + deciding how large an extract buffer to allocate.
int tsdf_grid_active_voxel_count(TsdfGrid* g);

// Diagnostic: number of committed Omnixels currently queued, waiting for
// a drain-mode extract to move them to the host.
int tsdf_grid_committed_count(TsdfGrid* g);

// Cap the per-voxel accumulated weight. Once weight_buf[idx] >= cap,
// integrate_points_kernel skips further accumulation into that voxel.
// Decreases TSDF "overlap" in already-confident regions: high-confidence
// surface voxels stop absorbing later (slightly drift-rotated) chunks
// that would otherwise smear them. A value <= 0 disables the cap
// (matches v0 behaviour). Sensible range: 8.0–32.0 for DA3-confidence
// weighted points (per-point weight ~ O(1)).
//
// Returns 0 on success, nonzero on null grid.
int tsdf_grid_set_weight_cap(TsdfGrid* g, float cap);

// Pin this grid's window origin to fixed world-voxel-coordinates and
// suppress further sliding from tsdf_grid_set_focus. Used by the
// multi-tile manager: each tile occupies a fixed sub-volume of the
// active cube, and shifts (when the cube as a whole moves) are owned
// by the manager rather than per-tile set_focus. After this call:
//   - tsdf_grid_set_focus returns 0 immediately without sliding.
//   - tsdf_grid_add_points still drops points outside this window.
// The arguments are world-voxel coordinates of the tile's voxel (0,0,0),
// i.e. world-coord = origin_v* * voxel_size_m for the tile corner.
//
// Returns 0 on success, -1 on null grid.
int tsdf_grid_set_origin(TsdfGrid* g, int origin_vx, int origin_vy, int origin_vz);

// Drain every active voxel into the per-grid committed buffer (using
// min_weight=1e-6 so any voxel that observed at least one point is
// emitted), zero the active voxel buffers, then set a fresh origin.
// Used by the multi-tile manager to slide the cube: trailing tiles
// drain and re-pin onto the leading edge in O(grid_dim^3) per tile.
// Subsequent extract calls (with drain_committed=1) deliver the
// drained voxels to host alongside whatever's currently active.
//
// Returns the number of voxels committed by this call (>=0), or -1
// on error.
int tsdf_grid_drain_and_reset(TsdfGrid* g, int new_origin_vx, int new_origin_vy, int new_origin_vz);

#ifdef __cplusplus
}  // extern "C"
#endif
