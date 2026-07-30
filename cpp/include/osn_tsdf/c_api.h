// C ABI for the CPU TSDF arm (A2).
//
// A C surface, not C++, so every arm can be bound from Rust through one
// mechanism. Keeping the surface identical across arms is what stops the
// harness from accidentally exercising a different code path per arm.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OsnTsdfVolume OsnTsdfVolume;

/// Returns NULL on invalid parameters or allocation failure.
/// `trunc_m <= 0` selects 4 * voxel_size_m. `weight_cap <= 0` disables capping.
OsnTsdfVolume* osn_tsdf_create(float voxel_size_m, int32_t pool_capacity_blocks,
                               float trunc_m, float weight_cap);

void osn_tsdf_destroy(OsnTsdfVolume* v);

/// 0 on success, nonzero on error. Points are host-side, world-space.
/// `cam` is required: the sign of a truncated signed distance is only defined
/// relative to a viewpoint.
int32_t osn_tsdf_integrate(OsnTsdfVolume* v, const float* positions, const uint8_t* colors,
                           const float* weights, int32_t n_points, int32_t chunk_id,
                           float cam_x, float cam_y, float cam_z, float radius_m);

/// Allocation and update as separate passes, so arm A5a can run a Triton
/// update over blocks allocated elsewhere.
int32_t osn_tsdf_allocate_blocks(OsnTsdfVolume* v, const float* positions, int32_t n_points,
                                 float cam_x, float cam_y, float cam_z, float radius_m);
int32_t osn_tsdf_update_voxels(OsnTsdfVolume* v, const float* positions, const uint8_t* colors,
                               const float* weights, int32_t n_points, int32_t chunk_id,
                               float cam_x, float cam_y, float cam_z, float radius_m);

/// Returns the vertex count, or -1 if `vert_cap` was too small.
/// `buffer_posnor` holds vert_cap*6 floats, `buffer_rgb` vert_cap*3 bytes (may be NULL).
int32_t osn_tsdf_extract_mesh(OsnTsdfVolume* v, float* buffer_posnor, uint8_t* buffer_rgb,
                              int32_t vert_cap, float min_weight, float iso);

int32_t osn_tsdf_block_count(const OsnTsdfVolume* v);
uint64_t osn_tsdf_drop_count(const OsnTsdfVolume* v);
void osn_tsdf_set_thread_count(OsnTsdfVolume* v, int32_t n);
int32_t osn_tsdf_thread_count(const OsnTsdfVolume* v);

#ifdef __cplusplus
}
#endif
