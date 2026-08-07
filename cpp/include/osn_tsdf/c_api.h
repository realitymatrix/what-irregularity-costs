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

/* ---- CUDA arm (A3) and shared device state -------------------------------
 *
 * The device view exists so other arms can run THEIR kernels against state this
 * library allocated, which is what "arms differ in the integrate path, and
 * extraction is shared" means in practice. A4 (Rust) and A5 (Triton) allocate
 * nothing of their own; they operate on these pointers and then hand the volume
 * back for extraction. See docs/ARM-SCOPE.md.
 */
typedef struct OsnTsdfCudaVolume OsnTsdfCudaVolume;

/// Device pointers and parameters describing a CUDA volume's storage.
/// All pointers are device-resident. Layout must match what every arm assumes:
/// the hash table is (int64 key, int32 block_idx, int32 pad) per slot.
typedef struct {
    void* table;          /* hash_size entries, 16 bytes each */
    void* block_count;    /* int32 */
    void* drop_count;     /* uint64 */
    void* block_coord;    /* int32[3] per pool slot */
    void* tsdf;           /* float, weighted sum */
    void* weight;         /* float */
    void* r;
    void* g;
    void* b;
    uint32_t hash_mask;
    uint32_t scratch_base;  /* first scratch slot; see Triton arm */
    int32_t pool_capacity;
    int32_t block_dim;
    float voxel_size_m;
    float trunc_m;
    float weight_cap;
} OsnTsdfDeviceView;

OsnTsdfCudaVolume* osn_tsdf_cuda_create(float voxel_size_m, int32_t pool_capacity_blocks,
                                        float trunc_m, float weight_cap);
void osn_tsdf_cuda_destroy(OsnTsdfCudaVolume* v);
int32_t osn_tsdf_cuda_valid(const OsnTsdfCudaVolume* v);

/// Fill `out` with the device pointers backing this volume.
int32_t osn_tsdf_cuda_device_view(const OsnTsdfCudaVolume* v, OsnTsdfDeviceView* out);

/// A3's own kernels. `positions` is a DEVICE pointer.
int32_t osn_tsdf_cuda_allocate_blocks(OsnTsdfCudaVolume* v, const float* d_positions,
                                      int32_t n_points, float cam_x, float cam_y, float cam_z,
                                      float radius_m);
int32_t osn_tsdf_cuda_update_voxels(OsnTsdfCudaVolume* v, const float* d_positions,
                                    int32_t n_points, float cam_x, float cam_y, float cam_z,
                                    float radius_m);

/// Shared extraction. Host output buffers.
int32_t osn_tsdf_cuda_extract_mesh(OsnTsdfCudaVolume* v, float* buffer_posnor, uint8_t* buffer_rgb,
                                   int32_t vert_cap, float min_weight, float iso);

/// Clear the volume to its just-constructed state without reallocating.
///
/// Required by the benchmark harness: allocation can only be measured from an
/// empty table, and reconstructing the volume per repetition would put ~1 GiB
/// of cudaMalloc and memset inside the measurement. Resetting is a memset of
/// the table plus the counters, which is untimed between repetitions.
int32_t osn_tsdf_cuda_reset(OsnTsdfCudaVolume* v);

/// MEASUREMENT ONLY: allocate, counting compare-exchange ATTEMPTS into the
/// drop counter. Only valid on a pool large enough that nothing is dropped.
int32_t osn_tsdf_cuda_allocate_counting_cas(OsnTsdfCudaVolume* v, const float* d_positions,
                                            int32_t n_points, float cam_x, float cam_y,
                                            float cam_z, float radius_m);

/// MEASUREMENT ONLY: probe and read, never insert. Mirrors arm A4's `-cas`.
int32_t osn_tsdf_cuda_allocate_no_cas(OsnTsdfCudaVolume* v, const float* d_positions,
                                      int32_t n_points, float cam_x, float cam_y, float cam_z,
                                      float radius_m);

/// MEASUREMENT ONLY: block index derived from the hash slot; no spin.
int32_t osn_tsdf_cuda_allocate_slot_index(OsnTsdfCudaVolume* v, const float* d_positions,
                                          int32_t n_points, float cam_x, float cam_y, float cam_z,
                                          float radius_m);

int32_t osn_tsdf_cuda_block_count(const OsnTsdfCudaVolume* v);
uint64_t osn_tsdf_cuda_drop_count(const OsnTsdfCudaVolume* v);
void osn_tsdf_cuda_synchronize(const OsnTsdfCudaVolume* v);

#ifdef __cplusplus
}
#endif
